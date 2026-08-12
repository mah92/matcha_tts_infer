#include "normalize.h"
#include "language_detector/language_detector.h"

#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <locale>
#include <codecvt>

// POSIX sockets + process control
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>

// ============================================================================
// Global socket path
// ============================================================================
static const char* SOCKET_PATH = "/tmp/tts_infer.sock";

// ============================================================================
// WAV Writer: 16-bit PCM, mono
// ============================================================================
static void write_wav(const std::string& path, const std::vector<float>& audio, int sample_rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Cannot open output file: " << path << std::endl;
        return;
    }

    int num_samples = static_cast<int>(audio.size());
    int num_channels = 1;
    int bits_per_sample = 16;
    int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int block_align = num_channels * bits_per_sample / 8;
    int data_size = num_samples * block_align;
    int chunk_size = 36 + data_size;

    // RIFF header
    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&chunk_size), 4);
    out.write("WAVE", 4);

    // fmt subchunk
    out.write("fmt ", 4);
    int fmt_size = 16;
    short audio_format = 1; // PCM
    out.write(reinterpret_cast<const char*>(&fmt_size), 4);
    out.write(reinterpret_cast<const char*>(&audio_format), 2);
    out.write(reinterpret_cast<const char*>(&num_channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    out.write(reinterpret_cast<const char*>(&block_align), 2);
    out.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    // data subchunk
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), 4);

    // Convert float [-1,1] to int16
    for (float sample : audio) {
        sample = std::max(-1.0f, std::min(1.0f, sample));
        int16_t int_sample = static_cast<int16_t>(sample * 32767.0f);
        out.write(reinterpret_cast<const char*>(&int_sample), 2);
    }

    out.close();
}

// ============================================================================
// Token map reader
// ============================================================================
static int read_token_map(const std::string& filepath,
                          std::map<std::string, int>& token_to_id,
                          std::map<int, std::string>& id_to_token) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot open token file: " << filepath << std::endl;
        return -1;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;

        auto last_space = line.rfind(' ');
        if (last_space == std::string::npos) continue;

        std::string token = line.substr(0, last_space);
        std::string id_str = line.substr(last_space + 1);

        int id = std::stoi(id_str);
        token_to_id[token] = id;
        id_to_token[id] = token;
    }

    f.close();
    std::cout << "Loaded " << token_to_id.size() << " tokens from " << filepath << std::endl;
    return 0;
}

// ============================================================================
// Convert IPA string to token IDs
// ============================================================================
static std::vector<int64_t> ipa_to_ids(const std::string& ipa_text,
                                        const std::map<std::string, int>& token_to_id) {
    std::vector<int64_t> ids;

    std::string remaining = ipa_text;
    while (!remaining.empty()) {
        bool found = false;

        for (int len = std::min(8, (int)remaining.size()); len >= 1; --len) {
            std::string candidate = remaining.substr(0, len);
            auto it = token_to_id.find(candidate);
            if (it != token_to_id.end()) {
                ids.push_back(static_cast<int64_t>(it->second));
                remaining = remaining.substr(len);
                found = true;
                break;
            }
        }

        if (!found) {
            unsigned char c = static_cast<unsigned char>(remaining[0]);
            std::cerr << "Warning: Unknown token '" << remaining[0]
                      << "' (0x" << std::hex << (int)c << std::dec << ") — skipping" << std::endl;
            remaining = remaining.substr(1);
        }
    }

    return ids;
}

// ============================================================================
// Intersperse blanks
// ============================================================================
static std::vector<int64_t> intersperse_blanks(const std::vector<int64_t>& ids, int64_t blank_id) {
    size_t n = ids.size();
    std::vector<int64_t> result(n * 2 + 1, blank_id);
    for (size_t i = 0; i < n; ++i) {
        result[i * 2 + 1] = ids[i];
    }
    return result;
}

// ============================================================================
// JSON helpers (minimal — no library dependency)
// ============================================================================
static std::string json_esc(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

static std::string json_str(const std::string& s) {
    return "\"" + json_esc(s) + "\"";
}

// Simple JSON value extractor: find "key":"value" or "key":number
static std::string json_get_str(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (pos >= json.size()) return "";

    if (json[pos] == '"') {
        pos++;
        std::string val;
        while (pos < json.size()) {
            if (json[pos] == '\\') {
                pos++;
                if (pos < json.size()) val += json[pos];
            } else if (json[pos] == '"') {
                break;
            } else {
                val += json[pos];
            }
            pos++;
        }
        return val;
    }

    if (json[pos] == '-' || (json[pos] >= '0' && json[pos] <= '9')) {
        size_t end = pos;
        while (end < json.size() && (json[end] == '-' || json[end] == '.' ||
               (json[end] >= '0' && json[end] <= '9'))) end++;
        return json.substr(pos, end - pos);
    }

    return "";
}

// ============================================================================
// Config and result types
// ============================================================================
struct SynthConfig {
    std::string matcha_model;
    std::string vocoder_model;
    std::string tokens_file;
    std::string espeak_data;
    std::string ezafe_onnx;
    std::string ezafe_spiece;
    std::string hazm_words;
    std::string hazm_verbs;
    std::string hazm_stopwords;
    std::string homograph_data;
    std::string shakkelha_onnx;
    std::string main_lang_str = "FA";
    float temperature = 0.667f;
    float speed = 1.0f;
    int sample_rate = 22050;
    bool use_gpu = false;
};

struct SynthResult {
    bool ok = false;
    std::string error;
    std::string output_path;
    double duration_secs = 0;
    double norm_ms = 0;
    double matcha_ms = 0;
    double vocos_ms = 0;
    double total_ms = 0;
};

// ============================================================================
// Global state (loaded once in daemon, once per run in direct mode)
// ============================================================================
static std::unique_ptr<Ort::Env> g_env;
static Ort::SessionOptions g_session_opts;
static Ort::MemoryInfo g_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
static bool g_models_loaded = false;

static std::map<std::string, int> g_token_to_id;
static std::map<int, std::string> g_id_to_token;
static NormalizeConfig g_norm_config;
static Language g_mainlang;

static std::unique_ptr<Ort::Session> g_matcha_session;
static std::unique_ptr<Ort::Session> g_vocoder_session;

static bool load_all_models(const SynthConfig& cfg) {
    if (g_models_loaded) return true;

    auto t0 = std::chrono::high_resolution_clock::now();

    g_env.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "tts_infer"));
    g_session_opts.SetIntraOpNumThreads(4);
    g_session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (read_token_map(cfg.tokens_file, g_token_to_id, g_id_to_token) != 0) {
        return false;
    }

    std::cout << "Loading Matcha model: " << cfg.matcha_model << std::endl;
    try {
        g_matcha_session.reset(new Ort::Session(*g_env, cfg.matcha_model.c_str(), g_session_opts));
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load Matcha model: " << e.what() << std::endl;
        return false;
    }

    std::cout << "Loading Vocoder model: " << cfg.vocoder_model << std::endl;
    try {
        g_vocoder_session.reset(new Ort::Session(*g_env, cfg.vocoder_model.c_str(), g_session_opts));
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load Vocoder model: " << e.what() << std::endl;
        return false;
    }

    g_norm_config.espeak_data_path = cfg.espeak_data;
    g_norm_config.shakkelha_onnx   = cfg.shakkelha_onnx;
    g_norm_config.ezafe_model_onnx = cfg.ezafe_onnx;
    g_norm_config.ezafe_model_spiece = cfg.ezafe_spiece;
    g_norm_config.hazm_words       = cfg.hazm_words;
    g_norm_config.hazm_verbs       = cfg.hazm_verbs;
    g_norm_config.hazm_stopwords   = cfg.hazm_stopwords;
    g_norm_config.homograph_data   = cfg.homograph_data;

    g_mainlang = LanguageDetector::string_to_language(cfg.main_lang_str);

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "All models loaded in " << load_secs << "s" << std::endl;

    g_models_loaded = true;
    return true;
}

// ============================================================================
// Core synthesis
// ============================================================================
static SynthResult synthesize(const SynthConfig& cfg, const std::string& text,
                               const std::string& output_path) {
    SynthResult result;
    auto t_total_start = std::chrono::high_resolution_clock::now();

    std::string normalized_text;
    std::string ipa_text;

    std::cout << "Normalizing text: " << text << std::endl;
    auto t_norm_start = std::chrono::high_resolution_clock::now();
    normalizeString(g_mainlang, 1 /* IPA mode */, text,
                    normalized_text, ipa_text, g_norm_config);
    auto t_norm_end = std::chrono::high_resolution_clock::now();

    ipa_text.erase(std::remove(ipa_text.begin(), ipa_text.end(), '\n'), ipa_text.end());

    std::cout << "IPA phonemes: " << ipa_text << std::endl;
    std::cout << "Normalized:   " << normalized_text << std::endl;

    result.norm_ms = std::chrono::duration<double, std::milli>(t_norm_end - t_norm_start).count();
    fprintf(stderr, "[TIMING] NormalizeText : %.1f ms\n", result.norm_ms);

    auto token_ids = ipa_to_ids(ipa_text, g_token_to_id);
    auto ids_with_blanks = intersperse_blanks(token_ids, 0);
    int64_t seq_len = static_cast<int64_t>(ids_with_blanks.size());
    std::cout << "Tokens: " << token_ids.size() << ", with blanks: " << seq_len << std::endl;

    // --- Matcha-TTS inference ---
    std::cout << "\n=== Matcha-TTS Inference ===" << std::endl;

    float speaking_rate = 1.0f / cfg.speed;

    std::vector<int64_t> x_shape = {1, seq_len};
    std::vector<int64_t> x_lengths_data = {seq_len};
    std::vector<float> scales_data = {cfg.temperature, speaking_rate};

    std::cout << "Temperature: " << cfg.temperature << ", Speed: " << cfg.speed << "x" << std::endl;

    Ort::Value x_tensor = Ort::Value::CreateTensor<int64_t>(
        g_mem_info, ids_with_blanks.data(), ids_with_blanks.size(),
        x_shape.data(), x_shape.size());
    Ort::Value x_lengths_tensor = Ort::Value::CreateTensor<int64_t>(
        g_mem_info, x_lengths_data.data(), 1,
        (std::vector<int64_t>{1}).data(), 1);
    Ort::Value scales_tensor = Ort::Value::CreateTensor<float>(
        g_mem_info, scales_data.data(), 2,
        (std::vector<int64_t>{2}).data(), 1);

    const char* matcha_input_names[] = {"x", "x_lengths", "scales"};
    const char* matcha_output_names[] = {"mel", "mel_lengths"};

    std::vector<Ort::Value> matcha_inputs;
    matcha_inputs.push_back(std::move(x_tensor));
    matcha_inputs.push_back(std::move(x_lengths_tensor));
    matcha_inputs.push_back(std::move(scales_tensor));

    std::cout << "Running Matcha inference..." << std::endl;
    auto t_matcha_start = std::chrono::high_resolution_clock::now();
    auto matcha_outputs = g_matcha_session->Run(
        Ort::RunOptions{nullptr},
        matcha_input_names, matcha_inputs.data(), matcha_inputs.size(),
        matcha_output_names, 2);
    auto t_matcha_end = std::chrono::high_resolution_clock::now();
    result.matcha_ms = std::chrono::duration<double, std::milli>(t_matcha_end - t_matcha_start).count();
    std::cout << "Matcha inference done in " << (result.matcha_ms / 1000.0) << "s" << std::endl;
    fprintf(stderr, "[TIMING] MatchaTTS      : %.1f ms\n", result.matcha_ms);

    float* mel_data = matcha_outputs[0].GetTensorMutableData<float>();
    auto mel_info = matcha_outputs[0].GetTensorTypeAndShapeInfo();
    auto mel_shape = mel_info.GetShape();
    int64_t mel_frames = mel_shape[2];
    std::cout << "Mel shape: [" << mel_shape[0] << ", " << mel_shape[1] << ", " << mel_shape[2] << "]" << std::endl;

    int64_t* mel_lengths_data_ptr = matcha_outputs[1].GetTensorMutableData<int64_t>();
    int64_t mel_len = mel_lengths_data_ptr[0];
    std::cout << "Mel length: " << mel_len << " frames" << std::endl;

    // --- Vocos vocoder inference ---
    std::cout << "\n=== Vocoder Inference ===" << std::endl;

    std::vector<int64_t> mel_input_shape = {1, 80, mel_frames};
    std::vector<float> denoise_data = {0.0f};

    Ort::Value mel_vocoder_tensor = Ort::Value::CreateTensor<float>(
        g_mem_info, mel_data, mel_shape[0] * mel_shape[1] * mel_shape[2],
        mel_input_shape.data(), mel_input_shape.size());
    Ort::Value denoise_tensor = Ort::Value::CreateTensor<float>(
        g_mem_info, denoise_data.data(), 1,
        (std::vector<int64_t>{1}).data(), 1);

    const char* vocoder_input_names[] = {"mel_spec", "denoise"};
    const char* vocoder_output_names[] = {"wave"};

    std::vector<Ort::Value> vocoder_inputs;
    vocoder_inputs.push_back(std::move(mel_vocoder_tensor));
    vocoder_inputs.push_back(std::move(denoise_tensor));

    std::cout << "Running vocoder inference..." << std::endl;
    auto t_vocos_start = std::chrono::high_resolution_clock::now();
    auto vocoder_outputs = g_vocoder_session->Run(
        Ort::RunOptions{nullptr},
        vocoder_input_names, vocoder_inputs.data(), vocoder_inputs.size(),
        vocoder_output_names, 1);
    auto t_vocos_end = std::chrono::high_resolution_clock::now();
    result.vocos_ms = std::chrono::duration<double, std::milli>(t_vocos_end - t_vocos_start).count();
    std::cout << "Vocoder inference done in " << (result.vocos_ms / 1000.0) << "s" << std::endl;
    fprintf(stderr, "[TIMING] Vocos          : %.1f ms\n", result.vocos_ms);

    float* wave_data = vocoder_outputs[0].GetTensorMutableData<float>();
    auto wave_info = vocoder_outputs[0].GetTensorTypeAndShapeInfo();
    auto wave_shape = wave_info.GetShape();
    int64_t num_samples = wave_shape[1];
    std::cout << "Wave shape: [" << wave_shape[0] << ", " << num_samples << "]" << std::endl;

    std::vector<float> audio(wave_data, wave_data + num_samples);
    std::string actual_output = output_path.empty() ? "output.wav" : output_path;
    write_wav(actual_output, audio, cfg.sample_rate);

    result.ok = true;
    result.output_path = actual_output;
    result.duration_secs = num_samples / (double)cfg.sample_rate;
    result.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_total_start).count();

    std::cout << "\n=== Done ===" << std::endl;
    std::cout << "Output: " << actual_output << std::endl;
    std::cout << "Duration: " << result.duration_secs << "s" << std::endl;

    fprintf(stderr, "\n[TIMING] ===== Summary =====\n");
    fprintf(stderr, "[TIMING] NormalizeText : %.1f ms\n", result.norm_ms);
    fprintf(stderr, "[TIMING] MatchaTTS      : %.1f ms\n", result.matcha_ms);
    fprintf(stderr, "[TIMING] Vocos          : %.1f ms\n", result.vocos_ms);
    fprintf(stderr, "[TIMING] Total           : %.1f ms\n", result.total_ms);

    return result;
}

// ============================================================================
// Unix socket helpers
// ============================================================================
static ssize_t write_all(int fd, const void* buf, size_t len) {
    size_t total = 0;
    const char* ptr = static_cast<const char*>(buf);
    while (total < len) {
        ssize_t n = write(fd, ptr + total, len - total);
        if (n <= 0) return n;
        total += n;
    }
    return total;
}

static std::string read_line(int fd, int timeout_secs = 10) {
    struct timeval tv;
    tv.tv_sec = timeout_secs;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    std::string line;
    char c;
    while (true) {
        ssize_t n = read(fd, &c, 1);
        if (n <= 0) break;
        if (c == '\n') break;
        line += c;
    }
    return line;
}

static bool write_line(int fd, const std::string& line) {
    std::string msg = line + "\n";
    return write_all(fd, msg.data(), msg.size()) > 0;
}

// ============================================================================
// DAEMON MODE
// ============================================================================
static int run_daemon(const SynthConfig& cfg) {
    signal(SIGPIPE, SIG_IGN);

    std::cout << "[DAEMON] Loading all models..." << std::endl;
    if (!load_all_models(cfg)) {
        std::cerr << "[DAEMON] Failed to load models" << std::endl;
        return 1;
    }

    unlink(SOCKET_PATH);

    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("[DAEMON] socket");
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("[DAEMON] bind");
        close(server_fd);
        return 1;
    }

    chmod(SOCKET_PATH, 0666);

    if (listen(server_fd, 5) < 0) {
        perror("[DAEMON] listen");
        close(server_fd);
        unlink(SOCKET_PATH);
        return 1;
    }

    std::cerr << "[DAEMON] Ready on " << SOCKET_PATH << std::endl;

    while (true) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) {
            perror("[DAEMON] accept");
            continue;
        }

        std::string request = read_line(client_fd, 30);
        if (request.empty()) {
            close(client_fd);
            continue;
        }

        std::string cmd = json_get_str(request, "command");
        if (cmd == "stop") {
            write_line(client_fd, "{\"status\":\"ok\",\"message\":\"shutting down\"}");
            close(client_fd);
            break;
        }

        std::string text = json_get_str(request, "text");
        if (text.empty()) {
            write_line(client_fd, "{\"status\":\"error\",\"message\":\"--text is required\"}");
            close(client_fd);
            continue;
        }

        SynthConfig req_cfg = cfg;
        std::string temp_str = json_get_str(request, "temperature");
        if (!temp_str.empty()) req_cfg.temperature = std::stof(temp_str);
        std::string speed_str = json_get_str(request, "speed");
        if (!speed_str.empty()) req_cfg.speed = std::stof(speed_str);
        std::string sr_str = json_get_str(request, "sample_rate");
        if (!sr_str.empty()) req_cfg.sample_rate = std::stoi(sr_str);
        std::string lang_str = json_get_str(request, "main_lang");
        if (!lang_str.empty()) req_cfg.main_lang_str = lang_str;

        std::string output = json_get_str(request, "output");
        if (output.empty()) output = "output.wav";

        SynthResult result = synthesize(req_cfg, text, output);

        char buf[4096];
        if (result.ok) {
            snprintf(buf, sizeof(buf),
                "{\"status\":\"ok\",\"output\":\"%s\",\"duration\":%.2f,\"norm_ms\":%.1f,\"matcha_ms\":%.1f,\"vocos_ms\":%.1f,\"total_ms\":%.1f}",
                json_esc(result.output_path).c_str(),
                result.duration_secs,
                result.norm_ms, result.matcha_ms, result.vocos_ms, result.total_ms);
        } else {
            snprintf(buf, sizeof(buf),
                "{\"status\":\"error\",\"message\":\"%s\"}",
                json_esc(result.error).c_str());
        }
        write_line(client_fd, std::string(buf));

        close(client_fd);
    }

    close(server_fd);
    unlink(SOCKET_PATH);
    std::cout << "[DAEMON] Stopped" << std::endl;
    return 0;
}

// ============================================================================
// STOP DAEMON
// ============================================================================
static int stop_daemon() {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket");
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "Daemon is not running (cannot connect to " << SOCKET_PATH << ")" << std::endl;
        close(fd);
        return 1;
    }

    write_line(fd, "{\"command\":\"stop\"}");
    std::string response = read_line(fd, 5);
    std::cout << "Daemon: " << response << std::endl;
    close(fd);
    return 0;
}

// ============================================================================
// CLIENT MODE
// ============================================================================
static bool daemon_is_running() {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    bool running = (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0);
    close(fd);
    return running;
}

static int run_client(const SynthConfig& cfg, const std::string& text,
                       const std::string& output_path, bool play_audio) {
    (void)cfg; // config is on the daemon side
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return 1; }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect"); close(fd); return 1;
    }
    std::string request = "{\"text\":" + json_str(text);
    if (!output_path.empty()) request += ",\"output\":" + json_str(output_path);
    request += "}";
    if (!write_line(fd, request)) { std::cerr << "Error: failed to send\n"; close(fd); return 1; }
    std::string response = read_line(fd, 30);
    close(fd);
    if (response.empty()) { std::cerr << "Error: no response\n"; return 1; }
    std::string st = json_get_str(response, "status");
    if (st == "ok") {
        std::string out = json_get_str(response, "output");
        std::cout << "Output: " << out << "\nDuration: " << json_get_str(response, "duration") << "s\n";
        if (play_audio && !out.empty()) {
            std::string cmd = "ffplay -nodisp -autoexit \"" + out + "\" 2>/dev/null";
            std::system(cmd.c_str());
        }
        return 0;
    }
    std::cerr << "Error from daemon: " << json_get_str(response, "message") << "\n";
    return 1;
}

// ============================================================================
// USAGE
// ============================================================================
static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [OPTIONS]\n"
              << "\n"
              << "  MODES:\n"
              << "    --daemon                    Start background daemon (loads models once, stays resident)\n"
              << "    --stop                      Stop the running daemon\n"
              << "\n"
              << "  SYNTHESIS OPTIONS:\n"
              << "    --text <text>               Text to synthesize (required unless --daemon/--stop)\n"
              << "    --matcha-model <path>       Matcha-TTS ONNX model path\n"
              << "    --vocoder-model <path>      Vocos vocoder ONNX model path\n"
              << "    --tokens <path>             Token file\n"
              << "    --espeak-data <path>        espeak-ng-data directory\n"
              << "    --ezafe-onnx <path>         Ezafe model ONNX path\n"
              << "    --ezafe-spiece <path>       Ezafe sentencepiece model path\n"
              << "    --hazm-words <path>         HAZM words.dat path\n"
              << "    --hazm-verbs <path>         HAZM verbs.dat path\n"
              << "    --hazm-stopwords <path>     HAZM stopwords.dat path\n"
              << "    --homograph <path>          Homograph data JSON path\n"
              << "    --shakkelha <path>          Shakkelha ONNX model path\n"
              << "    --output <path>             Output WAV file (default: output.wav)\n"
              << "    --play                      Play audio after generation\n"
              << "    --temperature <float>       Temperature (default: 0.667)\n"
              << "    --speed <float>             Speed: 1.0=1x, 1.5=1.5x, 2.0=2x (default: 1.0)\n"
              << "    --sample-rate <int>         Output sample rate (default: 22050)\n"
              << "    --main-lang <EN|FA|AR>      Main language (default: FA)\n"
              << "    --gpu                       Use GPU (default: CPU)\n"
              << "    --help                      Show this help\n"
              << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char* argv[]) {
    std::locale::global(std::locale("en_US.UTF-8"));

    // ----- Default paths -----
    SynthConfig cfg;
    cfg.matcha_model  = std::string(getenv("HOME")) + "/.hermes/skills/hermes-bale-messenger-skills/hermes-tts/models/matcha-fa_en-zahra-22050-5.onnx";
    cfg.vocoder_model = std::string(getenv("HOME")) + "/.hermes/skills/hermes-bale-messenger-skills/hermes-tts/models/vocos22.onnx";
    cfg.tokens_file   = std::string(getenv("HOME")) + "/.hermes/skills/hermes-bale-messenger-skills/hermes-tts/models/tokens_sherpa_with_fa.txt";
    cfg.espeak_data   = "/home/oem/Basir/TTS/Piper/piper_linux_x86_64/piper/espeak-ng-data";
    cfg.ezafe_onnx     = "./assets/ezafe_model.onnx";
    cfg.ezafe_spiece   = "./assets/ezafe_spiece.model";
    cfg.hazm_words     = "./assets/hazm_words.dat";
    cfg.hazm_verbs     = "./assets/hazm_verbs.dat";
    cfg.hazm_stopwords = "./assets/hazm_stopwords.dat";
    cfg.homograph_data = "./assets/homograph_data.json";
    cfg.shakkelha_onnx = "./assets/shakkelha.onnx";

    std::string text;
    std::string output_wav;
    bool play_audio = false;
    bool daemon_mode = false;
    bool stop_mode = false;

    // ----- Parse CLI args -----
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto require_val = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << name << " requires a value" << std::endl;
                exit(1);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--text")               text = require_val("--text");
        else if (arg == "--matcha-model")  cfg.matcha_model = require_val("--matcha-model");
        else if (arg == "--vocoder-model") cfg.vocoder_model = require_val("--vocoder-model");
        else if (arg == "--tokens")        cfg.tokens_file = require_val("--tokens");
        else if (arg == "--espeak-data")   cfg.espeak_data = require_val("--espeak-data");
        else if (arg == "--ezafe-onnx")    cfg.ezafe_onnx = require_val("--ezafe-onnx");
        else if (arg == "--ezafe-spiece")  cfg.ezafe_spiece = require_val("--ezafe-spiece");
        else if (arg == "--hazm-words")    cfg.hazm_words = require_val("--hazm-words");
        else if (arg == "--hazm-verbs")    cfg.hazm_verbs = require_val("--hazm-verbs");
        else if (arg == "--hazm-stopwords") cfg.hazm_stopwords = require_val("--hazm-stopwords");
        else if (arg == "--homograph")     cfg.homograph_data = require_val("--homograph");
        else if (arg == "--shakkelha")     cfg.shakkelha_onnx = require_val("--shakkelha");
        else if (arg == "--output")        output_wav = require_val("--output");
        else if (arg == "--play")          play_audio = true;
        else if (arg == "--temperature")   cfg.temperature = std::stof(require_val("--temperature"));
        else if (arg == "--speed")         cfg.speed = std::stof(require_val("--speed"));
        else if (arg == "--sample-rate")   cfg.sample_rate = std::stoi(require_val("--sample-rate"));
        else if (arg == "--main-lang")     cfg.main_lang_str = require_val("--main-lang");
        else if (arg == "--gpu")           cfg.use_gpu = true;
        else if (arg == "--daemon")        daemon_mode = true;
        else if (arg == "--stop")          stop_mode = true;
        else if (arg == "--help")          { print_usage(argv[0]); return 0; }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // ----- Dispatch -----
    if (daemon_mode) {
        return run_daemon(cfg);
    }

    if (stop_mode) {
        return stop_daemon();
    }

    // Default: use daemon (auto-start if not running)
    if (text.empty()) {
        std::cerr << "Error: --text is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (!daemon_is_running()) {
        // Auto-start daemon in background
        char self_path[4096];
        ssize_t len = readlink("/proc/self/exe", self_path, sizeof(self_path) - 1);
        if (len <= 0) {
            std::cerr << "Error: cannot determine own binary path" << std::endl;
            return 1;
        }
        self_path[len] = '\0';

        // Change to NormalizeText/ (sibling of build/)
        std::string bin_dir = self_path;
        auto last_slash = bin_dir.rfind('/');
        if (last_slash != std::string::npos) {
            bin_dir = bin_dir.substr(0, last_slash); // build/
        }
        std::string norm_dir = bin_dir + "/../NormalizeText";

        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0) {
            // Child: cd to NormalizeText, start daemon
            chdir(norm_dir.c_str());
            execl(self_path, self_path, "--daemon", nullptr);
            perror("execl");
            _exit(1);
        }

        // Wait for daemon to be ready
        for (int i = 0; i < 50; i++) {
            usleep(100000);
            if (daemon_is_running()) break;
        }
        if (!daemon_is_running()) {
            std::cerr << "Error: daemon failed to start" << std::endl;
            return 1;
        }
    }

    return run_client(cfg, text, output_wav, play_audio);
}
