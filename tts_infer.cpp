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
// Token map reader — handles space-separated "token ID" format (like tokens_sherpa_with_fa.txt)
// Line format: <token> <id>
// Special case: space token "  3" → the token is a literal space " "
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

        // Find the last space — the ID is everything after it
        // The token is everything before the last space
        auto last_space = line.rfind(' ');
        if (last_space == std::string::npos) continue;

        std::string token = line.substr(0, last_space);
        std::string id_str = line.substr(last_space + 1);

        // Handle edge case: token itself can be empty (space token " 3")
        // In the file this looks like "  3" → token=" " (one space), id=3
        // But our parsing gives token="" which is wrong
        // Actually: "  3" → rfind(' ') finds the second space, token = " ", id = "3"
        // Wait, let's trace: "  3" → rfind(' ') returns 1 (second space is at pos 1)
        // token = "  3".substr(0, 1) = " " ✓
        // id_str = "  3".substr(2) = "3" ✓

        int id = std::stoi(id_str);
        token_to_id[token] = id;
        id_to_token[id] = token;
    }

    f.close();
    std::cout << "Loaded " << token_to_id.size() << " tokens from " << filepath << std::endl;
    return 0;
}

// ============================================================================
// Convert IPA string to token IDs (character-by-character lookup)
// Uses the token map (not the vits2-tokenizer which adds BOS/EOS)
// ============================================================================
static std::vector<int64_t> ipa_to_ids(const std::string& ipa_text,
                                        const std::map<std::string, int>& token_to_id) {
    std::vector<int64_t> ids;

    // Iterate over IPA text character by character (UTF-8 aware if needed)
    // For IPA characters, many are multi-byte UTF-8. We'll try longest-match first.
    std::string remaining = ipa_text;
    while (!remaining.empty()) {
        bool found = false;

        // Try longest match first
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
            // Unknown character — warn and skip
            unsigned char c = static_cast<unsigned char>(remaining[0]);
            std::cerr << "Warning: Unknown token '" << remaining[0] 
                      << "' (0x" << std::hex << (int)c << std::dec << ") — skipping" << std::endl;
            remaining = remaining.substr(1);
        }
    }

    return ids;
}

// ============================================================================
// Intersperse blanks (same as Python: result = [item]*(len*2+1); result[1::2] = lst)
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
// USAGE
// ============================================================================
static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [OPTIONS]\n"
              << "  --text <text>               Text to synthesize (mixed Persian/English)\n"
              << "  --matcha-model <path>       Matcha-TTS ONNX model path\n"
              << "  --vocoder-model <path>      Vocos vocoder ONNX model path\n"
              << "  --tokens <path>             Token file (tokens_sherpa_with_fa.txt)\n"
              << "  --espeak-data <path>        espeak-ng-data directory\n"
              << "  --ezafe-onnx <path>         Ezafe model ONNX path\n"
              << "  --ezafe-spiece <path>       Ezafe sentencepiece model path\n"
              << "  --hazm-words <path>         HAZM words.dat path\n"
              << "  --hazm-verbs <path>         HAZM verbs.dat path\n"
              << "  --hazm-stopwords <path>     HAZM stopwords.dat path\n"
              << "  --homograph <path>          Homograph data JSON path\n"
              << "  --shakkelha <path>          Shakkelha ONNX model path\n"
              << "  --output <path>             Output WAV file (default: output.wav)\n"
              << "  --play                      Play audio immediately after generation\n"
              << "  --temperature <float>       Sampling temperature (default: 0.667)\n"
              << "  --speaking-rate <float>     Speaking rate (default: 1.0)\n"
              << "  --sample-rate <int>         Output sample rate (default: 22050)\n"
              << "  --main-lang <EN|FA|AR>      Main language (default: FA)\n"
              << "  --gpu                       Use GPU provider (default: CPU)\n"
              << "  --help                      Show this help\n"
              << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char* argv[]) {
    // Set UTF-8 locale
    std::locale::global(std::locale("en_US.UTF-8"));

    // ----- Default paths (can be overridden) -----
    std::string matcha_model = "/home/oem/Basir/TTS/Matcha/Trained/onnx/zahra/zahra-22050-5.onnx";
    std::string vocoder_model = "/home/oem/Basir/TTS/vocos22.onnx";
    std::string tokens_file = "/home/oem/Basir/TTS/Matcha/Matcha-TTS/configs/tokens/tokens_sherpa_with_fa.txt";
    std::string espeak_data = "/home/oem/Basir/TTS/Piper/piper_linux_x86_64/piper/espeak-ng-data";
    // NormalizeText resources (relative to cwd by default)
    std::string ezafe_onnx     = "./assets/ezafe_model.onnx";
    std::string ezafe_spiece   = "./assets/ezafe_spiece.model";
    std::string hazm_words     = "./assets/hazm_words.dat";
    std::string hazm_verbs     = "./assets/hazm_verbs.dat";
    std::string hazm_stopwords = "./assets/hazm_stopwords.dat";
    std::string homograph_data = "./assets/homograph_data.json";
    std::string shakkelha_onnx = "./assets/shakkelha.onnx";
    std::string text;
    std::string output_wav = "output.wav";
    bool play_audio = false;
    float temperature = 0.667f;
    float speaking_rate = 1.0f;
    int sample_rate = 22050;
    std::string main_lang_str = "FA";
    bool use_gpu = false;
    (void)use_gpu; // TODO: GPU provider support

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
        else if (arg == "--matcha-model")  matcha_model = require_val("--matcha-model");
        else if (arg == "--vocoder-model") vocoder_model = require_val("--vocoder-model");
        else if (arg == "--tokens")        tokens_file = require_val("--tokens");
        else if (arg == "--espeak-data")   espeak_data = require_val("--espeak-data");
        else if (arg == "--ezafe-onnx")    ezafe_onnx = require_val("--ezafe-onnx");
        else if (arg == "--ezafe-spiece")  ezafe_spiece = require_val("--ezafe-spiece");
        else if (arg == "--hazm-words")    hazm_words = require_val("--hazm-words");
        else if (arg == "--hazm-verbs")    hazm_verbs = require_val("--hazm-verbs");
        else if (arg == "--hazm-stopwords") hazm_stopwords = require_val("--hazm-stopwords");
        else if (arg == "--homograph")     homograph_data = require_val("--homograph");
        else if (arg == "--shakkelha")     shakkelha_onnx = require_val("--shakkelha");
        else if (arg == "--output")        output_wav = require_val("--output");
        else if (arg == "--play")          play_audio = true;
        else if (arg == "--temperature")   temperature = std::stof(require_val("--temperature"));
        else if (arg == "--speaking-rate") speaking_rate = std::stof(require_val("--speaking-rate"));
        else if (arg == "--sample-rate")   sample_rate = std::stoi(require_val("--sample-rate"));
        else if (arg == "--main-lang")     main_lang_str = require_val("--main-lang");
        else if (arg == "--gpu")           use_gpu = true;
        else if (arg == "--help")          { print_usage(argv[0]); return 0; }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (text.empty()) {
        std::cerr << "Error: --text is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // ========================================================================
    // STEP 1: Load token map
    // ========================================================================
    std::map<std::string, int> token_to_id;
    std::map<int, std::string> id_to_token;
    if (read_token_map(tokens_file, token_to_id, id_to_token) != 0) {
        return 1;
    }

    // ========================================================================
    // STEP 2: Normalize text → IPA phonemes
    // ========================================================================
    Language mainlang = LanguageDetector::string_to_language(main_lang_str);

    NormalizeConfig norm_config;
    norm_config.espeak_data_path = espeak_data;
    norm_config.shakkelha_onnx   = shakkelha_onnx;
    norm_config.ezafe_model_onnx = ezafe_onnx;
    norm_config.ezafe_model_spiece = ezafe_spiece;
    norm_config.hazm_words       = hazm_words;
    norm_config.hazm_verbs       = hazm_verbs;
    norm_config.hazm_stopwords   = hazm_stopwords;
    norm_config.homograph_data   = homograph_data;

    std::string normalized_text;
    std::string ipa_text;

    std::cout << "Normalizing text: " << text << std::endl;
    normalizeString(mainlang, 1 /* IPA mode */, text, normalized_text, ipa_text, norm_config);

    // Remove newlines from IPA output (phonemizer adds sentence breaks)
    ipa_text.erase(std::remove(ipa_text.begin(), ipa_text.end(), '\n'), ipa_text.end());

    std::cout << "IPA phonemes: " << ipa_text << std::endl;
    std::cout << "Normalized:   " << normalized_text << std::endl;

    // ========================================================================
    // STEP 3: Tokenize IPA → token IDs + intersperse blanks
    // ========================================================================
    auto token_ids = ipa_to_ids(ipa_text, token_to_id);
    std::cout << "Token IDs (" << token_ids.size() << "): ";
    for (size_t i = 0; i < std::min(token_ids.size(), size_t(20)); ++i) {
        std::cout << token_ids[i] << " ";
    }
    if (token_ids.size() > 20) std::cout << "...";
    std::cout << std::endl;

    auto ids_with_blanks = intersperse_blanks(token_ids, 0);  // 0 = blank/pad token
    int64_t seq_len = static_cast<int64_t>(ids_with_blanks.size());
    std::cout << "With blanks: " << seq_len << " tokens" << std::endl;

    // ========================================================================
    // STEP 4: Run Matcha-TTS ONNX model
    // ========================================================================
    std::cout << "\n=== Matcha-TTS Inference ===" << std::endl;
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tts_infer");
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(4);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Load Matcha model
    auto t0 = std::chrono::high_resolution_clock::now();
    Ort::Session matcha_session(env, matcha_model.c_str(), session_opts);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Matcha model loaded in " 
              << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;

    // Build inputs
    std::vector<int64_t> x_shape = {1, seq_len};
    std::vector<int64_t> x_lengths_data = {seq_len};
    std::vector<float> scales_data = {temperature, speaking_rate};

    Ort::Value x_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, ids_with_blanks.data(), ids_with_blanks.size(),
        x_shape.data(), x_shape.size());
    Ort::Value x_lengths_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, x_lengths_data.data(), 1,
        (std::vector<int64_t>{1}).data(), 1);
    Ort::Value scales_tensor = Ort::Value::CreateTensor<float>(
        mem_info, scales_data.data(), 2,
        (std::vector<int64_t>{2}).data(), 1);

    const char* matcha_input_names[] = {"x", "x_lengths", "scales"};
    const char* matcha_output_names[] = {"mel", "mel_lengths"};

    std::vector<Ort::Value> matcha_inputs;
    matcha_inputs.push_back(std::move(x_tensor));
    matcha_inputs.push_back(std::move(x_lengths_tensor));
    matcha_inputs.push_back(std::move(scales_tensor));

    std::cout << "Running Matcha inference..." << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    auto matcha_outputs = matcha_session.Run(
        Ort::RunOptions{nullptr},
        matcha_input_names, matcha_inputs.data(), matcha_inputs.size(),
        matcha_output_names, 2);
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Matcha inference done in " 
              << std::chrono::duration<double>(t3 - t2).count() << "s" << std::endl;

    // Get mel output: shape [1, 80, mel_len]
    float* mel_data = matcha_outputs[0].GetTensorMutableData<float>();
    auto mel_info = matcha_outputs[0].GetTensorTypeAndShapeInfo();
    auto mel_shape = mel_info.GetShape();
    int64_t mel_frames = mel_shape[2];
    std::cout << "Mel shape: [" << mel_shape[0] << ", " << mel_shape[1] << ", " << mel_shape[2] << "]" << std::endl;

    int64_t* mel_lengths_data = matcha_outputs[1].GetTensorMutableData<int64_t>();
    int64_t mel_len = mel_lengths_data[0];
    std::cout << "Mel length: " << mel_len << " frames" << std::endl;

    // ========================================================================
    // STEP 5: Run Vocos vocoder ONNX model
    // ========================================================================
    std::cout << "\n=== Vocoder Inference ===" << std::endl;

    Ort::Session vocoder_session(env, vocoder_model.c_str(), session_opts);
    std::cout << "Vocoder model loaded" << std::endl;

    // Build vocoder inputs: mel_spec [1, 80, mel_frames], denoise [1] = 0.0
    std::vector<int64_t> mel_input_shape = {1, 80, mel_frames};
    std::vector<float> denoise_data = {0.0f};

    Ort::Value mel_vocoder_tensor = Ort::Value::CreateTensor<float>(
        mem_info, mel_data, mel_shape[0] * mel_shape[1] * mel_shape[2],
        mel_input_shape.data(), mel_input_shape.size());
    Ort::Value denoise_tensor = Ort::Value::CreateTensor<float>(
        mem_info, denoise_data.data(), 1,
        (std::vector<int64_t>{1}).data(), 1);

    const char* vocoder_input_names[] = {"mel_spec", "denoise"};
    const char* vocoder_output_names[] = {"wave"};

    std::vector<Ort::Value> vocoder_inputs;
    vocoder_inputs.push_back(std::move(mel_vocoder_tensor));
    vocoder_inputs.push_back(std::move(denoise_tensor));

    std::cout << "Running vocoder inference..." << std::endl;
    auto t4 = std::chrono::high_resolution_clock::now();
    auto vocoder_outputs = vocoder_session.Run(
        Ort::RunOptions{nullptr},
        vocoder_input_names, vocoder_inputs.data(), vocoder_inputs.size(),
        vocoder_output_names, 1);
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "Vocoder inference done in "
              << std::chrono::duration<double>(t5 - t4).count() << "s" << std::endl;

    // Get wave output: shape [1, num_samples]
    float* wave_data = vocoder_outputs[0].GetTensorMutableData<float>();
    auto wave_info = vocoder_outputs[0].GetTensorTypeAndShapeInfo();
    auto wave_shape = wave_info.GetShape();
    int64_t num_samples = wave_shape[1];
    std::cout << "Wave shape: [" << wave_shape[0] << ", " << num_samples << "]" << std::endl;

    // ========================================================================
    // STEP 6: Write WAV file
    // ========================================================================
    std::vector<float> audio(wave_data, wave_data + num_samples);
    write_wav(output_wav, audio, sample_rate);

    double wav_secs = num_samples / (double)sample_rate;
    double total_secs = std::chrono::duration<double>(t5 - t0).count();
    std::cout << "\n=== Done ===" << std::endl;
    std::cout << "Output: " << output_wav << std::endl;
    std::cout << "Duration: " << wav_secs << "s" << std::endl;
    std::cout << "Total time: " << total_secs << "s" << std::endl;
    std::cout << "RTF: " << (total_secs / wav_secs) << std::endl;

    // Play audio if requested
    if (play_audio) {
        std::string cmd = "ffplay -nodisp -autoexit \"" + output_wav + "\" 2>/dev/null";
        std::system(cmd.c_str());
    }

    return 0;
}
