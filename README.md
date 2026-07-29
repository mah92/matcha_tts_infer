# MatchaTTSInfer

C++ TTS inference engine using Matcha-TTS acoustic model + Vocos vocoder via ONNX Runtime.

Supports mixed Persian/English text with diacritization, ezafe detection, and homograph disambiguation.

## Requirements

- CMake 3.10+
- GCC 9+ (C++17)
- ONNX Runtime (headers + shared lib)
- espeak-ng (1.52+)
- ICU (Unicode lib)
- libpulse-dev (for --play audio)

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

## Usage

```bash
./build/MatchaTTSInfer \
    --text "سلام دنیا" \
    --espeak-data /path/to/espeak-ng-data \
    --tokens /path/to/tokens_sherpa_with_fa.txt \
    --matcha-model /path/to/zahra-22050-5.onnx \
    --vocoder-model /path/to/vocos22.onnx \
    --output output.wav \
    --play
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--text` | Text to synthesize (required) | — |
| `--matcha-model` | Matcha-TTS ONNX model | `~/.../zahra-22050-5.onnx` |
| `--vocoder-model` | Vocos vocoder ONNX model | `~/.../vocos22.onnx` |
| `--tokens` | Token file | `~/.../tokens_sherpa_with_fa.txt` |
| `--espeak-data` | espeak-ng data dir | system default |
| `--output` | Output WAV file | `output.wav` |
| `--play` | Play audio after generation | off |

## Directory Structure

```
match_tts_infer/
├── CMakeLists.txt          # Build config
├── tts_infer.cpp           # Main TTS inference (Matcha + Vocos)
├── NormalizeText/          # Text normalization submodule
│   ├── assets/             # Models and data files
│   │   ├── ezafe_model.onnx
│   │   ├── ezafe_spiece.model
│   │   ├── shakkelha.onnx
│   │   ├── homograph_data.json
│   │   └── hazm_*.dat
│   ├── hazm_cpp/           # Persian NLP (stemming, lemmatization)
│   ├── persian-ezafe-albert-cpp/  # Ezafe detection model
│   └── ...                 # Additional normalization modules
└── build/                  # Build output
```

## License

See LICENSE file.
