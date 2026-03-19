# Migration Guide From llama.cpp

This guide maps the most common llama.cpp workflows to Barq Inference.

## Command Mapping

| llama.cpp | Barq Inference |
|-----------|----------------|
| `./main -m model.gguf -p "Hello"` | `barq-inference run -m model.gguf -p "Hello"` |
| `--temp 0.8` | `--temperature 0.8` |
| `--top-k 40` | `--top-k 40` |
| `--top-p 0.95` | `--top-p 0.95` |
| `--ctx-size 4096` | `--context-size 4096` |
| `server --host 0.0.0.0 --port 8000` | `barq-inference http-server --host 0.0.0.0 --port 8000` |

## Feature Mapping

- `--json` enables grammar-guided JSON output.
- `--mla` enables FlashMLA expansion for DeepSeek-family models.
- `--fmoe` enables fused MoE dispatch helpers.
- `--ser` enables smart expert reduction.
- `--speculative` enables speculative decoding.

## Chat Templates

Barq uses the chat template support in `vocab/src/chat_template.rs`. Most common model families already have templates registered, including:

- LLaMA
- Mistral
- Qwen
- Qwen2
- Qwen2-VL
- LLaVA

## Model Loading

Barq reads GGUF metadata directly and detects the architecture from the file contents. If a model fails to load:

1. Check the `general.architecture` metadata field.
2. Confirm the tensor names match the architecture family.
3. Run `barq-inference info -m model.gguf` to inspect the file.

## What Is Different

- The CLI uses `barq-inference` instead of llama.cpp's `main` binary name.
- Grammar mode is surfaced as `--json` for the common structured-output case.
- OpenAI-compatible endpoints are built in through `http-server`.
- The Rust API is split across focused crates: `models`, `vocab`, `sampling`, `advanced`, and `barq_core`.

## Suggested Migration Path

1. Start with the same GGUF model you used in llama.cpp.
2. Recreate the prompt and sampling flags with the Barq CLI.
3. Compare output quality and throughput using `barq-inference benchmark`.
4. Move your production deployment to `http-server` once the CLI parity is confirmed.

## Related Docs

- [User Guide](./USER_GUIDE.md)
- [Performance Guide](./PERFORMANCE.md)
- [README](../README.md)
