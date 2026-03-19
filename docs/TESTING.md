# Testing Guide

## Testing Strategy

The repository uses three layers of validation:

1. Unit tests for crate-local logic.
2. Integration tests for end-to-end model loading and CLI behavior.
3. Benchmarks for performance regression tracking.

## Common Commands

```bash
cargo test -p models --quiet
cargo test -p barq-inference --test integration_tests --quiet
cargo test -p sampling --quiet
cargo test -p barq_core --quiet
cargo bench -p barq-inference --benches --no-run
```

## Fixture Strategy

- `models/src/test_support.rs` creates deterministic GGUF fixtures for architecture and loader coverage.
- `cli/tests/integration/fixtures.rs` builds tiny models for end-to-end CLI tests.
- Coverage tests exercise the major quantization, sampler, rope, and registry paths.

## What To Test

When you add or change code:

- Add unit tests for small pure functions.
- Add integration tests for loader, CLI, and server behavior.
- Add doctests or API examples when you change public types.
- Add a benchmark when the change affects throughput or latency.

## Good Signals

The project is healthy when:

- The model crate compiles and passes doctests.
- The CLI integration suite passes.
- Benchmarks run without panicking.
- New features are represented in the implementation plan.

## Related Docs

- [Architecture Guide](./ARCHITECTURE.md)
- [Release Guide](./RELEASE.md)
- [Contributing](../CONTRIBUTING.md)
