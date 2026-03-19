# Contributing to Barq Inference

Thanks for helping improve the project. The codebase is organized around small crates and phase-based changes, so the contribution process is intentionally incremental.

## Workflow

1. Branch from `main`.
2. Make one atomic commit per logical change.
3. Run formatting and the relevant tests before pushing.
4. Update docs when you change public behavior or CLI flags.
5. Open a pull request against `main`.

## Expectations

- Keep diffs focused and reviewable.
- Do not mix unrelated refactors into feature work.
- Preserve existing behavior unless the change is explicitly intended.
- Avoid committing generated noise such as `.DS_Store` files.

## Validation

At minimum, run the checks that cover the area you touched:

- `cargo test -p models`
- `cargo test -p barq-inference`
- `cargo test -p sampling`
- `cargo bench -p barq-inference --benches --no-run`

If you touch docs-only files, still make sure the examples are accurate and the links resolve.

## Documentation References

- [Architecture Guide](docs/ARCHITECTURE.md)
- [Testing Guide](docs/TESTING.md)
- [Release Guide](docs/RELEASE.md)

