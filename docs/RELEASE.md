# Release Guide

## Release Process

Releases follow the same phase-based workflow used for development:

1. Create a phase branch from `main`.
2. Implement the phase in atomic commits.
3. Run the relevant tests and benchmarks.
4. Update the implementation plan and user-facing docs.
5. Push the branch and open a pull request.
6. Merge the PR into `main`.
7. Delete the phase branch and prune remote refs.

## Release Checklist

- All phase checkboxes are updated in `IMPLEMENTATION_PLAN.md`.
- The README reflects the shipped features.
- User guide and performance guide are up to date.
- Architecture and testing docs match the current code layout.
- The working tree is clean except for intentional local files.

## Versioning

The repository currently tracks progress by phase rather than by formal semver tags. If release tags are introduced later, they should point at a merged `main` commit after the phase docs and tests are complete.

## Related Docs

- [Contributing](../CONTRIBUTING.md)
- [Architecture Guide](./ARCHITECTURE.md)
- [Testing Guide](./TESTING.md)
