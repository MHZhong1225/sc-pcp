# Public runner boundary

The public package is designed around `scpcp.experiments` and the small unit
tests in `tests/per_step/`. Paper-scale runners, frozen study configurations,
figure renderers, and repair/replay scripts are private research operations;
they live under the local Git-ignored `internal/` archive and are not part of a
GitHub release.

This directory retains only compatibility runners required by existing local
research records. New users should call the public Python API instead of
starting a paper-scale run.
