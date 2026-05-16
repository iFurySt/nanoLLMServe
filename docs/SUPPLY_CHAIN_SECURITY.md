# Supply Chain Security

This document defines the supply-chain posture for the repository while no CI/CD automation is installed.

## Current Controls

- Keep dependency manifests and lockfiles committed once the project has real dependencies.
- Review dependency changes during code review until automated dependency review is reintroduced.
- Keep future GitHub Actions pinned to immutable commit SHAs instead of floating tags.
- Generate SBOMs and provenance only after the project has real release artifacts.

## Current Workflow Mapping

No supply-chain workflow is currently installed.

## Limits And Assumptions

- Dependency Review requires GitHub support for the repository plan and should be reintroduced only when pull-request automation is useful.
- OSV and SBOM quality depend on the project checking in recognizable manifests or lockfiles.
- Provenance is only meaningful once release packaging represents the real build output of the project.
- OpenSSF Scorecard is intentionally not enabled because the repository has no real branch protection, release history, or SAST posture to score yet.

## What To Do When The Project Becomes Real

- Add ecosystem-specific lockfiles and keep them committed.
- Make the build deterministic and produce explicit versioned artifacts.
- Add dependency review or vulnerability scanning to the first real CI workflow.
- Gate production deployment on release artifact provenance verification when possible.
- Consider verifying attestations in the deployment environment or cluster admission layer.
