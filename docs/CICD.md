# CI/CD Guide

This repository currently has no active CI/CD automation.

## Current State

- No GitHub Actions workflows are installed.
- No release packaging workflow is installed.
- No dependency review or scheduled vulnerability scan is installed.
- Local repository checks are still available through `make check-repo`.

## Design Principle

Avoid carrying template automation that the project does not actively use.

When CI/CD becomes necessary, add the smallest workflow that validates the real stack and documents the expected operator behavior in this file.

Any future GitHub Actions should be pinned to immutable commit SHAs instead of floating tags.

## Reintroduction Sequence

1. Define the repository's real build, test, and packaging commands.
2. Add a single pull-request workflow that runs those commands.
3. Add dependency and vulnerability checks once lockfiles or manifests exist.
4. Add release packaging only after there is a real artifact to publish.
5. Add deployment jobs after a runtime target and rollback path exist.

## Local Validation

Use `make check-repo` for the current local baseline. It verifies the documentation scaffold, repository hygiene, and shell script syntax without invoking remote automation.
