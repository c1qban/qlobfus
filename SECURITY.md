# Security Policy

This repository is a research prototype for C source transformation and empirical semantic validation.

## Supported Versions

Security-relevant reports should target the latest release or the current `main` branch.

## Reporting a Vulnerability

Please do not disclose exploitable issues publicly before maintainers have had a chance to respond. Open a private advisory if the hosting platform supports it, or contact the repository owner through the contact address listed in the paper.

## Important Safety Notes

- The verifier compiles and executes C programs. Run experiments only on trusted inputs or inside a sandbox/container.
- Do not run untrusted C submissions on a personal machine without process, filesystem, and network isolation.
- Generated binaries, logs, and checkpoints should not be committed unless they have been reviewed for licensing and sensitive content.
- Tigress and Project CodeNet are third-party resources with their own licenses and security considerations.
