# Security Policy — LEO Rigging AI

## ⚠️ CRITICAL: Exposed API Key — Immediate Action Required

A real OpenAI API key was committed to the Git history of this repository.

**The repository owner MUST complete these steps before any other action:**

1. **Revoke the exposed key** at <https://platform.openai.com/api-keys>.
2. **Review API usage and billing** for unauthorized charges at
   <https://platform.openai.com/usage>.
3. **Create a replacement key** in the OpenAI dashboard.
4. **Store the replacement ONLY** in:
   - A secret manager (HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager, etc.), OR
   - An untracked local `.env` file (already in `.gitignore`).
5. **Purge the old secret from Git history** using `git filter-repo` or
   BFG Repo Cleaner **after** receiving explicit authorization from all
   repository contributors and understanding the implications for shared
   forks and clones.
6. **Rotate any other credentials** (e.g., service accounts, tokens) that
   may have been stored alongside the exposed key in the same files.
7. **Never push** Git-history rewrites without notifying all collaborators.

> This repository does not have the ability to revoke external credentials.
> Only the key owner can do this. Do it now.

---

## Reporting a Vulnerability

If you discover a security vulnerability in LEO Rigging AI:

1. **Do not open a public GitHub issue.**
2. Email the maintainers directly (add your contact address here).
3. Include:
   - A description of the vulnerability and its potential impact.
   - Steps to reproduce or proof-of-concept code.
   - Any suggested mitigations.
4. You will receive an acknowledgement within 48 hours and a resolution
   timeline within 7 days.

---

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest `main` | ✅ Yes |
| All others | ❌ No |

---

## Security Controls Summary

| Control | Status |
|---------|--------|
| API authentication (bearer tokens) | ✅ Implemented |
| Scope-based authorization | ✅ Implemented |
| Path traversal prevention (upload/download/delete) | ✅ Implemented |
| PDF MIME + magic-byte validation | ✅ Implemented |
| Streaming upload (no full-file memory load) | ✅ Implemented |
| CORS allowlist (no wildcard + credentials) | ✅ Implemented |
| Security headers middleware | ✅ Implemented |
| Structured logs with credential redaction | ✅ Implemented |
| `/docs` disabled in production | ✅ Implemented |
| Minimum relevance threshold (grounded refusal) | ✅ Implemented |
| Prompt-injection delimiters | ✅ Implemented |
| Secret scanning in CI (trufflehog) | ✅ CI workflow |
| Dependency audit in CI (pip-audit) | ✅ CI workflow |
| Non-root Docker container | ✅ Dockerfile |
| `.env` in `.gitignore` | ✅ |
| Malware scanning (ClamAV) | ⚠️ Interface only — requires external daemon |
| TLS termination | ⚠️ Requires reverse proxy (nginx/caddy) |
| Multi-worker BM25 safety | ⚠️ Single-worker only; see DEPLOYMENT.md |
| OAuth/JWT provider | ❌ Not in scope — use static tokens initially |

---

## Domain Safety Notice

LEO Rigging AI is an **educational** chat assistant. It is NOT:

- A rigging calculator or lift-plan approval system.
- An engineering authority or competent person.
- A replacement for equipment inspection, manufacturer instructions,
  site procedures, or applicable regulations.

Answers must be verified by qualified rigging/lifting subject-matter
experts before operational use. See [docs/INCIDENT_RESPONSE.md](docs/INCIDENT_RESPONSE.md).
