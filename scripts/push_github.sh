#!/usr/bin/env bash
# Push to GitHub bypassing global ghfast.top URL rewrite (which breaks gh token auth).
#
# Usage:
#   cd /dev/gpc_code/agentic_es
#   git add -A && git commit -m "your message"
#   ./scripts/push_github.sh
#
#   ./scripts/push_github.sh https://github.com/JWgpc/agent_ES.git main

set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${1:-https://github.com/JWgpc/agent_ES.git}"
BRANCH="${2:-main}"

GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
  git -c http.version=HTTP/1.1 \
      -c http.postBuffer=524288000 \
      -c 'credential.helper=!/usr/bin/gh auth git-credential' \
  push -u "$REMOTE" "$BRANCH"

echo "Pushed to $REMOTE ($BRANCH)"
