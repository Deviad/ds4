#!/usr/bin/env bash
# Portable wrapper: resolves same-name target from the global role-pipeline skill.
# Requires: global role-pipeline skill installed under the Pi agent skills dir.
#
# Override the skill directory:
#   AGENT_SKILLS_DIR=/custom/path
#   PI_AGENT_SKILLS_DIR=/custom/path   (alternate name)

set -euo pipefail

_WRAPPER_NAME="$(basename "${BASH_SOURCE[0]}")"
_SKILLS_DIR="${AGENT_SKILLS_DIR:-${PI_AGENT_SKILLS_DIR:-${HOME}/.pi/agent/skills}}"
_TARGET="${_SKILLS_DIR}/role-pipeline/scripts/${_WRAPPER_NAME}"

if [[ ! -f "${_TARGET}" ]]; then
  echo "[pi-agent-wrapper] FATAL: role-pipeline target not found: ${_TARGET}" >&2
  echo "[pi-agent-wrapper] Install the global role-pipeline skill:" >&2
  echo "[pi-agent-wrapper]   git clone <role-pipeline-repo> \"${_SKILLS_DIR}/role-pipeline\"" >&2
  echo "[pi-agent-wrapper] Or override AGENT_SKILLS_DIR to point to your Pi agent skills root." >&2
  exit 1
fi

# Detect execution mode: sourced vs executed
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  # Sourced by another script — source the target library
  # shellcheck disable=SC1090
  source "${_TARGET}"
else
  # Executed directly — exec the target with all arguments
  exec "${_TARGET}" "$@"
fi
