#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_ARGS=(-f docker-compose.yml)
REMOVE_VOLUMES=0
STOP_CLOUDFLARED=0

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf '[%s] [warn] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
  cat <<'EOF'
Usage: ./stop_app.sh [--remove-volumes] [--stop-cloudflared]

Options:
  --remove-volumes    Also remove compose volumes (data loss risk for volume-backed data).
  --stop-cloudflared  Attempt to stop cloudflared system service with sudo.
  -h, --help          Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remove-volumes)
      REMOVE_VOLUMES=1
      ;;
    --stop-cloudflared)
      STOP_CLOUDFLARED=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd docker
require_cmd curl

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not reachable. Start Docker first." >&2
  exit 1
fi

log "Current compose service status before shutdown:"
docker compose "${COMPOSE_ARGS[@]}" ps || true

if (( REMOVE_VOLUMES == 1 )); then
  warn "Shutting down stack and removing volumes."
  docker compose "${COMPOSE_ARGS[@]}" down --remove-orphans --volumes
else
  log "Shutting down stack safely (containers + network, keeping volumes)."
  docker compose "${COMPOSE_ARGS[@]}" down --remove-orphans
fi

log "Compose service status after shutdown:"
docker compose "${COMPOSE_ARGS[@]}" ps || true

if curl -fsS --max-time 5 "http://localhost:8080/healthz" >/dev/null 2>/dev/null; then
  warn "web-app still responds on localhost:8080. Another process may still be running."
else
  log "[ok] web-app endpoint is down on localhost:8080."
fi

if (( STOP_CLOUDFLARED == 1 )); then
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files cloudflared.service --no-legend >/dev/null 2>&1; then
    log "Stopping cloudflared service..."
    if sudo systemctl stop cloudflared; then
      log "[ok] cloudflared service stopped."
    else
      warn "Could not stop cloudflared service."
    fi
  else
    warn "cloudflared service is not available on this host."
  fi
else
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files cloudflared.service --no-legend >/dev/null 2>&1; then
    local_state="$(systemctl is-active cloudflared 2>/dev/null || true)"
    log "cloudflared service state (unchanged): $local_state"
  fi
fi

log "Shutdown completed."
