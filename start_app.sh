#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_ARGS=(-f docker-compose.yml -f docker-compose.gpu-test.yml)
DO_BUILD=0
SKIP_PUBLIC_CHECK=0

usage() {
  cat <<'EOF'
Usage: ./start_app.sh [--build] [--skip-public-check]

Options:
  --build               Rebuild images before starting.
  --skip-public-check   Skip the https://jetsonocrai.cc readiness check.
  -h, --help            Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      DO_BUILD=1
      ;;
    --skip-public-check)
      SKIP_PUBLIC_CHECK=1
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

check_http_ready() {
  local url="$1"
  local label="$2"
  local max_attempts="${3:-40}"
  local sleep_seconds="${4:-3}"
  local attempt=1

  while (( attempt <= max_attempts )); do
    if curl -fsS --max-time 5 "$url" >/dev/null; then
      echo "[ok] $label ready: $url"
      return 0
    fi
    sleep "$sleep_seconds"
    (( attempt += 1 ))
  done

  echo "[warn] $label not ready after $((max_attempts * sleep_seconds))s: $url"
  return 1
}

check_cloudflared() {
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "[warn] systemctl not found, cannot verify cloudflared service."
    return 0
  fi

  if ! systemctl list-unit-files cloudflared.service --no-legend >/dev/null 2>&1; then
    echo "[warn] cloudflared.service is not installed on this host."
    return 0
  fi

  local active enabled
  active="$(systemctl is-active cloudflared 2>/dev/null || true)"
  enabled="$(systemctl is-enabled cloudflared 2>/dev/null || true)"

  if [[ "$active" == "active" ]]; then
    echo "[ok] cloudflared service is active (enabled: $enabled)."
  else
    echo "[warn] cloudflared service is not active (state: $active, enabled: $enabled)."
    echo "       Run: sudo systemctl restart cloudflared"
  fi
}

require_cmd docker
require_cmd curl

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not reachable. Start Docker first." >&2
  exit 1
fi

echo "Starting OCR stack with Docker Compose..."
if (( DO_BUILD == 1 )); then
  docker compose "${COMPOSE_ARGS[@]}" up -d --build
else
  docker compose "${COMPOSE_ARGS[@]}" up -d
fi

echo
echo "Compose service status:"
docker compose "${COMPOSE_ARGS[@]}" ps

echo
echo "Checking local service readiness..."
check_http_ready "http://localhost:8080/healthz" "web-app"
check_http_ready "http://localhost:8080/sessions/recent" "web-app sessions API"

echo
echo "Checking Cloudflare tunnel service..."
check_cloudflared

if (( SKIP_PUBLIC_CHECK == 0 )); then
  echo
  echo "Checking public endpoint..."
  check_http_ready "https://jetsonocrai.cc/" "public app" 20 3 || true
fi

echo
echo "Launch completed."
echo "Local app:  http://localhost:8080"
echo "Public app: https://jetsonocrai.cc"
