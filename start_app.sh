#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_ARGS=(-f docker-compose.yml)
BUILD_IMAGES=0
REBUILD_DEPS=0
PUBLIC_CHECK=1
LOCAL_TEST=0
ENV_FILE="$SCRIPT_DIR/.env"
DEPS_STATE_DIR="$SCRIPT_DIR/data/.start_app"
DEPS_STATE_FILE="$DEPS_STATE_DIR/deps_fingerprint.txt"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf '[%s] [warn] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

compute_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256
  else
    echo "Missing required command: sha256sum or shasum" >&2
    exit 1
  fi
}

dependency_input_files() {
  local -a files=(
    "docker/Dockerfile"
    "docker/Dockerfile.llm"
    "docker/Dockerfile.web"
    "requirements/ocr.txt"
    "requirements/llm.txt"
    "requirements/web.txt"
  )
  local wheel
  shopt -s nullglob
  for wheel in wheels/paddlepaddle_gpu-*.whl; do
    files+=("$wheel")
  done
  shopt -u nullglob
  printf '%s\n' "${files[@]}" | sort
}

build_dependency_manifest() {
  local file
  while IFS= read -r file; do
    [[ -n "$file" ]] || continue
    if [[ ! -f "$file" ]]; then
      continue
    fi
    printf 'FILE %s\n' "$file"
    compute_sha256 <"$file"
  done < <(dependency_input_files)
}

current_dependency_fingerprint() {
  build_dependency_manifest | compute_sha256 | awk '{print $1}'
}

saved_dependency_fingerprint() {
  if [[ ! -f "$DEPS_STATE_FILE" ]]; then
    return 1
  fi
  sed -n '1p' "$DEPS_STATE_FILE"
}

warn_if_dependency_rebuild_recommended() {
  local current_fingerprint saved_fingerprint
  current_fingerprint="$(current_dependency_fingerprint)"
  if saved_fingerprint="$(saved_dependency_fingerprint)"; then
    if [[ "$current_fingerprint" != "$saved_fingerprint" ]]; then
      warn "Dependency inputs changed since the last successful --rebuild-deps run."
      warn "Run ./start_app.sh --rebuild-deps if Docker/system/Python dependencies were intentionally changed."
    fi
    return 0
  fi

  warn "No dependency rebuild baseline found yet."
  warn "Run ./start_app.sh --rebuild-deps once to record a baseline after dependency changes."
}

record_dependency_fingerprint() {
  mkdir -p "$DEPS_STATE_DIR"
  {
    current_dependency_fingerprint
    build_dependency_manifest
  } >"$DEPS_STATE_FILE"
}

usage() {
  cat <<'EOF'
Usage: ./start_app.sh [--build] [--rebuild-deps] [--no-build] [--skip-public-check] [--local_test]

Options:
  --build               Rebuild images with normal Docker layer caching.
  --rebuild-deps        Rebuild images and refresh dependency layers.
  --no-build            Recreate containers without rebuilding images.
  --skip-public-check   Skip https://jetsonocrai.cc readiness check.
  --local_test          Local mode: WEB_APP_COOKIE_SECURE=0 and skip public check.
  -h, --help            Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      BUILD_IMAGES=1
      ;;
    --rebuild-deps)
      BUILD_IMAGES=1
      REBUILD_DEPS=1
      ;;
    --no-build)
      BUILD_IMAGES=0
      REBUILD_DEPS=0
      ;;
    --skip-public-check)
      PUBLIC_CHECK=0
      ;;
    --local_test|--local-test)
      LOCAL_TEST=1
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

load_env_file() {
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing required .env file: $ENV_FILE" >&2
    exit 1
  fi

  # Export everything sourced from .env for docker compose variable expansion.
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

cleanup_failed_startup() {
  warn "Cleaning up partially started stack..."
  docker compose "${COMPOSE_ARGS[@]}" down --remove-orphans || true
  log "Cleanup complete."
}

report_failure_and_cleanup() {
  warn "$1"
  docker compose "${COMPOSE_ARGS[@]}" ps -a || true
  docker compose "${COMPOSE_ARGS[@]}" logs --no-color --tail=160 llm ocr web-app || true
  cleanup_failed_startup
  exit 1
}

check_http_ready() {
  local url="$1"
  local label="$2"
  local max_attempts="${3:-40}"
  local sleep_seconds="${4:-3}"
  local attempt=1

  while (( attempt <= max_attempts )); do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>/dev/null; then
      log "[ok] $label ready: $url"
      return 0
    fi
    log "$label pending ($attempt/$max_attempts): $url"
    sleep "$sleep_seconds"
    (( attempt += 1 ))
  done

  warn "$label not ready after $((max_attempts * sleep_seconds))s: $url"
  return 1
}

check_cloudflared() {
  if ! command -v systemctl >/dev/null 2>&1; then
    warn "systemctl not found; cannot verify cloudflared service."
    return 0
  fi

  if ! systemctl list-unit-files cloudflared.service --no-legend >/dev/null 2>&1; then
    warn "cloudflared.service is not installed on this host."
    return 0
  fi

  local active enabled
  active="$(systemctl is-active cloudflared 2>/dev/null || true)"
  enabled="$(systemctl is-enabled cloudflared 2>/dev/null || true)"

  if [[ "$active" == "active" ]]; then
    log "[ok] cloudflared service is active (enabled: $enabled)."
  else
    warn "cloudflared service is not active (state: $active, enabled: $enabled)."
    warn "Run: sudo systemctl restart cloudflared"
  fi
}

require_cmd docker
require_cmd curl

load_env_file

if (( LOCAL_TEST == 1 )); then
  export WEB_APP_COOKIE_SECURE=0
  PUBLIC_CHECK=0
  log "Running in local test mode (WEB_APP_COOKIE_SECURE=0)."
else
  export WEB_APP_COOKIE_SECURE=1
  log "Running in production mode (WEB_APP_COOKIE_SECURE=1)."
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not reachable. Start Docker first." >&2
  exit 1
fi

if (( REBUILD_DEPS == 0 )); then
  warn_if_dependency_rebuild_recommended
fi

if (( BUILD_IMAGES == 1 )); then
  if (( REBUILD_DEPS == 1 )); then
    export DEPS_CACHE_BUSTER="$(date +%s)"
    log "Starting OCR stack with Docker Compose (dependency rebuild enabled)..."
  else
    unset DEPS_CACHE_BUSTER || true
    log "Starting OCR stack with Docker Compose (image rebuild enabled)..."
  fi
  if ! docker compose "${COMPOSE_ARGS[@]}" up -d --build; then
    report_failure_and_cleanup "Compose startup failed."
  fi
else
  unset DEPS_CACHE_BUSTER || true
  log "Starting OCR stack with Docker Compose (no rebuild, recreate containers)..."
  if ! docker compose "${COMPOSE_ARGS[@]}" up -d --force-recreate; then
    report_failure_and_cleanup "Compose startup failed."
  fi
fi

log "Compose service status:"
docker compose "${COMPOSE_ARGS[@]}" ps

log "Checking local service readiness..."
if ! check_http_ready "http://localhost:8080/healthz" "web-app" 50 2; then
  report_failure_and_cleanup "web-app health endpoint did not become ready."
fi
if ! check_http_ready "http://localhost:8080/sessions/recent" "web-app sessions API" 50 2; then
  report_failure_and_cleanup "web-app sessions API did not become ready."
fi

if (( LOCAL_TEST == 0 )); then
  log "Checking Cloudflare tunnel service..."
  check_cloudflared
else
  log "Cloudflare tunnel check skipped in local test mode."
fi

if (( PUBLIC_CHECK == 1 )); then
  log "Checking public endpoint..."
  check_http_ready "https://jetsonocrai.cc/" "public app" 25 3 || true
else
  log "Public endpoint check skipped."
fi

if (( REBUILD_DEPS == 1 )); then
  record_dependency_fingerprint
  log "Recorded dependency rebuild baseline."
fi

log "Launch completed."
log "Local app:  http://localhost:8080"
log "Public app: https://jetsonocrai.cc"
