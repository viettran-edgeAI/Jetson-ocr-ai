# Jetson OCR Project

Local OCR + LLM system for a Jetson Orin Nano Super.

## Documentation

- [Directory structure](./docs/DIRECTORY_STRUCTURE.md)
- [Design history](./docs/DESIGN_HISTORY.md)
- [Web-app session interface and system design](./docs/web_app.md)

## Current status

- `ocr-service` runs locally and in Docker.
- `ocr-service` now uses the replacement Jetson OCR pipeline internally while preserving the existing `POST /v1/ocr` Markdown contract for `web-app`.
- `llm-service` runs with Gemma GGUF, opt-in Thinking mode, and a 12288-token context window.
- `web-app` provides browser sessions, OCR preview, chat, account flows, recent-session persistence, and rate limits by tier.
- Public deployments target `https://jetsonocrai.cc` through Cloudflare Tunnel.

## Services

- `src/ocr_service`: internal OCR API on `POST /v1/ocr`
- `src/llm_service`: internal grounded-answer API on `POST /v1/answer`
- `src/web_app`: public FastAPI app on port `8080`

## Runtime layout

- `docs/`: project documentation
- `configs/`: model-path configuration files for host and container runtimes
- `docker/`: service Dockerfiles and related build assets
- `data/`: runtime uploads, OCR outputs, SQLite state, and other local artifacts
- external models: stored in `/home/viettran_orin/models`
- `requirements/`: grouped Python dependency files by service
- `third_party/`: packaged host runtime dependencies such as `llama.cpp` binaries

## Running the stack

Use Docker Compose for the full application:

```bash
docker compose up --build web-app
```

Or use the helper script for the single Compose stack plus readiness checks:

```bash
./start_app.sh
```

Useful flags:

```bash
./start_app.sh --build
./start_app.sh --rebuild-deps
./start_app.sh --no-build
./start_app.sh --skip-public-check
./start_app.sh --local_test
```

`--build` rebuilds images with normal Docker layer caching. Use `--rebuild-deps` only when dependency installation layers should be refreshed.

Source code under `src/` is bind-mounted into the containers for local development, so code-only changes can usually be picked up with `./start_app.sh --no-build`.

`start_app.sh` also tracks dependency inputs locally and warns when files in `requirements/`, the service Dockerfiles in `docker/`, or the Paddle wheel input changed since the last successful `--rebuild-deps`.

Stop the stack:

```bash
./stop_app.sh
```

Optional shutdown flags:

```bash
./stop_app.sh --remove-volumes
./stop_app.sh --stop-cloudflared
```

## Environment

Set the main web-app settings in the project-root `.env` file:

```bash
WEB_APP_SECRET_KEY=replace-with-at-least-32-random-characters
WEB_APP_OWNER_EMAIL=your-email@example.com
WEB_APP_COOKIE_SECURE=1
WEB_APP_SMTP_HOST=smtp.example.com
WEB_APP_SMTP_USERNAME=mailer@example.com
WEB_APP_SMTP_PASSWORD=replace-me
```

`start_app.sh` loads `.env` automatically. `--local_test` forces `WEB_APP_COOKIE_SECURE=0`; otherwise startup forces `WEB_APP_COOKIE_SECURE=1`.

SMTP settings are required for normal signup verification. Set `WEB_APP_AUTH_DEBUG_CODES=1` only for local testing if verification codes should be exposed in the browser.

## Deployment notes

- Public traffic should reach only `web-app`.
- `ocr-service` and `llm-service` remain private Docker services.
- The web app protects against stale frontend assets by returning `no-store` headers for `/` and `/static/*` and by emitting versioned asset URLs.

Cloudflare Tunnel details:

- Domain: `jetsonocrai.cc`
- Tunnel name: `jetson-ocr-ai`
- Tunnel id: `a41bac72-717c-401b-a0c3-fa4f4cf2ac60`
- Local origin: `http://localhost:8080`
- Installed service config: `/etc/cloudflared/config.yml`

## Maintenance

If iterative rebuilds leave dangling Docker artifacts:

```bash
docker image prune -f
docker builder prune -f
```

Cloudflared is managed through systemd:

```bash
sudo systemctl status cloudflared
sudo systemctl restart cloudflared
```
