# Jetson OCR Project

Local OCR + LLM system for a Jetson Orin Nano Super.

Related docs:

- [Directory structure](./DIRECTORY_STRUCTURE.md)
- [Design history](./DESIGN_HISTORY.md)
- [Web-app session interface and system design](./WEB_APP_SESSION_INTERFACE_AND_SYSTEM_DESIGN.md)

This README is a concise project overview. The detailed browser UI and session design live in the dedicated web-app document.

## Current status

- `ocr-service` works locally and in the containerized runtime.
- `llm-service` works with Gemma GGUF and uses short one-shot defaults.
- The sample QA fixture in `data/ocr_markdown_run_v2/documents/question_0.md` is validated.
- `web-app` now provides the browser session layer, upload flow, OCR preview, grounded prompt flow, persistent recent sessions, cache-busted public assets for `jetsonocrai.cc`, and bulk session selection/deletion.

## System At A Glance

- `ocr-service`: image or PDF upload -> coordinate-arranged Markdown
- `llm-service`: OCR Markdown + user request -> concise grounded answer
- `web-app`: browser UI, session state, upload/preview/ask flow

## Module Structure

- `src/ocr_service`: internal OCR API on `POST /v1/ocr`.
- `src/llm_service`: internal grounded answer API on `POST /v1/answer`.
- `src/web_app`: public FastAPI web app on port `8080`, serving the browser UI and session APIs.

## Key Decisions

- Expose only the web app publicly.
- Keep OCR and LLM services internal to the container network.
- Use `jetsonocrai.cc` as the fixed public hostname for the browser application.
- Use SQLite plus local files for the first session store.
- Start with a single active session for the initial personal-use version.
- Keep Gemma thinking disabled and the OCR context small for one-shot requests.
- Keep OCR artifacts during development, but minimize them in operational mode.

## Running The Web App

Use Docker Compose for the three-container stack:

```bash
docker compose up --build web-app
```

Or start the full GPU runtime stack plus readiness checks (local APIs + cloudflared + public URL) in one command:

```bash
./start_app.sh
```

Optional flags:

```bash
./start_app.sh --build
./start_app.sh --skip-public-check
```

The local origin is `http://localhost:8080`, but production validation should be done against `https://jetsonocrai.cc`. Public traffic should go to `web-app`; OCR and LLM services are addressed internally by Compose service name.

## Public Access

The production public hostname is:

```text
https://jetsonocrai.cc
```

Cloudflare Tunnel terminates the public hostname and forwards traffic to the local `web-app` at `http://localhost:8080`. The `ocr-service` and `llm-service` remain private Docker services and are not exposed directly to the public internet.

The web app now protects against stale public UI shells by:

- returning `no-store` headers for `/` and `/static/*`
- emitting versioned CSS and JavaScript URLs from `/`
- serving uploaded-image preview responses inline from `GET /sessions/{id}/original`

Tunnel details:

- Cloudflare account domain: `jetsonocrai.cc`
- Cloudflare tunnel name: `jetson-ocr-ai`
- Cloudflare tunnel id: `a41bac72-717c-401b-a0c3-fa4f4cf2ac60`
- Public application: `web-app`
- Local origin: `http://localhost:8080`
- Installed service config: `/etc/cloudflared/config.yml`
- User config source: `/home/viettran_orin/.cloudflared/config.yml`

For the GPU-enabled runtime tested on Jetson Orin Nano, start the stack with:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu-test.yml up -d
```

After a rebuild, verify the public hostname instead of only the local origin:

```bash
curl -sS https://jetsonocrai.cc/ | head
curl -sS https://jetsonocrai.cc/sessions/recent
```

If build or test iterations created dangling Docker artifacts, clean them without taking down the running stack:

```bash
docker image prune -f
docker builder prune -f
```

The tunnel service is managed by systemd:

```bash
sudo systemctl status cloudflared
sudo systemctl restart cloudflared
```
# Jetson-ocr-ai
