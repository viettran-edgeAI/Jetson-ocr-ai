# Web-App Session Interface And System Design

This document captures the browser interface and the supporting system design for the first `web-app` implementation.

The interface is based on the current mockup:

- top header with app name, subtitle, and utility icons
- large upload and paste area on the left
- OCR/output area on the right
- prompt input and action buttons below the output area
- recent sessions list across the bottom

## Summary

The web app is the user-facing orchestration layer for the OCR-plus-LLM stack.

It should:

- accept image and PDF uploads
- call `ocr-service`
- render OCR Markdown for preview
- collect user questions or task prompts
- call `llm-service`
- keep a small persistent session history

The initial version is a single-user, personal-use application. The design should not block later multi-user expansion, but that is not part of the first implementation.

## Interface Requirements

### Overall layout

- Desktop layout uses a two-column workspace.
- The left column is the input surface.
- The right column is the output and prompt surface.
- Recent sessions are shown in a full-width section below.
- Mobile layout stacks the sections vertically.

### Header

- Show the product name: `OCR AI Assistant`.
- Show a short subtitle such as `Extract text and get intelligent answers`.
- Keep a minimal utility area on the right for theme/help actions.

### Input panel

- Use a large drop zone with a dashed accent border.
- Support drag and drop, file picker upload, and clipboard paste for images.
- Show a clear upload button.
- Display supported file types prominently.
- For the first version, the UI should only advertise formats the backend actually accepts: PNG, JPG, JPEG, and PDF.
- If a file type is not supported, fail fast with a clear message.
- When a file is already active in the session, block replacement until the user explicitly clicks `Start again` or removes the current thumbnail.

### Output panel

- Show an empty state before OCR completes.
- Render OCR Markdown directly once text is available.
- Keep the OCR result visible after an assistant answer is added.
- Provide a copy action for the OCR result.
- Keep the output card visually distinct from the input card.
- Provide an obvious place for the LLM answer after the prompt is submitted.
- Keep source references visible when possible.

### Prompt area

- Place a single-line prompt input under the output card.
- Include a primary send button.
- Include two quick actions:
  - `Answer a question`
  - `Solve this problem`
- Keep the prompt flow short and one-shot by default.

### Recent sessions

- Show recent uploads and runs in a list.
- Include filename, file type, page or image count, and relative time.
- Allow reopening a prior session.
- Make the list informative enough that the user can resume work without re-uploading.

## Visual System

The screenshot implies a soft, polished application shell. The implementation should follow that direction instead of default admin-dashboard styling.

Recommended visual choices:

- white or near-white canvas with faint tinted gradients
- blue-violet accent color family
- soft shadows and thin borders
- rounded cards
- subtle iconography
- restrained motion for state transitions

The UI should feel calm and technical, not busy.

## System Design

### Container boundary

The first release should keep three application containers:

- `web-app`
- `ocr-service`
- `llm-service`

Public traffic should reach only `web-app`.

### Public deployment

The fixed public hostname for the application is `jetsonocrai.cc`.

Cloudflare Tunnel should route `https://jetsonocrai.cc` to the local `web-app` origin at `http://localhost:8080`. The tunnel must not route public traffic directly to `ocr-service` or `llm-service`; those services stay internal and are reached only by `web-app` over the Docker network.

Deployment identity:

- Cloudflare tunnel name: `jetson-ocr-ai`
- Cloudflare tunnel id: `a41bac72-717c-401b-a0c3-fa4f4cf2ac60`
- Public application container: `web-app`
- Private backend containers: `ocr-service`, `llm-service`
- Installed tunnel service config: `/etc/cloudflared/config.yml`

### Web-app responsibilities

- serve the browser UI
- manage upload and session state
- call `ocr-service`
- call `llm-service`
- persist lightweight metadata
- render the user-facing conversation and OCR preview
- prevent stale public assets from leaving the browser on an older UI shell after deployment

### OCR integration

- Send uploads to `ocr-service` using the existing `POST /v1/ocr` contract.
- Store the returned Markdown with the session.
- Use OCR Markdown as the primary input for the LLM request.

### LLM integration

- Send the stored OCR Markdown plus the current user request to `llm-service`.
- Use `POST /v1/answer`.
- Persist the returned answer and latency metadata.

### Storage

Use local files plus SQLite.

Suggested file layout:

- uploaded originals
- OCR Markdown outputs
- optional answer transcripts
- thumbnails or previews if needed later

Suggested database tables:

- `sessions`
- `messages`

Suggested session fields:

- session id
- filename
- content type
- artifact paths
- status
- created timestamp
- updated timestamp

Suggested message fields:

- session id
- role
- content
- elapsed time
- token counts if present
- created timestamp

### Static asset delivery

Because the public entrypoint is `jetsonocrai.cc`, deployment safety must include asset invalidation.

The web app should:

- send `Cache-Control: no-store, no-cache, must-revalidate, max-age=0` for `/` and `/static/*`
- emit versioned CSS and JavaScript URLs from `/` so new deployments invalidate older browser shells
- avoid offline cache layers such as service workers unless they are intentionally designed and tested
- serve thumbnail/original image responses inline so browser previews work in both the active session and recent sessions

## Session Flow

1. User opens the app.
2. The web app shows recent sessions and an empty active workspace.
3. User uploads a document or pastes an image.
4. The web app stores the file and creates a session record.
5. The web app sends the file to `ocr-service`.
6. The OCR Markdown is saved and rendered in the output panel.
7. User types a question or chooses a quick action.
8. The web app sends OCR Markdown plus the prompt to `llm-service`.
9. The answer is saved, rendered, and attached to the session.
10. The session remains available in the recent sessions list.

## Behavior And States

### Empty state

- Show the upload prompt on the left.
- Show a placeholder on the right such as `OCR result will appear here`.

### Uploading state

- Disable duplicate submissions.
- Show that OCR is in progress.
- Preserve the uploaded file and session metadata immediately.

### OCR success state

- Render the OCR Markdown.
- Keep page order and line order visible.
- Preserve line breaks and approximate horizontal structure.
- Keep the OCR card present while later chat messages render below it.

### Answering state

- Keep the prompt area active.
- Show a loading indicator while the LLM request is in flight.
- Preserve the OCR output while the answer is generated.

### Error state

- Show a concise error message.
- Preserve the uploaded document and session state when possible.
- Distinguish upload, OCR, and LLM failures.
- Avoid frontend hard-failures when a browser tab still holds an older HTML shell during a deployment transition.

## API Shape

The web app should stay thin and use the existing internal APIs.

- `POST /v1/ocr` on `ocr-service`
- `POST /v1/answer` on `llm-service`

The web app itself can expose:

- `GET /` for the main interface
- `POST /sessions/upload` for file intake
- `POST /sessions/{id}/ask` for prompt submission
- `GET /sessions/{id}` for restoring a session
- `GET /sessions/recent` for the sidebar list

## Implementation Notes

- Use server-rendered HTML or a similarly simple approach for the first version.
- Keep browser state and server state aligned through session ids.
- Avoid overbuilding chat features before the upload-to-answer path works.
- Keep the first pass optimized for one active user and one active session.
- Keep the visual hierarchy close to the mockup, especially the upload and output panels.

## Acceptance Criteria

- The UI matches the intended layout and visual balance from the mockup.
- Uploading a PNG, JPG, JPEG, or PDF creates a session and triggers OCR.
- OCR Markdown renders in the output panel.
- OCR remains visible after the user asks a question.
- The OCR result can be copied from the UI.
- A new upload cannot replace the active session until the user explicitly starts again or removes the current file.
- A prompt submission triggers an LLM answer.
- Recent sessions persist across reloads.
- The design works on desktop and mobile.
- The implementation does not advertise unsupported input types.
- A fresh deployment does not leave the public hostname serving an older UI shell.
