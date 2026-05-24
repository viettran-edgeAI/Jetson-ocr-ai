# Web App

This document captures the browser interface and the supporting system design for the first `web-app` implementation.

The interface is based on the current browser application:

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
- collect user questions or task prompts before or after an upload
- call `llm-service`
- keep a small persistent session history scoped to the current registered user or guest identity

The current web app supports signed guest identities, verified local accounts with username-based account controls, email verification, TOTP two-factor login, user-scoped session history, and hourly OCR-upload limits by tier.

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
- Keep a utility area on the right for theme/help plus common site-shell actions such as login, logout, and related account links.
- Show authenticated users by username only in the header, using the username color to reflect tier instead of showing a separate avatar or tier badge.
- Clicking the username should open an account-management panel with email, tier, usage, and account actions.
- Use tier colors consistently: gray for guest, light blue for free, dark green for pro, and red for owner.

### Input panel

- Use a large drop zone with a dashed accent border.
- Support drag and drop, file picker upload, and clipboard paste for images.
- Show a clear upload button.
- Show the hourly OCR usage limit message in the upload area rather than beside the chat input.
- Display supported file types prominently.
- For the first version, the UI should only advertise formats the backend actually accepts: PNG, JPG, JPEG, and PDF.
- If a file type is not supported, fail fast with a clear message.
- When a file is already active in the session, block replacement until the user explicitly clicks `Start again` or removes the current thumbnail.

### Output panel

- Show an empty state before OCR completes.
- Render OCR Markdown directly once text is available.
- Keep the OCR result visible after an assistant answer is added.
- Provide a copy action for the OCR result, positioned at the bottom-right of the OCR result box.
- Keep the output card visually distinct from the input card.
- Provide an obvious place for the LLM answer after the prompt is submitted.
- Keep source references visible when possible.

### Prompt area

- Place a single-line prompt input under the output card.
- Include a primary send button.
- Include two quick actions:
  - `Answer a question`
  - `Solve this problem`
- Allow normal chat before any document is converted.
- Once a document is attached and successfully OCRed, append the OCR Markdown to the current session context automatically.

### Recent sessions

- Show recent uploads and runs for the current user or guest in a list.
- Include filename, file type, page or image count, and relative time.
- Allow reopening a prior session.
- Allow a Gmail-style selection mode so multiple old sessions can be selected and deleted together.
- Make the list informative enough that the user can resume work without re-uploading.

## Visual System

The interface should stay soft and technical rather than default admin-dashboard styling.

Recommended visual choices:

- white or near-white canvas with faint tinted gradients
- blue-violet accent color family
- soft shadows and thin borders
- rounded cards
- subtle iconography
- restrained motion for state transitions

## System Design

### Container boundary

The first release keeps three application containers:

- `web-app`
- `ocr-service`
- `llm-service`

Public traffic should reach only `web-app`.

### Public deployment

The fixed public hostname for the application is `jetsonocrai.cc`.

Cloudflare Tunnel routes `https://jetsonocrai.cc` to the local `web-app` origin at `http://localhost:8080`. The tunnel must not route public traffic directly to `ocr-service` or `llm-service`; those services stay internal and are reached only by `web-app` over the Docker network.

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
- Use OCR Markdown as the primary document context for the LLM request when a document is attached.

### LLM integration

- Send the current user request plus any stored OCR Markdown to `llm-service`.
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

- `users`
- `auth_sessions`
- `sessions`
- `messages`
- `rate_limit_events`

Suggested session fields:

- session id
- owner type and owner id
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
- emit versioned CSS and JavaScript URLs from `/`
- avoid offline cache layers such as service workers unless they are intentionally designed and tested
- serve thumbnail and original image responses inline so browser previews work in both the active session and recent sessions

## Session Flow

1. User opens the app.
2. The web app shows recent sessions and an empty active workspace.
3. User uploads a document or pastes an image.
4. The web app stores the file and creates or updates the current session record.
5. The web app sends the file to `ocr-service`.
6. The OCR Markdown is saved and rendered in the output panel.
7. User types a question or chooses a quick action. This may also happen before any upload.
8. The web app sends the prompt plus any available OCR Markdown to `llm-service`.
9. The answer is saved, rendered, and attached to the session.
10. The session remains available in the recent sessions list.

## Behavior And States

### Empty state

- Show the upload prompt on the left.
- Show a placeholder on the right inviting the user to chat or attach a document for OCR context.

### Uploading state

- Disable duplicate submissions.
- Show that OCR is in progress.
- Preserve the uploaded file and session metadata immediately.

### OCR success state

- Render the OCR Markdown.
- Keep page order and line order visible.
- Preserve line breaks and approximate horizontal structure.
- Keep the OCR card present while later chat messages render below it.
- If chat messages already exist, keep them in the same session and make the OCR Markdown available to subsequent LLM calls.

### Answering state

- Keep the prompt area active.
- Show a loading indicator while the LLM request is in flight.
- If the user selected Thinking mode, show a collapsible thinking panel immediately and stream concise reasoning updates into it without blocking animation or click handling.
- Preserve the OCR output while the answer is generated.
- Warm the LLM path on service startup so the first chat response is not substantially slower than later responses.

### Error state

- Show a concise error message.
- Preserve the uploaded document and session state when possible.
- Distinguish upload, OCR, and LLM failures.
- Avoid frontend hard failures when a browser tab still holds an older HTML shell during a deployment transition.

## API Shape

The web app should stay thin and use the existing internal APIs.

- `POST /v1/ocr` on `ocr-service`
- `POST /v1/answer` on `llm-service`

The web app itself can expose:

- `GET /` for the main interface
- `POST /sessions/chat` for creating a chat-only session
- `POST /sessions/upload` for file intake
- `POST /sessions/{id}/ask` for prompt submission
- `GET /sessions/{id}` for restoring a session
