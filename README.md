<h1 align="center">OCR AI Assistant</h1>

<p align="center">
  <strong>Fast local OCR, quick question answering, and multilingual translation on Jetson Orin Nano Super.</strong>
</p>

<p align="center">
  <a href="https://jetsonocrai.cc"><strong>jetsonocrai.cc</strong></a>
</p>

<p align="center">
  <img src="./docs/ocr_ai_demo.gif" alt="OCR AI Assistant demo" width="960" />
</p>

## Overview

OCR AI Assistant is a lightweight local OCR + LLM application built for fast document reading tasks. It extracts text from images and documents, helps answer short questions or multiple-choice quizzes, and translates OCR results across languages.

## Best For

- Screenshots, notes, exercises, and scanned pages.
- Short OCR tasks that need immediate answers.
- Translating extracted text between languages.
- Reading documents that mix normal text and formulas.

## Limitations

This project is optimized for quick and simple OCR workflows. Very large PDFs, dense academic papers, complex layouts, low-quality scans, or heavily formatted documents may require more processing time and can produce less accurate results.

## Architecture

<p align="center">
  <img src="./docs/ocr_ai_architecture.png" alt="OCR AI Assistant architecture" width="960" />
</p>

The application is split into three main services:

- **web_app** — browser interface, upload flow, sessions, and API calls.
- **ocr-service** — OCR pipeline for text, layout, and formula extraction.
- **llm-service** — local LLM service for answers, translation, and prompt handling.

## Documentation

For system design, service details, and workspace layout, see [docs/](./docs/).

## Model Configuration

Runtime model files are stored outside the repo in `/home/viettran_orin/models`.

- `configs/models.host.env` points local host runs to that model root.
- `configs/models.container.env` points Docker services to the same model root mounted at `/models`.
