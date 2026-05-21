const state = {
  activeSessionId: null,
  activeMode: null,
  isBusy: false,
  pendingFile: null,
  pendingPreviewUrl: null,
  currentMessages: [],
  currentMarkdown: "",
  hasDocument: false,
  selectionMode: false,
  selectedSessionIds: new Set(),
  recentSessionIds: [],
  account: null,
  authMode: "login",
};

let copyFeedbackTimer = null;

const els = {
  dropZone: document.querySelector("#dropZone"),
  fileInput: document.querySelector("#fileInput"),
  uploadButton: document.querySelector("#uploadButton"),
  convertButton: document.querySelector("#convertButton"),
  startAgainButton: document.querySelector("#startAgainButton"),
  selectedFilePreview: document.querySelector("#selectedFilePreview"),
  uploadState: document.querySelector("#uploadState"),
  dropTitle: document.querySelector("#dropTitle"),
  outputCard: document.querySelector("#outputCard"),
  emptyOutput: document.querySelector("#emptyOutput"),
  ocrPanel: document.querySelector("#ocrPanel"),
  ocrResult: document.querySelector("#ocrResult"),
  copyOcrButton: document.querySelector("#copyOcrButton"),
  copyOcrStatus: document.querySelector("#copyOcrStatus"),
  answerResult: document.querySelector("#answerResult"),
  promptForm: document.querySelector("#promptForm"),
  promptInput: document.querySelector("#promptInput"),
  promptLimitStatus: document.querySelector("#promptLimitStatus"),
  sendButton: document.querySelector("#sendButton"),
  sessionList: document.querySelector("#sessionList"),
  themeToggle: document.querySelector("#themeToggle"),
  accountStatus: document.querySelector("#accountStatus"),
  pricingButton: document.querySelector("#pricingButton"),
  loginButton: document.querySelector("#loginButton"),
  signupButton: document.querySelector("#signupButton"),
  logoutButton: document.querySelector("#logoutButton"),
  helpButton: document.querySelector("#helpButton"),
  selectSessionsButton: document.querySelector("#selectSessionsButton"),
  selectAllSessionsButton: document.querySelector("#selectAllSessionsButton"),
  deleteSelectedButton: document.querySelector("#deleteSelectedButton"),
  cancelSelectionButton: document.querySelector("#cancelSelectionButton"),
  selectionSummary: document.querySelector("#selectionSummary"),
  viewAllButton: document.querySelector("#viewAllButton"),
  authModal: document.querySelector("#authModal"),
  authForm: document.querySelector("#authForm"),
  authTitle: document.querySelector("#authTitle"),
  authEmail: document.querySelector("#authEmail"),
  authPassword: document.querySelector("#authPassword"),
  authSubmitButton: document.querySelector("#authSubmitButton"),
  authCloseButton: document.querySelector("#authCloseButton"),
  authStatus: document.querySelector("#authStatus"),
};

const iconFile = `
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M7 3.5h7l4 4v13H7z"></path>
    <path d="M14 3.7v4.1h4"></path>
    <path d="M10 12h5M10 15.5h5"></path>
  </svg>
`;

const iconMore = `
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 6h.01M12 12h.01M12 18h.01"></path>
  </svg>
`;

document.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  await loadAccountState();
  loadRecentSessions();
});

function bindEvents() {
  on(els.uploadButton, "click", () => els.fileInput?.click());
  on(els.convertButton, "click", convertSelectedFile);
  on(els.startAgainButton, "click", () => {
    if (state.isBusy) return;
    clearSelectedFile();
    setStatus("Ready for a new upload.");
  });
  on(els.copyOcrButton, "click", copyCurrentOcr);
  on(els.fileInput, "change", () => {
    const file = els.fileInput.files?.[0];
    if (file) attachFile(file);
    els.fileInput.value = "";
  });

  on(els.dropZone, "click", (event) => {
    if (event.target.closest("button")) return;
    if (!canAttachDocument()) {
      setStatus("Click Start again or remove the current file to load a new one.");
      return;
    }
    els.fileInput.click();
  });
  on(els.dropZone, "keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (!canAttachDocument()) {
        setStatus("Click Start again or remove the current file to load a new one.");
        return;
      }
      els.fileInput.click();
    }
  });
  on(els.dropZone, "dragover", (event) => {
    event.preventDefault();
    els.dropZone.classList.add("is-dragging");
  });
  on(els.dropZone, "dragleave", () => {
    els.dropZone.classList.remove("is-dragging");
  });
  on(els.dropZone, "drop", (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("is-dragging");
    if (!canAttachDocument()) {
      setStatus("Click Start again or remove the current file to load a new one.");
      return;
    }
    const file = event.dataTransfer?.files?.[0];
    if (file) attachFile(file);
  });

  document.addEventListener("paste", (event) => {
    const file = [...(event.clipboardData?.files || [])].find((item) =>
      item.type.startsWith("image/")
    );
    if (file) {
      event.preventDefault();
      if (!canAttachDocument()) {
        setStatus("Click Start again or remove the current file to load a new one.");
        return;
      }
      const pasted = new File([file], "pasted-image.png", { type: file.type || "image/png" });
      attachFile(pasted);
    }
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".session-menu-wrap")) {
      closeSessionMenus();
    }
  });

  on(els.promptForm, "submit", (event) => {
    event.preventDefault();
    askQuestion();
  });

  document.querySelectorAll(".quick-actions button").forEach((button) => {
    button.addEventListener("click", () => {
      submitQuickAction(button.dataset.mode);
    });
  });

  on(els.themeToggle, "click", () => {
    document.body.classList.toggle("dark");
  });
  on(els.pricingButton, "click", () => setStatus("Pricing is not configured."));
  on(els.loginButton, "click", () => openAuthModal("login"));
  on(els.signupButton, "click", () => openAuthModal("signup"));
  on(els.logoutButton, "click", logout);
  on(els.helpButton, "click", () => setStatus("Help center is not configured."));
  on(els.selectSessionsButton, "click", enterSelectionMode);
  on(els.selectAllSessionsButton, "click", toggleSelectAllSessions);
  on(els.cancelSelectionButton, "click", exitSelectionMode);
  on(els.deleteSelectedButton, "click", deleteSelectedSessions);

  on(els.viewAllButton, "click", loadRecentSessions);
  on(els.authCloseButton, "click", closeAuthModal);
  on(els.authModal, "click", (event) => {
    if (event.target === els.authModal) closeAuthModal();
  });
  on(els.authForm, "submit", submitAuthForm);
}

function attachFile(file) {
  if (state.isBusy) return;
  if (!canAttachDocument()) {
    setStatus("Click Start again or remove the current file to load a new one.");
    return;
  }
  const validation = validateFile(file);
  if (validation) {
    setStatus(validation, "error");
    return;
  }

  const attachToCurrentSession = Boolean(state.activeSessionId && !state.hasDocument);
  clearSelectedFile({
    keepSession: attachToCurrentSession,
    keepOutput: attachToCurrentSession,
  });
  state.pendingFile = file;
  if (file.type.startsWith("image/")) {
    state.pendingPreviewUrl = URL.createObjectURL(file);
  }
  renderSelectedFile({
    filename: file.name,
    contentType: file.type,
    previewUrl: state.pendingPreviewUrl,
    removable: true,
  });
  uploadFile(file);
}

function clearSelectedFile(options = {}) {
  if (state.pendingPreviewUrl) {
    URL.revokeObjectURL(state.pendingPreviewUrl);
  }
  state.pendingFile = null;
  state.pendingPreviewUrl = null;
  if (!options.keepSession) {
    state.activeSessionId = null;
    state.currentMessages = [];
    state.currentMarkdown = "";
    state.hasDocument = false;
  }
  if (!options.keepOutput) {
    clearOutput();
  }
  state.activeMode = null;
  els.promptInput.value = "";
  setQuickActionActive(null);
  clearCopyFeedback();
  els.selectedFilePreview.hidden = true;
  els.selectedFilePreview.innerHTML = "";
  els.convertButton.hidden = true;
  if (els.startAgainButton) {
    els.startAgainButton.hidden = true;
  }
  els.uploadButton.hidden = false;
  els.dropTitle.textContent = "Attach a document anytime";
  setStatus("");
}

function renderSelectedFile({ filename, contentType, previewUrl, removable }) {
  const isImage = String(contentType || "").startsWith("image/");
  const media = isImage && previewUrl
    ? `<img src="${previewUrl}" alt="" />`
    : iconFile;
  els.selectedFilePreview.innerHTML = `
    <div class="selected-thumb">${media}</div>
    <div class="selected-meta">
      <strong>${escapeHtml(filename)}</strong>
      <span>${escapeHtml(fileTypeLabel(filename, contentType))}</span>
    </div>
    ${removable ? `<button class="remove-file" type="button" aria-label="Remove selected file" title="Remove selected file">&times;</button>` : ""}
  `;
  els.selectedFilePreview.hidden = false;
  els.uploadButton.hidden = true;
  els.convertButton.hidden = true;
  if (els.startAgainButton) {
    els.startAgainButton.hidden = false;
  }
  els.dropTitle.textContent = filename;
  const remove = els.selectedFilePreview.querySelector(".remove-file");
  if (remove) {
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      clearSelectedFile();
    });
  }
}

async function convertSelectedFile() {
  if (!state.pendingFile || state.isBusy) return;
  await uploadFile(state.pendingFile);
}

async function uploadFile(file) {
  const attachToCurrentSession = Boolean(state.activeSessionId && !state.hasDocument);
  state.isBusy = true;
  setControlsBusy(true);
  setStatus(`Running OCR for ${file.name}...`);
  if (!attachToCurrentSession) {
    clearOutput();
  }

  const body = new FormData();
  body.append("file", file);
  const uploadUrl = attachToCurrentSession
    ? `/sessions/upload?session_id=${encodeURIComponent(state.activeSessionId)}`
    : "/sessions/upload";

  try {
    const response = await fetch(uploadUrl, { method: "POST", body });
    const data = await readJsonResponse(response);
    state.activeSessionId = data.id;
    state.pendingFile = null;
    if (state.pendingPreviewUrl) {
      URL.revokeObjectURL(state.pendingPreviewUrl);
      state.pendingPreviewUrl = null;
    }
    renderSession(data);
    const elapsedText = formatElapsedSeconds(data.ocr_elapsed_ms);
    const completeText = elapsedText ? `OCR complete in ${elapsedText}.` : "OCR complete.";
    setStatus(completeText, "success");
    if (data.rate_limit) {
      renderRateLimit(data.rate_limit);
    }
    await loadRecentSessions();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    state.isBusy = false;
    setControlsBusy(false);
  }
}

async function submitQuickAction(mode) {
  if (state.isBusy) return;
  state.activeMode = mode;
  setQuickActionActive(mode);
  const prompt = els.promptInput.value.trim() || quickActionPrompt(mode);
  await askQuestion({ prompt, mode });
}

async function askQuestion(options = {}) {
  const prompt = (options.prompt || els.promptInput.value).trim();
  const mode = options.mode ?? state.activeMode;
  if (!prompt) {
    els.promptInput.focus();
    return;
  }

  state.isBusy = true;
  setControlsBusy(true);

  try {
    await ensureChatSession();
    const optimisticMessages = [
      ...state.currentMessages,
      { role: "user", content: prompt },
      { role: "assistant", content: "Thinking..." },
    ];
    els.emptyOutput.hidden = true;
    renderMessages(optimisticMessages);
    const response = await fetch(`/sessions/${state.activeSessionId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, mode }),
    });
    const data = await readJsonResponse(response);
    renderSession(data);
    els.promptInput.value = "";
    state.activeMode = null;
    setQuickActionActive(null);
    await loadRecentSessions();
  } catch (error) {
    renderMessages([
      ...state.currentMessages,
      { role: "user", content: prompt },
      { role: "assistant", content: error.message, error: true },
    ]);
    setStatus(error.message, "error");
  } finally {
    state.isBusy = false;
    setControlsBusy(false);
  }
}

async function loadRecentSessions() {
  try {
    const response = await fetch("/sessions/recent");
    const data = await readJsonResponse(response);
    renderRecentSessions(data.sessions || []);
  } catch (error) {
    els.sessionList.innerHTML = `<div class="empty-list">${escapeHtml(error.message)}</div>`;
  }
}

async function loadAccountState() {
  try {
    const response = await fetch("/auth/me");
    const data = await readJsonResponse(response);
    state.account = data.identity;
    renderAccountState(data.identity);
    renderRateLimit(data.rate_limit);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function ensureChatSession() {
  if (state.activeSessionId) return state.activeSessionId;

  const response = await fetch("/sessions/chat", { method: "POST" });
  const data = await readJsonResponse(response);
  renderSession(data);
  await loadRecentSessions();
  return data.id;
}

function openAuthModal(mode) {
  state.authMode = mode;
  els.authTitle.textContent = mode === "signup" ? "Sign up" : "Log in";
  els.authSubmitButton.textContent = mode === "signup" ? "Sign up" : "Log in";
  els.authStatus.textContent = "";
  els.authStatus.className = "state-text";
  els.authEmail.value = "";
  els.authPassword.value = "";
  els.authModal.hidden = false;
  els.authEmail.focus();
}

function closeAuthModal() {
  els.authModal.hidden = true;
}

async function submitAuthForm(event) {
  event.preventDefault();
  const email = els.authEmail.value.trim();
  const password = els.authPassword.value;
  if (!email || !password) return;

  els.authSubmitButton.disabled = true;
  els.authStatus.textContent = state.authMode === "signup" ? "Creating account..." : "Logging in...";
  els.authStatus.className = "state-text";
  try {
    const response = await fetch(`/auth/${state.authMode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    await readJsonResponse(response);
    closeAuthModal();
    clearSelectedFile();
    await loadAccountState();
    await loadRecentSessions();
    setStatus(state.authMode === "signup" ? "Account created." : "Logged in.", "success");
  } catch (error) {
    els.authStatus.textContent = error.message;
    els.authStatus.className = "state-text error";
  } finally {
    els.authSubmitButton.disabled = false;
  }
}

async function logout() {
  try {
    await fetch("/auth/logout", { method: "POST" });
    clearSelectedFile();
    await loadAccountState();
    await loadRecentSessions();
    setStatus("Logged out.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function restoreSession(id) {
  if (state.isBusy) return;
  try {
    const response = await fetch(`/sessions/${id}`);
    const data = await readJsonResponse(response);
    state.activeSessionId = data.id;
    state.pendingFile = null;
    if (state.pendingPreviewUrl) {
      URL.revokeObjectURL(state.pendingPreviewUrl);
      state.pendingPreviewUrl = null;
    }
    renderSession(data);
    const label = data.has_document ? data.filename : "chat session";
    setStatus(`Restored ${label}.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function renameSession(id, currentName) {
  const filename = window.prompt("Rename session", currentName);
  if (filename === null) return;
  const cleaned = filename.trim();
  if (!cleaned) return;

  try {
    const response = await fetch(`/sessions/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: cleaned }),
    });
    const data = await readJsonResponse(response);
    if (state.activeSessionId === id) {
      renderSession(data);
    }
    await loadRecentSessions();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function deleteSession(id) {
  if (!window.confirm("Delete this session?")) return;
  try {
    const response = await fetch(`/sessions/${id}`, { method: "DELETE" });
    await readJsonResponse(response);
    if (state.activeSessionId === id) {
      clearSelectedFile();
    }
    await loadRecentSessions();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function deleteSelectedSessions() {
  const ids = [...state.selectedSessionIds];
  if (!ids.length) return;
  if (!window.confirm(`Delete ${ids.length} selected session${ids.length === 1 ? "" : "s"}?`)) return;
  try {
    state.isBusy = true;
    setControlsBusy(true);
    const response = await fetch("/sessions/bulk-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_ids: ids }),
    });
    const data = await readJsonResponse(response);
    if (ids.includes(state.activeSessionId)) {
      clearSelectedFile();
    }
    exitSelectionMode({ silent: true });
    await loadRecentSessions();
    setStatus(`Deleted ${data.deleted_count} session${data.deleted_count === 1 ? "" : "s"}.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    state.isBusy = false;
    setControlsBusy(false);
  }
}

function renderSession(session) {
  state.activeSessionId = session.id;
  state.currentMessages = session.messages || [];
  state.currentMarkdown = session.ocr_markdown || "";
  state.hasDocument = Boolean(session.has_document);
  if (state.hasDocument) {
    renderSelectedFile({
      filename: session.filename,
      contentType: session.content_type,
      previewUrl: session.thumbnail_url,
      removable: true,
    });
    els.convertButton.hidden = true;
    els.uploadButton.hidden = true;
  } else {
    renderChatOnlySessionShell();
  }

  if (state.currentMarkdown) {
    showOcrContent(renderMarkdown(state.currentMarkdown, { preserveWhitespace: true }));
  } else {
    hideOcrContent();
    clearCopyFeedback();
  }

  renderMessages(state.currentMessages);
  updateOutputPlaceholderVisibility();
}

function renderAccountState(identity) {
  const isAuthenticated = Boolean(identity?.authenticated);
  const tier = identity?.tier || "guest";
  if (els.accountStatus) {
    els.accountStatus.textContent = isAuthenticated
      ? `${identity.email} · ${tier.toUpperCase()}`
      : "Guest";
  }
  if (els.loginButton) els.loginButton.hidden = isAuthenticated;
  if (els.signupButton) els.signupButton.hidden = isAuthenticated;
  if (els.logoutButton) els.logoutButton.hidden = !isAuthenticated;
}

function renderRateLimit(rateLimit) {
  const text = rateLimitText(rateLimit);
  setLimitStatus(text);
}

function rateLimitText(rateLimit) {
  if (!rateLimit) return "";
  if (rateLimit.unlimited) {
    return "OCR uploads: unlimited.";
  }
  const remaining = Number(rateLimit.remaining);
  const limit = Number(rateLimit.limit);
  if (!Number.isFinite(remaining) || !Number.isFinite(limit)) return "";
  return `OCR uploads: ${remaining}/${limit} remaining this hour.`;
}

function setLimitStatus(message) {
  if (!els.promptLimitStatus) return;
  els.promptLimitStatus.textContent = message || "";
}

function renderChatOnlySessionShell() {
  els.selectedFilePreview.hidden = true;
  els.selectedFilePreview.innerHTML = "";
  els.convertButton.hidden = true;
  els.uploadButton.hidden = false;
  if (els.startAgainButton) {
    els.startAgainButton.hidden = !state.activeSessionId;
  }
  els.dropTitle.textContent = "Attach a document anytime";
}

function renderMessages(messages) {
  const visibleMessages = messages || [];
  if (!visibleMessages.length) {
    els.answerResult.hidden = true;
    els.answerResult.innerHTML = "";
    updateOutputPlaceholderVisibility();
    return;
  }

  els.answerResult.hidden = false;
  els.answerResult.innerHTML = visibleMessages
    .map((message) => {
      const role = message.role === "user" ? "user" : "assistant";
      const label = role === "user" ? "You" : "Jetson AI";
      const errorClass = message.error ? " is-error" : "";
      const content = role === "user" ? displayUserPrompt(message.content || "") : message.content || "";
      const speed = role === "assistant" ? answerSpeedText(message) : "";
      return `
        <article class="chat-message ${role}${errorClass}">
          <div class="chat-role">${label}</div>
          <div class="chat-bubble">
            <div class="chat-bubble-content">${renderMarkdown(content)}</div>
            ${speed ? `<div class="chat-bubble-meta">${escapeHtml(speed)}</div>` : ""}
          </div>
        </article>
      `;
    })
    .join("");
  updateOutputPlaceholderVisibility();
  els.outputCard.scrollTop = els.outputCard.scrollHeight;
}

function answerSpeedText(message) {
  const elapsedMs = Number(message.elapsed_ms);
  if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) return "";
  const completionTokens = Number(message.completion_tokens);
  const totalTokens = Number(message.total_tokens);
  const tokens = Number.isFinite(completionTokens) && completionTokens > 0
    ? completionTokens
    : Number.isFinite(totalTokens) && totalTokens > 0
      ? totalTokens
      : 0;
  if (tokens <= 0) return "";
  const perSecond = tokens / (elapsedMs / 1000);
  if (!Number.isFinite(perSecond) || perSecond <= 0) return "";
  return `${Math.round(perSecond)} tok/s`;
}

function renderRecentSessions(sessions) {
  state.recentSessionIds = (sessions || []).map((session) => session.id);
  reconcileSelectedSessions(sessions);
  updateSelectionUi();
  if (!sessions.length) {
    els.sessionList.innerHTML = `<div class="empty-list">Recent sessions will appear here.</div>`;
    return;
  }

  els.sessionList.innerHTML = sessions
    .map((session) => {
      const count = session.page_count || 1;
      const unit = session.file_type === "PDF" ? (count === 1 ? "page" : "pages") : "image";
      const subtitle = session.has_document
        ? `${escapeHtml(session.file_type)} &nbsp;•&nbsp; ${count} ${unit}`
        : "Chat session";
      const thumbnail = session.thumbnail_url
        ? `<img src="${session.thumbnail_url}" alt="" loading="lazy" />`
        : iconFile;
      const isSelected = state.selectedSessionIds.has(session.id);
      const selector = state.selectionMode
        ? `
          <label class="session-selector">
            <input type="checkbox" data-select-session="${session.id}" ${isSelected ? "checked" : ""} />
            <span></span>
          </label>
        `
        : "";
      return `
        <div class="session-row${isSelected ? " is-selected" : ""}${state.selectionMode ? " is-selecting" : ""}" data-session-id="${session.id}">
          ${selector}
          <button class="session-open" type="button" data-open-session="${session.id}">
            <span class="session-main">
              <span class="thumb">${thumbnail}</span>
              <span class="session-copy">
                <span class="session-title">${escapeHtml(session.filename)}</span>
                <span class="session-subtitle">${subtitle}</span>
              </span>
            </span>
          </button>
          <span class="session-meta">
            <span>${relativeTime(session.updated_at)}</span>
            <span class="session-menu-wrap">
              <button class="kebab" type="button" aria-label="Session options" title="Session options" data-menu-session="${session.id}">
                ${iconMore}
              </button>
              <span class="session-menu" hidden>
                <button type="button" data-rename-session="${session.id}" data-session-name="${escapeHtml(session.filename)}">Rename session</button>
                <button type="button" data-delete-session="${session.id}">Delete session</button>
              </span>
            </span>
          </span>
        </div>
      `;
    })
    .join("");

  els.sessionList.querySelectorAll("[data-open-session]").forEach((button) => {
    button.addEventListener("click", () => {
      if (state.selectionMode) {
        toggleSessionSelection(button.dataset.openSession);
        return;
      }
      restoreSession(button.dataset.openSession);
    });
  });
  els.sessionList.querySelectorAll("[data-select-session]").forEach((input) => {
    input.addEventListener("click", (event) => {
      event.stopPropagation();
    });
    input.addEventListener("change", () => {
      toggleSessionSelection(input.dataset.selectSession, input.checked);
    });
  });
  els.sessionList.querySelectorAll("[data-menu-session]").forEach((button) => {
    button.addEventListener("click", (event) => {
      if (state.selectionMode) {
        event.preventDefault();
        return;
      }
      event.stopPropagation();
      const menu = button.parentElement.querySelector(".session-menu");
      const willOpen = menu.hidden;
      closeSessionMenus();
      menu.hidden = !willOpen;
      button.closest(".session-row")?.classList.toggle("is-menu-open", willOpen);
    });
  });
  els.sessionList.querySelectorAll("[data-rename-session]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      closeSessionMenus();
      renameSession(button.dataset.renameSession, button.dataset.sessionName);
    });
  });
  els.sessionList.querySelectorAll("[data-delete-session]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      closeSessionMenus();
      deleteSession(button.dataset.deleteSession);
    });
  });
}

function closeSessionMenus() {
  els.sessionList.querySelectorAll(".session-menu").forEach((menu) => {
    menu.hidden = true;
  });
  els.sessionList.querySelectorAll(".session-row.is-menu-open").forEach((row) => {
    row.classList.remove("is-menu-open");
  });
}

function clearOutput() {
  state.currentMessages = [];
  state.currentMarkdown = "";
  hideOcrContent();
  els.answerResult.hidden = true;
  els.answerResult.innerHTML = "";
  clearCopyFeedback();
  updateOutputPlaceholderVisibility();
}

function setStatus(message, kind = "") {
  els.uploadState.textContent = message;
  els.uploadState.className = `state-text ${kind}`.trim();
}

function setControlsBusy(isBusy) {
  els.uploadButton.disabled = isBusy;
  els.convertButton.disabled = isBusy;
  if (els.startAgainButton) {
    els.startAgainButton.disabled = isBusy;
  }
  els.sendButton.disabled = isBusy;
  els.fileInput.disabled = isBusy;
  if (els.copyOcrButton) {
    els.copyOcrButton.disabled = isBusy;
  }
  if (els.selectSessionsButton) {
    els.selectSessionsButton.disabled = isBusy;
  }
  if (els.selectAllSessionsButton) {
    els.selectAllSessionsButton.disabled = isBusy || !(state.recentSessionIds || []).length;
  }
  if (els.deleteSelectedButton) {
    els.deleteSelectedButton.disabled = isBusy || !state.selectedSessionIds.size;
  }
  if (els.cancelSelectionButton) {
    els.cancelSelectionButton.disabled = isBusy;
  }
}

async function readJsonResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }
  return data;
}

function validateFile(file) {
  const allowedTypes = ["image/png", "image/jpeg", "application/pdf"];
  const allowedExtensions = [".png", ".jpg", ".jpeg", ".pdf"];
  const name = file.name.toLowerCase();
  const hasAllowedExtension = allowedExtensions.some((ext) => name.endsWith(ext));
  if (!allowedTypes.includes(file.type) && !hasAllowedExtension) {
    return "Only PNG, JPG, JPEG, and PDF uploads are supported.";
  }
  return "";
}

function renderMarkdown(value, options = {}) {
  const preserveWhitespace = Boolean(options.preserveWhitespace);
  const lines = String(value).split(/\r?\n/);
  const html = [];
  let listItems = [];
  let fenceMarker = null;
  let fenceLanguage = "";
  let fenceLines = [];

  const flushList = () => {
    if (listItems.length) {
      html.push(`<ul>${listItems.map((item) => `<li>${item}</li>`).join("")}</ul>`);
      listItems = [];
    }
  };

  const flushFence = () => {
    if (!fenceMarker) return;
    const className = fenceLanguage ? ` class="language-${escapeHtml(fenceLanguage)}"` : "";
    html.push(
      `<pre class="markdown-code"><code${className}>${escapeHtml(fenceLines.join("\n"))}</code></pre>`
    );
    fenceMarker = null;
    fenceLanguage = "";
    fenceLines = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (fenceMarker) {
      if (line === fenceMarker) {
        flushFence();
      } else {
        fenceLines.push(rawLine);
      }
      continue;
    }
    if (!line) {
      flushList();
      continue;
    }
    if (/^<!--.*-->$/.test(line)) {
      flushList();
      continue;
    }
    if (/^`{3,}/.test(line)) {
      flushList();
      fenceMarker = line.match(/^`+/)?.[0] || "```";
      fenceLanguage = line.slice(fenceMarker.length).trim();
      fenceLines = [];
      continue;
    }
    if (/^---+$/.test(line)) {
      flushList();
      html.push("<hr />");
      continue;
    }
    if (/^#{1,6}\s+/.test(line)) {
      flushList();
      const level = Math.min(line.match(/^#+/)?.[0].length || 3, 4);
      html.push(`<h${level}>${inlineMarkdown(escapeHtml(line.replace(/^#{1,6}\s+/, "")))}</h${level}>`);
    } else if (/^[-*]\s+/.test(line)) {
      listItems.push(inlineMarkdown(escapeHtml(line.replace(/^[-*]\s+/, ""))));
    } else {
      flushList();
      const content = preserveWhitespace ? rawLine.replace(/\t/g, "    ") : line;
      const escaped = inlineMarkdown(escapeHtml(content));
      const className = preserveWhitespace ? ` class="ocr-line"` : "";
      html.push(`<p${className}>${escaped}</p>`);
    }
  }
  flushFence();
  flushList();
  return html.join("") || "<p></p>";
}

function updateOutputPlaceholderVisibility() {
  const hasMarkdown = isOcrVisible() && Boolean(els.ocrResult?.innerHTML.trim());
  const hasMessages = !els.answerResult.hidden && Boolean(els.answerResult.innerHTML.trim());
  els.emptyOutput.hidden = hasMarkdown || hasMessages;
}

function setQuickActionActive(mode) {
  document.querySelectorAll(".quick-actions button").forEach((button) => {
    button.classList.toggle("is-active", mode && button.dataset.mode === mode);
  });
}

function quickActionPrompt(mode) {
  if (mode === "solve") return "Solve this problem";
  return "Answer this question";
}

function inlineMarkdown(value) {
  return value
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function fileTypeLabel(filename, contentType) {
  if (contentType === "application/x-chat-session") return "CHAT";
  const suffix = String(filename || "").split(".").pop();
  if (suffix && suffix !== filename) return suffix.toUpperCase();
  if (contentType === "application/pdf") return "PDF";
  if (String(contentType || "").startsWith("image/")) {
    return contentType.split("/", 2)[1].toUpperCase();
  }
  return "FILE";
}

function displayUserPrompt(value) {
  return String(value)
    .replace(/^Answer this question from the OCR text:\s*/i, "")
    .replace(/^Solve this problem using the OCR text:\s*/i, "")
    .replace(/^Answer this question:\s*/i, "")
    .replace(/^Solve this problem:\s*/i, "");
}

function canAttachDocument() {
  return !state.pendingFile && !state.hasDocument && !state.isBusy;
}

async function copyCurrentOcr() {
  if (!state.currentMarkdown) return;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(state.currentMarkdown);
    } else {
      fallbackCopyText(state.currentMarkdown);
    }
    setCopyFeedback("Copied");
  } catch (error) {
    setCopyFeedback("Copy failed");
  }
}

function fallbackCopyText(value) {
  const field = document.createElement("textarea");
  field.value = value;
  field.setAttribute("readonly", "readonly");
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.appendChild(field);
  field.select();
  document.execCommand("copy");
  document.body.removeChild(field);
}

function setCopyFeedback(message) {
  clearCopyFeedback();
  if (!els.copyOcrStatus) return;
  els.copyOcrStatus.textContent = message;
  if (!message) return;
  copyFeedbackTimer = window.setTimeout(() => {
    els.copyOcrStatus.textContent = "";
    copyFeedbackTimer = null;
  }, 1800);
}

function clearCopyFeedback() {
  if (copyFeedbackTimer) {
    window.clearTimeout(copyFeedbackTimer);
    copyFeedbackTimer = null;
  }
  if (els.copyOcrStatus) {
    els.copyOcrStatus.textContent = "";
  }
}

function enterSelectionMode() {
  state.selectionMode = true;
  state.selectedSessionIds.clear();
  closeSessionMenus();
  updateSelectionUi();
  loadRecentSessions();
}

function exitSelectionMode(options = {}) {
  state.selectionMode = false;
  state.selectedSessionIds.clear();
  updateSelectionUi();
  if (!options.silent) {
    loadRecentSessions();
  }
}

function toggleSessionSelection(id, forceChecked) {
  if (!id) return;
  const shouldSelect = forceChecked ?? !state.selectedSessionIds.has(id);
  if (shouldSelect) {
    state.selectedSessionIds.add(id);
  } else {
    state.selectedSessionIds.delete(id);
  }
  updateSelectionUi();
  renderSessionSelectionState();
}

function toggleSelectAllSessions() {
  if (!state.selectionMode) return;
  const visibleIds = state.recentSessionIds || [];
  if (!visibleIds.length) return;

  const allSelected = visibleIds.every((id) => state.selectedSessionIds.has(id));
  if (allSelected) {
    state.selectedSessionIds.clear();
  } else {
    visibleIds.forEach((id) => {
      state.selectedSessionIds.add(id);
    });
  }

  updateSelectionUi();
  renderSessionSelectionState();
}

function renderSessionSelectionState() {
  els.sessionList.querySelectorAll(".session-row").forEach((row) => {
    const id = row.dataset.sessionId;
    const isSelected = state.selectedSessionIds.has(id);
    row.classList.toggle("is-selected", isSelected);
    row.classList.toggle("is-selecting", state.selectionMode);
  });
  els.sessionList.querySelectorAll("[data-select-session]").forEach((input) => {
    input.checked = state.selectedSessionIds.has(input.dataset.selectSession);
  });
}

function updateSelectionUi() {
  if (els.selectSessionsButton) {
    els.selectSessionsButton.hidden = state.selectionMode;
  }
  if (els.deleteSelectedButton) {
    els.deleteSelectedButton.hidden = !state.selectionMode;
    els.deleteSelectedButton.disabled = state.isBusy || !state.selectedSessionIds.size;
  }
  if (els.selectAllSessionsButton) {
    const visibleIds = state.recentSessionIds || [];
    const hasVisible = visibleIds.length > 0;
    const allSelected = hasVisible && visibleIds.every((id) => state.selectedSessionIds.has(id));
    els.selectAllSessionsButton.hidden = !state.selectionMode;
    els.selectAllSessionsButton.disabled = state.isBusy || !hasVisible;
    els.selectAllSessionsButton.textContent = allSelected ? "Clear all" : "Select all";
  }
  if (els.cancelSelectionButton) {
    els.cancelSelectionButton.hidden = !state.selectionMode;
  }
  if (els.selectionSummary) {
    const count = state.selectedSessionIds.size;
    els.selectionSummary.hidden = !state.selectionMode;
    els.selectionSummary.textContent = `${count} selected`;
  }
}

function reconcileSelectedSessions(sessions) {
  const validIds = new Set((sessions || []).map((session) => session.id));
  state.selectedSessionIds.forEach((id) => {
    if (!validIds.has(id)) {
      state.selectedSessionIds.delete(id);
    }
  });
}

function showOcrContent(html) {
  if (!els.ocrResult) return;
  els.ocrResult.innerHTML = html;
  if (els.ocrPanel) {
    els.ocrPanel.hidden = false;
  } else {
    els.ocrResult.hidden = false;
  }
}

function hideOcrContent() {
  if (!els.ocrResult) return;
  els.ocrResult.innerHTML = "";
  if (els.ocrPanel) {
    els.ocrPanel.hidden = true;
  } else {
    els.ocrResult.hidden = true;
  }
}

function isOcrVisible() {
  if (els.ocrPanel) {
    return !els.ocrPanel.hidden;
  }
  return Boolean(els.ocrResult) && !els.ocrResult.hidden;
}

function on(element, eventName, handler) {
  if (!element) return;
  element.addEventListener(eventName, handler);
}

function relativeTime(value) {
  const then = new Date(value);
  if (Number.isNaN(then.getTime())) return "";
  const seconds = Math.max(1, Math.floor((Date.now() - then.getTime()) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return then.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  const days = Math.floor(hours / 24);
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days} days ago`;
  return then.toLocaleDateString([], { month: "short", day: "numeric" });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatElapsedSeconds(elapsedMs) {
  const ms = Number(elapsedMs);
  if (!Number.isFinite(ms) || ms < 0) return "";
  return `${(ms / 1000).toFixed(1)}s`;
}
