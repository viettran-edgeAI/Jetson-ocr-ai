const state = {
  activeSessionId: null,
  activeMode: null,
  translateLanguage: "Vietnamese",
  thinkingMode: "fast",
  isBusy: false,
  pendingFile: null,
  pendingPreviewUrl: null,
  currentMessages: [],
  currentMarkdown: "",
  hasDocument: false,
  selectionMode: false,
  selectedSessionIds: new Set(),
  recentSessionIds: [],
  showAllSessions: false,
  account: null,
  accountDetails: null,
  rateLimit: null,
  authMode: "login",
  signupPendingId: null,
};

let copyFeedbackTimer = null;
let mathRenderFrame = 0;

const els = {
  dropZone: document.querySelector("#dropZone"),
  fileInput: document.querySelector("#fileInput"),
  uploadButton: document.querySelector("#uploadButton"),
  convertButton: document.querySelector("#convertButton"),
  startAgainButton: document.querySelector("#startAgainButton"),
  selectedFilePreview: document.querySelector("#selectedFilePreview"),
  uploadState: document.querySelector("#uploadState"),
  uploadLimitStatus: document.querySelector("#uploadLimitStatus"),
  dropTitle: document.querySelector("#dropTitle"),
  outputCard: document.querySelector("#outputCard"),
  emptyOutput: document.querySelector("#emptyOutput"),
  ocrPanel: document.querySelector("#ocrPanel"),
  ocrResult: document.querySelector("#ocrResult"),
  copyOcrButton: document.querySelector("#copyOcrButton"),
  copyOcrStatus: document.querySelector("#copyOcrStatus"),
  answerResult: document.querySelector("#answerResult"),
  promptForm: document.querySelector("#promptForm"),
  thinkingModeWrap: document.querySelector("#thinkingModeWrap"),
  thinkingModeButton: document.querySelector("#thinkingModeButton"),
  thinkingModeLabel: document.querySelector("#thinkingModeLabel"),
  thinkingModeMenu: document.querySelector("#thinkingModeMenu"),
  translateLanguageButton: document.querySelector("#translateLanguageButton"),
  translateLanguageLabel: document.querySelector("#translateLanguageLabel"),
  translateLanguageMenu: document.querySelector("#translateLanguageMenu"),
  translateQuickAction: document.querySelector("#translateQuickAction"),
  promptInput: document.querySelector("#promptInput"),
  sendButton: document.querySelector("#sendButton"),
  sessionList: document.querySelector("#sessionList"),
  themeToggle: document.querySelector("#themeToggle"),
  accountStatus: document.querySelector("#accountStatus"),
  accountName: document.querySelector("#accountName"),
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
  authCredentialsFields: document.querySelector("#authCredentialsFields"),
  authUsernameWrap: document.querySelector("#authUsernameWrap"),
  authUsername: document.querySelector("#authUsername"),
  authEmail: document.querySelector("#authEmail"),
  authPassword: document.querySelector("#authPassword"),
  authPasswordConfirmWrap: document.querySelector("#authPasswordConfirmWrap"),
  authPasswordConfirm: document.querySelector("#authPasswordConfirm"),
  authWebsite: document.querySelector("#authWebsite"),
  emailVerificationFields: document.querySelector("#emailVerificationFields"),
  emailVerificationCode: document.querySelector("#emailVerificationCode"),
  authSubmitButton: document.querySelector("#authSubmitButton"),
  authCloseButton: document.querySelector("#authCloseButton"),
  authStatus: document.querySelector("#authStatus"),
  accountModal: document.querySelector("#accountModal"),
  accountCloseButton: document.querySelector("#accountCloseButton"),
  accountPanelUsername: document.querySelector("#accountPanelUsername"),
  accountTierChip: document.querySelector("#accountTierChip"),
  accountEmail: document.querySelector("#accountEmail"),
  accountTierValue: document.querySelector("#accountTierValue"),
  accountUsage: document.querySelector("#accountUsage"),
  accountUpgradeButton: document.querySelector("#accountUpgradeButton"),
  accountDeleteButton: document.querySelector("#accountDeleteButton"),
  accountStatusMessage: document.querySelector("#accountStatusMessage"),
  helpModal: document.querySelector("#helpModal"),
  helpCloseButton: document.querySelector("#helpCloseButton"),
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
  ensureDynamicChatStyles();
  bindEvents();
  hideOcrContent();
  await loadAccountState();
  loadRecentSessions();
  queueMathRender(document.body);
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
    if (!event.target.closest(".thinking-mode-wrap")) {
      closeThinkingModeMenu();
    }
    if (!event.target.closest(".quick-action-language")) {
      closeTranslateLanguageMenu();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeThinkingModeMenu();
      closeTranslateLanguageMenu();
    }
  });

  on(els.promptForm, "submit", (event) => {
    event.preventDefault();
    askQuestion();
  });
  on(els.thinkingModeButton, "click", (event) => {
    event.stopPropagation();
    toggleThinkingModeMenu();
  });
  els.thinkingModeMenu?.querySelectorAll("[data-thinking-mode]").forEach((button) => {
    on(button, "click", () => {
      const mode = button.dataset.thinkingMode;
      if (mode === "fast" || mode === "thinking") {
        setThinkingMode(mode);
      }
      closeThinkingModeMenu();
    });
  });

  document.querySelectorAll(".quick-actions [data-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      submitQuickAction(button.dataset.mode);
    });
  });
  on(els.translateLanguageButton, "click", (event) => {
    event.stopPropagation();
    toggleTranslateLanguageMenu();
  });
  els.translateLanguageMenu?.querySelectorAll("[data-translate-language]").forEach((button) => {
    on(button, "click", () => {
      const language = String(button.dataset.translateLanguage || "").trim();
      if (language) {
        setTranslateLanguage(language);
      }
      closeTranslateLanguageMenu();
    });
  });

  on(els.themeToggle, "click", () => {
    document.body.classList.toggle("dark");
  });
  on(els.accountStatus, "click", openAccountModal);
  on(els.pricingButton, "click", () => setStatus("Pricing is not configured."));
  on(els.loginButton, "click", () => {
    openAuthModal("login").catch((error) => setStatus(error.message, "error"));
  });
  on(els.signupButton, "click", () => {
    openAuthModal("signup").catch((error) => setStatus(error.message, "error"));
  });
  on(els.logoutButton, "click", logout);
  on(els.helpButton, "click", openHelpModal);
  on(els.selectSessionsButton, "click", enterSelectionMode);
  on(els.selectAllSessionsButton, "click", toggleSelectAllSessions);
  on(els.cancelSelectionButton, "click", exitSelectionMode);
  on(els.deleteSelectedButton, "click", deleteSelectedSessions);

  on(els.viewAllButton, "click", () => {
    state.showAllSessions = true;
    loadRecentSessions();
  });
  on(els.authCloseButton, "click", closeAuthModal);
  on(els.authForm, "submit", submitAuthForm);
  on(els.accountCloseButton, "click", closeAccountModal);
  on(els.accountUpgradeButton, "click", () => setAccountStatus("Account upgrades are not configured yet."));
  on(els.accountDeleteButton, "click", () => setAccountStatus("Account deletion is not configured yet."));
  on(els.helpCloseButton, "click", closeHelpModal);

  setThinkingMode(state.thinkingMode);
  setTranslateLanguage(state.translateLanguage);
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
  state.translateLanguage = "Vietnamese";
  els.promptInput.value = "";
  setQuickActionActive(null);
  setTranslateLanguage(state.translateLanguage);
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
  const resolvedMode = normalizeQuickActionMode(mode);
  state.activeMode = resolvedMode;
  setQuickActionActive(resolvedMode);
  const prompt = els.promptInput.value.trim() || quickActionPrompt(resolvedMode);
  await askQuestion({ prompt, mode: resolvedMode });
}

async function askQuestion(options = {}) {
  const prompt = (options.prompt || els.promptInput.value).trim();
  const mode = options.mode ?? state.activeMode;
  const allowEmptyPrompt = !prompt && state.hasDocument && !state.currentMessages.length;
  if (!prompt && !allowEmptyPrompt) {
    els.promptInput.focus();
    return;
  }

  state.isBusy = true;
  setControlsBusy(true);

  try {
    await ensureChatSession();
    const userMessage = prompt ? [{ role: "user", content: prompt }] : [];
    const useThinkingTrace = state.thinkingMode === "thinking";
    const optimisticMessages = [
      ...state.currentMessages,
      ...userMessage,
      {
        role: "assistant",
        content: "",
        answer_complete: false,
        thinking_trace: useThinkingTrace ? "" : undefined,
        thinking_in_progress: useThinkingTrace,
        thinking_open: false,
      },
    ];
    const assistantIndex = optimisticMessages.length - 1;
    let finalSession = null;
    els.emptyOutput.hidden = true;
    renderMessages(optimisticMessages);
    const streamUpdater = createStreamingMessageUpdater(optimisticMessages, assistantIndex);

    await streamAskResponse({
      sessionId: state.activeSessionId,
      payload: {
        prompt,
        mode,
        thinking_mode: state.thinkingMode,
      },
      onToken(delta, kind) {
        const tokenKind = kind || (useThinkingTrace ? "reasoning" : "answer");
        if (tokenKind === "reasoning") {
          if (!useThinkingTrace) {
            return;
          }
          optimisticMessages[assistantIndex].thinking_trace += delta;
        } else {
          optimisticMessages[assistantIndex].content += delta;
        }
        streamUpdater.queue(tokenKind);
      },
      onDone(data) {
        const answerText = String(data?.answer || "").trim();
        const reasoningText = String(data?.reasoning_text || optimisticMessages[assistantIndex].thinking_trace || "").trim();
        optimisticMessages[assistantIndex].thinking_in_progress = false;
        if (useThinkingTrace && reasoningText) {
          optimisticMessages[assistantIndex].thinking_trace = reasoningText;
        }
        if (answerText) {
          optimisticMessages[assistantIndex].content = answerText;
        }
        optimisticMessages[assistantIndex].answer_complete = true;
        optimisticMessages[assistantIndex].elapsed_ms = toFiniteNumberOrUndefined(data?.elapsed_ms);
        optimisticMessages[assistantIndex].prompt_tokens = toFiniteNumberOrUndefined(data?.prompt_tokens);
        optimisticMessages[assistantIndex].completion_tokens = toFiniteNumberOrUndefined(data?.completion_tokens);
        optimisticMessages[assistantIndex].total_tokens = toFiniteNumberOrUndefined(data?.total_tokens);
        if (data?.session && typeof data.session === "object") {
          finalSession = data.session;
        }
        streamUpdater.flush();
        renderMessages(optimisticMessages);
      },
    });

    const finalThinkingTrace = useThinkingTrace
      ? String(optimisticMessages[assistantIndex].thinking_trace || "").trim()
      : "";
    if (finalSession) {
      decorateSessionWithThinkingTrace(finalSession, finalThinkingTrace);
      renderSession(finalSession);
    } else {
      const response = await fetch(`/sessions/${state.activeSessionId}`);
      const data = await readJsonResponse(response);
      decorateSessionWithThinkingTrace(data, finalThinkingTrace);
      renderSession(data);
    }
    els.promptInput.value = "";
    state.activeMode = null;
    setQuickActionActive(null);
    await loadRecentSessions();
  } catch (error) {
    renderMessages([
      ...state.currentMessages,
      ...(prompt ? [{ role: "user", content: prompt }] : []),
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
    const query = state.showAllSessions ? "?include_all=1" : "";
    const response = await fetch(`/sessions/recent${query}`);
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
    state.accountDetails = null;
    state.rateLimit = data.rate_limit || null;
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

async function openAuthModal(mode) {
  state.authMode = mode;
  state.signupPendingId = null;
  els.authTitle.textContent = mode === "signup" ? "Sign up" : "Log in";
  els.authSubmitButton.textContent = mode === "signup" ? "Sign up" : "Log in";
  els.authStatus.textContent = "";
  els.authStatus.className = "state-text";
  els.authUsername.value = "";
  els.authEmail.value = "";
  els.authPassword.value = "";
  if (els.authPasswordConfirm) {
    els.authPasswordConfirm.value = "";
  }
  els.authWebsite.value = "";
  els.emailVerificationCode.value = "";
  renderAuthStage(mode === "signup" ? "signup" : "login");
  els.authModal.hidden = false;
  if (mode === "signup") {
    els.authUsername.focus();
  } else {
    els.authEmail.focus();
  }
}

function closeAuthModal() {
  els.authModal.hidden = true;
}

async function submitAuthForm(event) {
  event.preventDefault();
  els.authSubmitButton.disabled = true;
  els.authStatus.textContent = authBusyText();
  els.authStatus.className = "state-text";
  try {
    if (state.authMode === "signup") {
      await startSignup();
    } else if (state.authMode === "signup-verify") {
      await verifySignupEmail();
    } else {
      await startLogin();
    }
  } catch (error) {
    els.authStatus.textContent = error.message;
    els.authStatus.className = "state-text error";
  } finally {
    els.authSubmitButton.disabled = false;
  }
}

function renderAuthStage(stage) {
  state.authMode = stage;
  const isSignupStart = stage === "signup";
  els.authCredentialsFields.hidden = !(stage === "signup" || stage === "login");
  els.emailVerificationFields.hidden = stage !== "signup-verify";
  document.querySelectorAll(".signup-only").forEach((node) => {
    node.hidden = !isSignupStart;
  });
  if (els.authPassword) {
    els.authPassword.autocomplete = isSignupStart ? "new-password" : "current-password";
  }
  const titles = {
    signup: "Sign up",
    "signup-verify": "Verify email",
    login: "Log in",
  };
  const buttons = {
    signup: "Send verification code",
    "signup-verify": "Verify email",
    login: "Log in",
  };
  els.authTitle.textContent = titles[stage] || "Log in";
  els.authSubmitButton.textContent = buttons[stage] || "Continue";
}

function authBusyText() {
  return {
    signup: "Creating signup session...",
    "signup-verify": "Verifying email...",
    login: "Checking credentials...",
  }[state.authMode] || "Working...";
}

async function startSignup() {
  if (els.authPassword.value !== (els.authPasswordConfirm?.value || "")) {
    throw new Error("Passwords do not match.");
  }
  const payload = {
    username: els.authUsername.value.trim(),
    email: els.authEmail.value.trim(),
    password: els.authPassword.value,
    website: els.authWebsite.value,
  };
  const response = await fetch("/auth/signup/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await readJsonResponse(response);
  state.signupPendingId = data.pending_id;
  renderAuthStage("signup-verify");
  els.authStatus.textContent = data.verification_code
    ? `Verification code: ${data.verification_code}`
    : "Verification code sent.";
  els.emailVerificationCode.focus();
}

async function verifySignupEmail() {
  const response = await fetch("/auth/signup/verify-email", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      pending_id: state.signupPendingId,
      code: els.emailVerificationCode.value.trim(),
    }),
  });
  await readJsonResponse(response);
  await completeSignup();
}

async function completeSignup() {
  const response = await fetch("/auth/signup/complete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      pending_id: state.signupPendingId,
    }),
  });
  await readJsonResponse(response);
  closeAuthModal();
  clearSelectedFile();
  await loadAccountState();
  await loadRecentSessions();
  setStatus("Account created.", "success");
}

async function startLogin() {
  const email = els.authEmail.value.trim();
  const password = els.authPassword.value;
  if (!email || !password) return;
  const response = await fetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  await readJsonResponse(response);
  closeAuthModal();
  clearSelectedFile();
  await loadAccountState();
  await loadRecentSessions();
  setStatus("Logged in.", "success");
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

  hideOcrContent();
  clearCopyFeedback();

  renderMessages(state.currentMessages);
  updateOutputPlaceholderVisibility();
}

function renderAccountState(identity) {
  const isAuthenticated = Boolean(identity?.authenticated);
  const tier = identity?.tier || "guest";
  const username = identity?.username || (isAuthenticated ? "user" : "Guest");
  if (els.accountStatus) {
    els.accountStatus.className = `account-status tier-${tierColorClass(tier)}`;
    els.accountStatus.disabled = false;
  }
  if (els.accountName) {
    els.accountName.textContent = username;
  }
  if (els.loginButton) els.loginButton.hidden = isAuthenticated;
  if (els.signupButton) els.signupButton.hidden = isAuthenticated;
  if (els.logoutButton) els.logoutButton.hidden = !isAuthenticated;
}

function renderRateLimit(rateLimit) {
  state.rateLimit = rateLimit || null;
  const text = rateLimitText(rateLimit);
  setLimitStatus(text);
  if (!els.accountModal?.hidden) {
    renderAccountModalBody();
  }
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

function tierColorClass(tier) {
  return {
    guest: "gray",
    free: "light-blue",
    pro: "dark-green",
    owner: "red",
  }[tier] || "gray";
}

function setLimitStatus(message) {
  if (!els.uploadLimitStatus) return;
  els.uploadLimitStatus.textContent = message || "";
}

async function openAccountModal() {
  setAccountStatus("");
  if (state.account?.authenticated) {
    try {
      const response = await fetch("/account");
      const data = await readJsonResponse(response);
      state.accountDetails = data.account || null;
    } catch (error) {
      state.accountDetails = null;
      setAccountStatus(error.message, "error");
    }
  } else {
    state.accountDetails = null;
  }
  renderAccountModalBody();
  els.accountModal.hidden = false;
}

function closeAccountModal() {
  els.accountModal.hidden = true;
}

function openHelpModal() {
  els.helpModal.hidden = false;
}

function closeHelpModal() {
  els.helpModal.hidden = true;
}

function renderAccountModalBody() {
  const identity = state.account || {};
  const tier = identity.tier || "guest";
  const username = identity.username || "Guest";
  const isAuthenticated = Boolean(identity.authenticated);
  const details = state.accountDetails;
  const rateLimit = details?.rate_limit || state.rateLimit;

  els.accountPanelUsername.textContent = details?.username || username;
  els.accountTierChip.className = `account-chip tier-${tierColorClass(tier)}`;
  els.accountTierChip.textContent = tier.toUpperCase();
  els.accountEmail.textContent = details?.email || "Sign in to view your account email.";
  els.accountTierValue.textContent = prettyTierName(details?.tier || tier);
  els.accountUsage.textContent = rateLimitText(rateLimit) || "OCR uploads: unavailable.";
  els.accountUpgradeButton.hidden = !isAuthenticated;
  els.accountDeleteButton.hidden = !isAuthenticated;
}

function setAccountStatus(message, tone = "") {
  if (!els.accountStatusMessage) return;
  els.accountStatusMessage.textContent = message || "";
  els.accountStatusMessage.className = `state-text${tone ? ` ${tone}` : ""}`;
}

function prettyTierName(tier) {
  return {
    guest: "Guest",
    free: "Free",
    pro: "Pro",
    owner: "Owner",
  }[tier] || "Unknown";
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
  const visibleMessages = buildVisibleMessages(messages);
  if (!visibleMessages.length) {
    els.answerResult.hidden = true;
    els.answerResult.innerHTML = "";
    updateOutputPlaceholderVisibility();
    return;
  }

  els.answerResult.hidden = false;
  els.answerResult.innerHTML = visibleMessages
    .map((message, index) => {
      const role = message.role === "user" ? "user" : "assistant";
      const label = role === "assistant" ? "Jetson AI" : "";
      const errorClass = message.error ? " is-error" : "";
      const content = role === "user"
        ? message.is_ocr_result
          ? String(message.content || "")
          : displayUserPrompt(message.content || "")
        : message.content || "";
      const speed = role === "assistant" ? answerSpeedText(message) : "";
      const showCopyButton = role === "assistant"
        && Boolean(String(message.content || "").trim())
        && isAssistantMessageComplete(message, index, visibleMessages);
      const thinkingTrace = role === "assistant" ? String(message.thinking_trace || "").trim() : "";
      const hasThinkingTrace = Boolean(thinkingTrace);
      const thinkingInProgress = Boolean(role === "assistant" && message.thinking_in_progress);
      const showThinkingBox = thinkingInProgress || hasThinkingTrace;
      const thinkingOpen = Boolean(role === "assistant" && message.thinking_open);
      const hasFooter = role === "assistant" && (showCopyButton || speed);
      const thinkingBox = showThinkingBox
        ? `
          <section class="thinking-trace ${thinkingInProgress ? "is-live" : ""} ${thinkingOpen ? "is-open" : ""}" data-thinking-trace>
            <button
              type="button"
              class="thinking-trace-toggle"
              data-thinking-toggle="${index}"
              aria-expanded="${thinkingOpen ? "true" : "false"}"
            >
              <span class="thinking-trace-label">${thinkingInProgress ? "Thinking..." : "Thinking process"}</span>
              ${
                thinkingInProgress
                  ? `<span class="thinking-dots" aria-hidden="true"><span></span><span></span><span></span></span>`
                  : `<span class="thinking-trace-state">done</span>`
              }
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="m7 10 5 5 5-5" />
              </svg>
            </button>
            <div class="thinking-trace-panel" data-thinking-panel ${thinkingOpen ? "" : "hidden"}>
              <div class="thinking-trace-content" data-thinking-content>
                ${renderThinkingTraceContent(thinkingTrace)}
              </div>
            </div>
          </section>
        `
        : "";
      return `
        <article class="chat-message ${role}${errorClass}" data-message-index="${index}" data-message-role="${role}">
          ${label ? `<div class="chat-role">${label}</div>` : ""}
          <div class="chat-bubble">
            ${thinkingBox}
            ${
              role === "assistant"
                ? `<div class="chat-bubble-content" data-chat-content ${content ? "" : "hidden"}>${content ? renderMarkdown(content) : ""}</div>`
                : content
                  ? `<div class="chat-bubble-content">${renderMarkdown(content, { preserveWhitespace: Boolean(message.is_ocr_result) })}</div>`
                  : ""
            }
            ${
              hasFooter
                ? `
                  <div class="chat-bubble-footer">
                    ${
                      showCopyButton
                        ? `<button type="button" class="copy-answer-button" data-copy-answer="${index}">Copy</button>`
                        : "<span></span>"
                    }
                    ${speed ? `<div class="chat-bubble-meta" data-chat-meta>${escapeHtml(speed)}</div>` : ""}
                  </div>
                `
                : ""
            }
          </div>
        </article>
      `;
    })
    .join("");
  els.answerResult.querySelectorAll("[data-thinking-toggle]").forEach((button) => {
    on(button, "click", () => {
      const index = Number(button.dataset.thinkingToggle);
      if (!Number.isInteger(index) || index < 0 || index >= visibleMessages.length) {
        return;
      }
      const message = visibleMessages[index];
      if (!message || message.role !== "assistant") {
        return;
      }
      message.thinking_open = !Boolean(message.thinking_open);
      setThinkingTraceOpen(button, message);
    });
  });
  els.answerResult.querySelectorAll("[data-copy-answer]").forEach((button) => {
    on(button, "click", async () => {
      const index = Number(button.dataset.copyAnswer);
      if (!Number.isInteger(index) || index < 0 || index >= visibleMessages.length) {
        return;
      }
      const message = visibleMessages[index];
      const content = String(message?.content || "").trim();
      if (!content) return;
      try {
        await copyText(content);
        setCopyButtonFeedback(button, "Copied");
      } catch (_error) {
        setCopyButtonFeedback(button, "Failed");
      }
    });
  });
  queueMathRender(els.answerResult);
  updateOutputPlaceholderVisibility();
  els.outputCard.scrollTop = els.outputCard.scrollHeight;
}

function createStreamingMessageUpdater(messages, assistantIndex) {
  const pending = {
    answer: false,
    reasoning: false,
  };
  let frameQueued = false;

  function apply() {
    frameQueued = false;
    const message = messages[assistantIndex];
    const visibleAssistantIndex = mapRawMessageIndexToVisibleIndex(messages, assistantIndex);
    const article = els.answerResult?.querySelector(
      `[data-message-index="${visibleAssistantIndex}"][data-message-role="assistant"]`
    );
    if (!message || !article) {
      pending.answer = false;
      pending.reasoning = false;
      return;
    }

    const shouldStickToBottom = isOutputScrolledNearBottom();
    if (pending.reasoning) {
      updateThinkingTraceElement(article, message, { onlyWhenOpen: true });
    }
    if (pending.answer) {
      updateChatContentElement(article, message);
    }
    pending.answer = false;
    pending.reasoning = false;

    if (shouldStickToBottom) {
      els.outputCard.scrollTop = els.outputCard.scrollHeight;
    }
  }

  return {
    queue(kind) {
      if (kind === "reasoning") {
        pending.reasoning = true;
      } else {
        pending.answer = true;
      }
      if (frameQueued) return;
      frameQueued = true;
      const scheduleFrame = window.requestAnimationFrame
        ? (callback) => window.requestAnimationFrame(callback)
        : (callback) => window.setTimeout(callback, 16);
      scheduleFrame(apply);
    },
    flush() {
      apply();
    },
  };
}

function updateChatContentElement(article, message) {
  const contentEl = article.querySelector("[data-chat-content]");
  if (!contentEl) return;
  const content = String(message.content || "");
  contentEl.hidden = !content;
  contentEl.innerHTML = content ? renderMarkdown(content) : "";
  queueMathRender(contentEl);
}

function updateThinkingTraceElement(article, message, options = {}) {
  const traceEl = article.querySelector("[data-thinking-trace]");
  const contentEl = article.querySelector("[data-thinking-content]");
  const panelEl = article.querySelector("[data-thinking-panel]");
  if (!traceEl || !contentEl) return;
  if (options.onlyWhenOpen && panelEl?.hidden) return;
  traceEl.classList.toggle("is-live", Boolean(message.thinking_in_progress));
  contentEl.innerHTML = renderThinkingTraceContent(String(message.thinking_trace || "").trim());
  queueMathRender(contentEl);
}

function setThinkingTraceOpen(button, message) {
  const article = button.closest("[data-message-role='assistant']");
  const traceEl = article?.querySelector("[data-thinking-trace]");
  const panelEl = article?.querySelector("[data-thinking-panel]");
  if (!traceEl || !panelEl) return;
  const isOpen = Boolean(message.thinking_open);
  traceEl.classList.toggle("is-open", isOpen);
  panelEl.hidden = !isOpen;
  button.setAttribute("aria-expanded", isOpen ? "true" : "false");
  if (isOpen) {
    updateThinkingTraceElement(article, message);
  }
}

function renderThinkingTraceContent(thinkingTrace) {
  const trace = String(thinkingTrace || "").trim();
  return trace
    ? renderMarkdown(trace)
    : `<p class="thinking-placeholder">Model is preparing the response.</p>`;
}

function isOutputScrolledNearBottom() {
  if (!els.outputCard) return true;
  const distance = els.outputCard.scrollHeight - els.outputCard.scrollTop - els.outputCard.clientHeight;
  return distance < 80;
}

function setThinkingMode(mode) {
  state.thinkingMode = mode === "thinking" ? "thinking" : "fast";
  if (els.thinkingModeLabel) {
    els.thinkingModeLabel.textContent = state.thinkingMode === "thinking" ? "Thinking" : "Fast";
  }
  els.thinkingModeMenu?.querySelectorAll("[data-thinking-mode]").forEach((button) => {
    const isActive = button.dataset.thinkingMode === state.thinkingMode;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
  });
}

function toggleThinkingModeMenu() {
  const menu = els.thinkingModeMenu;
  const button = els.thinkingModeButton;
  if (!menu || !button) return;
  const willOpen = menu.hidden;
  menu.hidden = !willOpen;
  button.setAttribute("aria-expanded", willOpen ? "true" : "false");
}

function closeThinkingModeMenu() {
  const menu = els.thinkingModeMenu;
  const button = els.thinkingModeButton;
  if (!menu || !button) return;
  menu.hidden = true;
  button.setAttribute("aria-expanded", "false");
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
  if (els.thinkingModeButton) {
    els.thinkingModeButton.disabled = isBusy;
  }
  els.thinkingModeMenu?.querySelectorAll("button").forEach((button) => {
    button.disabled = isBusy;
  });
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

async function streamAskResponse({ sessionId, payload, onToken, onDone }) {
  const response = await fetch(`/sessions/${sessionId}/ask/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await readErrorResponse(response));
  }
  if (!response.body) {
    throw new Error("Streaming is not supported in this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let rawBuffer = "";
  let hasDoneEvent = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    rawBuffer += decoder.decode(value, { stream: true });
    const frames = splitSseFrames(rawBuffer);
    rawBuffer = frames.remainder;

    for (const frame of frames.frames) {
      const parsed = parseSseFrame(frame);
      if (!parsed) continue;
      const { event, data } = parsed;
      if (event === "token") {
        const delta = String(data?.delta || "");
        if (delta && typeof onToken === "function") {
          onToken(delta, String(data?.kind || ""));
        }
        continue;
      }
      if (event === "done") {
        hasDoneEvent = true;
        if (typeof onDone === "function") {
          onDone(data);
        }
        continue;
      }
      if (event === "error") {
        throw new Error(String(data?.detail || "Streaming request failed."));
      }
    }
  }

  if (!hasDoneEvent) {
    throw new Error("Stream ended before completion.");
  }
}

function splitSseFrames(buffer) {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const frames = [];
  let start = 0;
  while (true) {
    const idx = normalized.indexOf("\n\n", start);
    if (idx === -1) break;
    frames.push(normalized.slice(start, idx));
    start = idx + 2;
  }
  return {
    frames,
    remainder: normalized.slice(start),
  };
}

function parseSseFrame(frame) {
  const trimmed = String(frame || "").trim();
  if (!trimmed) return null;
  const lines = trimmed.split("\n");
  let event = "message";
  const dataLines = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice(6).trim() || "message";
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (!dataLines.length) return null;
  try {
    return {
      event,
      data: JSON.parse(dataLines.join("\n")),
    };
  } catch (_error) {
    return null;
  }
}

async function readJsonResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }
  return data;
}

async function readErrorResponse(response) {
  const text = await response.text().catch(() => "");
  if (!text) return "Request failed.";
  try {
    const parsed = JSON.parse(text);
    return parsed?.detail || text || "Request failed.";
  } catch (_error) {
    return text;
  }
}

function toFiniteNumberOrUndefined(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
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
  let mathBlockLines = null;

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

  const flushMathBlock = () => {
    if (mathBlockLines === null) return;
    const latex = mathBlockLines.join("\n").trim();
    if (latex) {
      html.push(`<div class="math-block" data-math-block>\\[${escapeHtml(latex)}\\]</div>`);
    }
    mathBlockLines = null;
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
    if (mathBlockLines !== null) {
      if (line === "$$") {
        flushMathBlock();
      } else {
        mathBlockLines.push(rawLine);
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
    if (line === "$$") {
      flushList();
      mathBlockLines = [];
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
  flushMathBlock();
  flushFence();
  flushList();
  return html.join("") || "<p></p>";
}

function queueMathRender(root) {
  const target = root instanceof Element ? root : document.body;
  if (!target) return;
  if (mathRenderFrame && window.cancelAnimationFrame) {
    window.cancelAnimationFrame(mathRenderFrame);
    mathRenderFrame = 0;
  }
  const run = () => {
    mathRenderFrame = 0;
    renderMath(target);
  };
  mathRenderFrame = window.requestAnimationFrame
    ? window.requestAnimationFrame(run)
    : window.setTimeout(run, 16);
}

function renderMath(root) {
  const target = root instanceof Element ? root : document.body;
  if (!target || !window.MathJax?.typesetPromise) return;
  const startup = window.MathJax.startup?.promise || Promise.resolve();
  startup
    .then(() => {
      if (typeof window.MathJax.typesetClear === "function") {
        window.MathJax.typesetClear([target]);
      }
      return window.MathJax.typesetPromise([target]);
    })
    .catch((_error) => {});
}

function updateOutputPlaceholderVisibility() {
  const hasMarkdown = isOcrVisible() && Boolean(els.ocrResult?.innerHTML.trim());
  const hasMessages = !els.answerResult.hidden && Boolean(els.answerResult.innerHTML.trim());
  els.emptyOutput.hidden = hasMarkdown || hasMessages;
}

function setQuickActionActive(mode) {
  document.querySelectorAll(".quick-actions [data-mode]").forEach((button) => {
    const buttonMode = button.dataset.mode || "";
    button.classList.toggle("is-active", Boolean(mode) && buttonMode === normalizeQuickActionMode(mode));
  });
  if (els.translateQuickAction) {
    els.translateQuickAction.classList.toggle("is-active", String(mode || "").startsWith("translate:"));
  }
}

function quickActionPrompt(mode) {
  if (String(mode || "").startsWith("translate:")) return `Translate to ${state.translateLanguage}`;
  return "Answer the question(s)";
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
    .replace(/^Answer the question\(s\) from the OCR text:\s*/i, "")
    .replace(/^Translate to [^:]+ using the OCR text:\s*/i, "")
    .replace(/^Answer this question:\s*/i, "")
    .replace(/^Answer the question\(s\):\s*/i, "")
    .replace(/^Translate to [^:]+:\s*/i, "");
}

function toggleTranslateLanguageMenu() {
  const menu = els.translateLanguageMenu;
  const button = els.translateLanguageButton;
  if (!menu || !button) return;
  const willOpen = menu.hidden;
  menu.hidden = !willOpen;
  button.setAttribute("aria-expanded", willOpen ? "true" : "false");
}

function closeTranslateLanguageMenu() {
  const menu = els.translateLanguageMenu;
  const button = els.translateLanguageButton;
  if (!menu || !button) return;
  menu.hidden = true;
  button.setAttribute("aria-expanded", "false");
}

function setTranslateLanguage(language) {
  state.translateLanguage = String(language || "Vietnamese").trim() || "Vietnamese";
  if (els.translateLanguageLabel) {
    els.translateLanguageLabel.textContent = state.translateLanguage;
  }
  if (String(state.activeMode || "").startsWith("translate:")) {
    state.activeMode = normalizeQuickActionMode("translate");
    setQuickActionActive(state.activeMode);
  }
  els.translateLanguageMenu?.querySelectorAll("[data-translate-language]").forEach((button) => {
    const isActive = button.dataset.translateLanguage === state.translateLanguage;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
  });
}

function normalizeQuickActionMode(mode) {
  if (mode === "translate" || String(mode || "").startsWith("translate:")) {
    return `translate:${state.translateLanguage}`;
  }
  return String(mode || "");
}

function canAttachDocument() {
  return !state.pendingFile && !state.hasDocument && !state.isBusy;
}

async function copyCurrentOcr() {
  if (!state.currentMarkdown) return;
  const textToCopy = cleanedOcrMarkdownForCopy(state.currentMarkdown);
  try {
    await copyText(textToCopy);
    setCopyFeedback("Copied");
  } catch (error) {
    setCopyFeedback("Copy failed");
  }
}

async function copyText(value) {
  const textToCopy = String(value || "");
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(textToCopy);
      return;
    } catch (_error) {
      fallbackCopyText(textToCopy);
      return;
    }
  }
  fallbackCopyText(textToCopy);
}

function cleanedOcrMarkdownForCopy(value) {
  return String(value || "")
    .replace(/^<!--\s*source:\s*.*?-->\s*\n*/i, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function fallbackCopyText(value) {
  const field = document.createElement("textarea");
  field.value = value;
  field.setAttribute("readonly", "readonly");
  field.style.position = "fixed";
  field.style.opacity = "0";
  field.style.left = "-9999px";
  field.style.top = "0";
  document.body.appendChild(field);
  field.select();
  field.setSelectionRange(0, field.value.length);
  const copied = document.execCommand("copy");
  document.body.removeChild(field);
  if (!copied) {
    throw new Error("Copy command failed");
  }
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

function setCopyButtonFeedback(button, message) {
  if (!button) return;
  if (button._feedbackTimer) {
    window.clearTimeout(button._feedbackTimer);
    button._feedbackTimer = null;
  }
  button.disabled = true;
  button.textContent = message;
  button._feedbackTimer = window.setTimeout(() => {
    button.textContent = "Copy";
    button.disabled = false;
    button._feedbackTimer = null;
  }, 1200);
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
  }
  els.ocrResult.hidden = false;
  queueMathRender(els.ocrResult);
}

function decorateSessionWithThinkingTrace(session, thinkingTrace) {
  const trace = String(thinkingTrace || "").trim();
  if (!trace || !session || !Array.isArray(session.messages)) {
    return;
  }
  for (let idx = session.messages.length - 1; idx >= 0; idx -= 1) {
    const message = session.messages[idx];
    if (message?.role !== "assistant") continue;
    message.thinking_trace = trace;
    message.thinking_in_progress = false;
    message.thinking_open = false;
    break;
  }
}

function hideOcrContent() {
  if (!els.ocrResult) return;
  els.ocrResult.innerHTML = "";
  if (els.ocrPanel) {
    els.ocrPanel.hidden = false;
  }
  els.ocrResult.hidden = true;
}

function isOcrVisible() {
  return Boolean(els.ocrResult) && !els.ocrResult.hidden;
}

function buildVisibleMessages(messages) {
  const rawMessages = Array.isArray(messages) ? messages : [];
  if (!shouldPrependOcrMessage(rawMessages)) {
    return rawMessages;
  }
  return [
    {
      role: "user",
      content: cleanedOcrMarkdownForCopy(state.currentMarkdown),
      is_ocr_result: true,
      synthetic_ocr_message: true,
    },
    ...rawMessages,
  ];
}

function shouldPrependOcrMessage(messages) {
  const ocrText = cleanedOcrMarkdownForCopy(state.currentMarkdown);
  if (!ocrText) return false;
  const firstMessage = Array.isArray(messages) && messages.length ? messages[0] : null;
  if (!firstMessage || firstMessage.role !== "user") return true;
  return cleanedOcrMarkdownForCopy(firstMessage.content) !== ocrText;
}

function mapRawMessageIndexToVisibleIndex(rawMessages, rawIndex) {
  if (!Number.isInteger(rawIndex) || rawIndex < 0) return rawIndex;
  return shouldPrependOcrMessage(rawMessages) ? rawIndex + 1 : rawIndex;
}

function isAssistantMessageComplete(message, index, visibleMessages) {
  if (!message || message.role !== "assistant" || message.error) {
    return false;
  }
  if (message.answer_complete === true) return true;
  if (message.answer_complete === false) return false;
  if (message.thinking_in_progress) return false;
  return true;
}

function ensureDynamicChatStyles() {
  if (document.getElementById("dynamic-chat-overrides")) {
    return;
  }
  const style = document.createElement("style");
  style.id = "dynamic-chat-overrides";
  style.textContent = `
    .ocr-panel {
      position: sticky;
      top: 0;
      z-index: 5;
      border-bottom: 1px solid var(--line);
      background: rgba(247, 250, 255, 0.96);
    }
    .ocr-toolbar {
      position: static !important;
      top: auto !important;
    }
    body.dark .ocr-panel {
      background: rgba(22, 32, 54, 0.96);
    }
    .chat-message.user .chat-bubble {
      color: #1b2747 !important;
      border: 1px solid #d8e1ec !important;
      background: #ffffff !important;
    }
    .chat-message.user .chat-bubble code {
      color: #1b2747 !important;
      background: #eef2ff !important;
    }
    .chat-message.assistant .chat-bubble {
      color: #1b2747 !important;
      border: 1px solid #d8e1ec !important;
      background: #f2f4f7 !important;
    }
    .chat-bubble-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      min-height: 20px;
      margin-top: 8px;
    }
    .chat-bubble-footer .chat-bubble-meta {
      margin-top: 0;
    }
    .copy-answer-button {
      min-height: 20px;
      padding: 0 6px;
      border: 1px solid var(--line-strong);
      border-radius: 5px;
      color: #2f426f;
      background: linear-gradient(180deg, #ffffff, #f5f8ff);
      font-size: 11px;
      font-weight: 700;
      line-height: 1;
    }
  `;
  document.head.appendChild(style);
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
