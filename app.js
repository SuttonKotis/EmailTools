// DCX Intake Analyzer — browser-direct demo.
//
// Single static page. Paste an email, click Analyze, get a structured
// sufficiency verdict + draft response. One LLM call per email; no
// follow-up turns. To run again the user must paste a new email.
//
// SECURITY: The Anthropic API key is supplied at run time by the tester
// as a "bundle" — a JSON file dropped onto the page, or pasted into the
// fallback textarea. Bundle schema:
//
//   {
//     "anthropic_api_key": "sk-ant-...",   // required
//     "bundle_version":    1,               // optional; reserved for future
//     "additional_context": "...",          // optional; appended to system prompt
//     "model":             "claude-opus-4-7", // optional; overrides default
//     "effort":            "high"           // optional; one of low|medium|high|xhigh|max
//   }
//
// The bundle is held in module-scope memory only — no localStorage, no
// sessionStorage, no logging. Reloading the page or clicking "Clear key"
// wipes it. The key (and any private prompt context bundled with it) is
// still exposed to the browser process (extensions, devtools, screen
// captures) and travels to api.anthropic.com over the wire — treat the
// whole bundle like a password. Production should proxy through a server.

import Anthropic from "@anthropic-ai/sdk";
import { marked } from "marked";

// ---------- config ----------

const MODEL = "claude-opus-4-7";
const MAX_TOKENS = 4096;
const VALID_EFFORT = ["low", "medium", "high", "xhigh", "max"];
const ADDITIONAL_CONTEXT_MAX = 8000;

// ---------- prompt + schema ----------

const SYSTEM_PROMPT = `You are an intake-analysis assistant. You read one \
inbound message and produce a structured verdict according to \
operator-provided rules.

The schema's source.pattern enum is fixed at:
  attached_files | logo_library | prior_job | brand_guidelines | none
The operator spec defines what content qualifies for each bucket; you do \
not invent your own definitions.

WHEN AN OPERATOR SPEC IS PROVIDED (an "## Operator spec" section appears \
below this scaffold)
Follow it literally. The spec defines:
- What to extract for the WHAT field
- The rules for assigning each SOURCE pattern bucket
- Yellow-flag phrases to detect
- Sufficiency criteria (when a message is "ready" vs "not_ready")
- The format of the draft response (clarifying reply vs structured handoff)

If the spec is silent on something, return conservative defaults: low \
confidence, "not_ready", empty arrays.

WHEN NO OPERATOR SPEC IS PROVIDED
Do not attempt analysis. Return this exact placeholder verdict:
- what.extracted: null, what.confidence: "low"
- source.pattern: "none", source.evidence: null
- yellow_flags: []
- sufficiency.verdict: "not_ready", sufficiency.missing: []
- draft.kind: "clarifying_email"
- draft.body: "(no analysis specification was loaded — please configure the bundle with operator instructions)"
- rationale: "No operator spec was provided in the bundle. Cannot analyze without context defining what to extract and how to evaluate sufficiency."

Return only the JSON object.`;

const SCHEMA = {
    type: "object",
    properties: {
        what: {
            type: "object",
            properties: {
                extracted: { type: ["string", "null"] },
                confidence: { type: "string", enum: ["high", "medium", "low"] }
            },
            required: ["extracted", "confidence"],
            additionalProperties: false
        },
        source: {
            type: "object",
            properties: {
                pattern: {
                    type: "string",
                    enum: ["attached_files", "logo_library", "prior_job",
                           "brand_guidelines", "none"]
                },
                evidence: { type: ["string", "null"] }
            },
            required: ["pattern", "evidence"],
            additionalProperties: false
        },
        yellow_flags: {
            type: "array",
            items: {
                type: "object",
                properties: {
                    phrase: { type: "string" },
                    concern: { type: "string" }
                },
                required: ["phrase", "concern"],
                additionalProperties: false
            }
        },
        sufficiency: {
            type: "object",
            properties: {
                verdict: { type: "string", enum: ["ready", "not_ready"] },
                missing: {
                    type: "array",
                    items: { type: "string", enum: ["what", "source"] }
                }
            },
            required: ["verdict", "missing"],
            additionalProperties: false
        },
        draft: {
            type: "object",
            properties: {
                kind: { type: "string", enum: ["clarifying_email", "actionbox"] },
                body: { type: "string" }
            },
            required: ["kind", "body"],
            additionalProperties: false
        },
        rationale: { type: "string" }
    },
    required: ["what", "source", "yellow_flags", "sufficiency", "draft", "rationale"],
    additionalProperties: false
};

const SOURCE_LABEL = {
    attached_files:    "Attached art files",
    logo_library:      "Logo library reference",
    prior_job:         "Prior job reference",
    brand_guidelines:  "Brand guidelines / external link",
    none:              "No identifiable source"
};

// ---------- state ----------

const STATE = {
    mode: "idle",   // idle | loaded | analyzing | done | error
    email: null,
    verdict: null
};

// Memory-only credentials + bundle. Never persisted, never logged.
// Shape: { apiKey, additionalContext?, model?, effort?, version? }
let _bundle = null;
let _client = null;

const $ = (id) => document.getElementById(id);
const els = {
    keyCard:         $("key-card"),
    keyLoadArea:     $("key-load-area"),
    keyDropzone:     $("key-dropzone"),
    keyPasteInput:   $("key-paste-input"),
    btnUseKey:       $("btn-use-key"),
    keyStatus:       $("key-status"),
    keyFingerprint:  $("key-fingerprint"),
    extraCtx:        $("extra-ctx-indicator"),
    btnClearKey:     $("btn-clear-key"),

    setupBar:        $("setup-bar"),
    btnPaste:        $("btn-paste"),
    btnClear:        $("btn-clear"),
    btnAnalyze:      $("btn-analyze"),
    btnCopyDraft:    $("btn-copy-draft"),
    statePill:       $("state-pill"),
    statusLine:      $("status-line"),

    emailCard:       $("email-card"),
    emailBody:       $("email-body"),
    emailMeta:       $("email-meta"),

    verdictCard:     $("verdict-card"),
    verdictBadge:    $("verdict-badge"),
    verdictMeta:     $("verdict-meta"),
    verdictWhat:     $("verdict-what"),
    verdictWhatConf: $("verdict-what-conf"),
    verdictSource:   $("verdict-source"),
    verdictSourceEv: $("verdict-source-evidence"),
    verdictDraft:    $("verdict-draft"),
    verdictRationale:$("verdict-rationale"),
    yellowFlagsBlock:$("yellow-flags-block"),
    yellowFlags:     $("yellow-flags"),
    draftKindLabel:  $("draft-kind-label"),

    errorCard:       $("error-card"),
    errorMessage:    $("error-message")
};

// ---------- bundle handling ----------

function fingerprint(key) {
    if (!key) return "";
    if (key.length <= 11) return key;
    return `${key.slice(0, 7)}…${key.slice(-4)}`;  // sk-ant-…4f2a
}

// Parse a bundle from text. Returns {bundle, error}.
// allowBareKey=true accepts a non-JSON string as a bare API key (paste fallback).
function parseBundle(text, { allowBareKey = false } = {}) {
    const raw = (text || "").trim();
    if (!raw) return { bundle: null, error: "Empty input." };

    let obj;
    try {
        obj = JSON.parse(raw);
    } catch (_) {
        if (allowBareKey) {
            return validateBundle({ anthropic_api_key: raw });
        }
        return { bundle: null, error: "File is not valid JSON." };
    }
    if (typeof obj !== "object" || obj === null) {
        return { bundle: null, error: "Bundle must be a JSON object." };
    }
    return validateBundle(obj);
}

function validateBundle(obj) {
    const apiKey = obj.anthropic_api_key;
    if (typeof apiKey !== "string" || !apiKey.trim()) {
        return { bundle: null, error: 'Missing required "anthropic_api_key" string field.' };
    }
    const trimmedKey = apiKey.trim();
    if (!trimmedKey.startsWith("sk-ant-")) {
        return { bundle: null, error: 'Key does not look like an Anthropic key (expected to start with "sk-ant-").' };
    }

    const bundle = { apiKey: trimmedKey };

    if (obj.additional_context !== undefined) {
        if (typeof obj.additional_context !== "string") {
            return { bundle: null, error: '"additional_context" must be a string.' };
        }
        if (obj.additional_context.length > ADDITIONAL_CONTEXT_MAX) {
            return { bundle: null, error: `"additional_context" too long (${obj.additional_context.length} chars; max ${ADDITIONAL_CONTEXT_MAX}).` };
        }
        if (obj.additional_context.trim()) {
            bundle.additionalContext = obj.additional_context;
        }
    }

    if (obj.model !== undefined) {
        if (typeof obj.model !== "string" || !obj.model.trim()) {
            return { bundle: null, error: '"model" must be a non-empty string.' };
        }
        bundle.model = obj.model.trim();
    }

    if (obj.effort !== undefined) {
        if (!VALID_EFFORT.includes(obj.effort)) {
            return { bundle: null, error: `"effort" must be one of: ${VALID_EFFORT.join(", ")}.` };
        }
        bundle.effort = obj.effort;
    }

    if (obj.bundle_version !== undefined) {
        bundle.version = obj.bundle_version;
    }

    return { bundle, error: null };
}

function setBundle(bundle) {
    _bundle = bundle;
    _client = null;  // force rebuild on next call
}

function clearBundle() {
    _bundle = null;
    _client = null;
    STATE.email = null;
    STATE.verdict = null;
    els.emailBody.innerHTML = "";
    els.emailCard.classList.add("hidden");
    els.verdictCard.classList.add("hidden");
    clearError();

    els.keyLoadArea.classList.remove("hidden");
    els.keyStatus.classList.add("hidden");
    els.extraCtx.classList.add("hidden");
    els.keyPasteInput.value = "";
    els.setupBar.classList.add("hidden");
    setMode("idle");
}

function onBundleLoaded() {
    els.keyLoadArea.classList.add("hidden");
    els.keyStatus.classList.remove("hidden");
    els.keyFingerprint.textContent = fingerprint(_bundle.apiKey);

    const overrides = [];
    if (_bundle.additionalContext) {
        overrides.push(`extra context (${_bundle.additionalContext.length} chars)`);
    }
    if (_bundle.model) overrides.push(`model: ${_bundle.model}`);
    if (_bundle.effort) overrides.push(`effort: ${_bundle.effort}`);

    if (overrides.length > 0) {
        els.extraCtx.classList.remove("hidden");
        els.extraCtx.title = overrides.join(" · ");
    } else {
        els.extraCtx.classList.add("hidden");
        els.extraCtx.title = "";
    }

    els.setupBar.classList.remove("hidden");
    clearError();
    setMode("idle");
}

async function handleDroppedFile(file) {
    if (!file) return;
    let text;
    try {
        text = await file.text();
    } catch (e) {
        showError(`Could not read file: ${e.message}`);
        return;
    }
    const { bundle, error } = parseBundle(text, { allowBareKey: false });
    if (error) {
        showError(error);
        return;
    }
    setBundle(bundle);
    onBundleLoaded();
}

function handlePastedBundle() {
    const { bundle, error } = parseBundle(els.keyPasteInput.value, { allowBareKey: true });
    if (error) {
        showError(error);
        return;
    }
    setBundle(bundle);
    els.keyPasteInput.value = "";
    onBundleLoaded();
}

function getClient() {
    if (_client) return _client;
    if (!_bundle) throw new Error("No key loaded. Drop a key.json onto the page.");
    _client = new Anthropic({
        apiKey: _bundle.apiKey,
        dangerouslyAllowBrowser: true
    });
    return _client;
}

// ---------- state machine ----------

function setMode(mode) {
    STATE.mode = mode;
    const pillText = {
        idle:      "no email",
        loaded:    "ready",
        analyzing: "analyzing…",
        done:      "done",
        error:     "error"
    }[mode];
    els.statePill.textContent = pillText;
    els.statePill.className = `state-pill state-${mode}`;

    const haveKey = _bundle != null;
    els.btnPaste.disabled    = !haveKey || mode === "analyzing";
    els.btnClear.disabled    = !haveKey || mode === "idle" || mode === "analyzing";
    els.btnAnalyze.disabled  = !haveKey || mode !== "loaded";
}

function setStatus(text) {
    els.statusLine.textContent = text || "";
}

function showError(message) {
    els.errorMessage.textContent = message;
    els.errorCard.classList.remove("hidden");
}

function clearError() {
    els.errorCard.classList.add("hidden");
    els.errorMessage.textContent = "";
}

// ---------- email intake ----------

async function pasteFromClipboard() {
    clearError();
    if (!navigator.clipboard?.readText) {
        showError("Clipboard API not available. Use a recent Chrome / Edge / Safari over HTTPS or localhost.");
        return;
    }
    let text;
    try {
        text = await navigator.clipboard.readText();
    } catch (e) {
        showError(`Clipboard read failed: ${e.message}. Browser likely blocked the read — try clicking the page first.`);
        return;
    }
    text = (text || "").trim();
    if (!text) {
        showError("Clipboard is empty.");
        return;
    }
    loadEmail(text);
}

function loadEmail(text) {
    STATE.email = text;
    STATE.verdict = null;
    els.emailBody.innerHTML = marked.parse(text, { breaks: true });
    els.emailCard.classList.remove("hidden");
    els.emailMeta.textContent = `${text.length.toLocaleString()} chars`;
    els.verdictCard.classList.add("hidden");
    setMode("loaded");
    setStatus("");
}

function clearEmail() {
    STATE.email = null;
    STATE.verdict = null;
    els.emailBody.innerHTML = "";
    els.emailCard.classList.add("hidden");
    els.verdictCard.classList.add("hidden");
    clearError();
    setMode("idle");
    setStatus("");
}

// ---------- analysis ----------

async function analyze() {
    if (STATE.mode !== "loaded" || !STATE.email) return;
    clearError();
    setMode("analyzing");
    setStatus("calling Claude…");

    const startedAt = performance.now();
    let client;
    try {
        client = getClient();
    } catch (e) {
        showError(e.message);
        setMode("loaded");
        return;
    }

    const model = _bundle.model || MODEL;
    const effort = _bundle.effort || "high";
    const systemText = _bundle.additionalContext
        ? `${SYSTEM_PROMPT}\n\n## Operator spec\n\n${_bundle.additionalContext}`
        : SYSTEM_PROMPT;

    let response;
    try {
        response = await client.messages.create({
            model,
            max_tokens: MAX_TOKENS,
            thinking: { type: "adaptive" },
            output_config: {
                effort,
                format: { type: "json_schema", schema: SCHEMA }
            },
            system: systemText,
            messages: [{
                role: "user",
                content: `## Inbound client email\n\n${STATE.email}\n\nAnalyze the email now.`
            }]
        });
    } catch (e) {
        const detail = e?.error?.error?.message || e?.message || String(e);
        showError(`API call failed: ${detail}`);
        setMode("loaded");
        return;
    }

    const textBlock = response.content.find((b) => b.type === "text");
    if (!textBlock) {
        showError("Model returned no text block.");
        setMode("loaded");
        return;
    }

    let verdict;
    try {
        verdict = JSON.parse(textBlock.text);
    } catch (e) {
        showError(`Could not parse model output as JSON: ${e.message}`);
        setMode("loaded");
        return;
    }

    STATE.verdict = verdict;
    renderVerdict(verdict, performance.now() - startedAt, response.usage);
    setMode("done");
    setStatus("");
}

// ---------- verdict rendering ----------

function renderVerdict(v, elapsedMs, usage) {
    const ready = v.sufficiency.verdict === "ready";

    els.verdictBadge.textContent = ready ? "READY" : "NOT READY";
    els.verdictBadge.className = `badge ${ready ? "pass" : "fail"}`;

    const elapsedSec = (elapsedMs / 1000).toFixed(1);
    const tokens = usage ? ` · ${usage.input_tokens} in / ${usage.output_tokens} out` : "";
    els.verdictMeta.textContent = `${elapsedSec}s${tokens}`;

    els.verdictWhat.textContent = v.what.extracted || "(not identified)";
    els.verdictWhatConf.textContent = v.what.extracted
        ? `confidence: ${v.what.confidence}`
        : "";

    els.verdictSource.textContent = SOURCE_LABEL[v.source.pattern] || v.source.pattern;
    els.verdictSourceEv.textContent = v.source.evidence
        ? `evidence: ${v.source.evidence}`
        : "";

    if (v.yellow_flags && v.yellow_flags.length > 0) {
        els.yellowFlags.innerHTML = "";
        for (const flag of v.yellow_flags) {
            const li = document.createElement("li");
            const phraseEl = document.createElement("strong");
            phraseEl.textContent = `"${flag.phrase}"`;
            li.appendChild(phraseEl);
            li.appendChild(document.createTextNode(` — ${flag.concern}`));
            els.yellowFlags.appendChild(li);
        }
        els.yellowFlagsBlock.classList.remove("hidden");
    } else {
        els.yellowFlagsBlock.classList.add("hidden");
    }

    els.draftKindLabel.textContent = v.draft.kind === "clarifying_email"
        ? "Draft clarifying reply"
        : "Draft actionbox";
    els.verdictDraft.textContent = v.draft.body;

    els.verdictRationale.textContent = v.rationale;

    els.verdictCard.classList.remove("hidden");
}

async function copyDraft() {
    if (!STATE.verdict) return;
    try {
        await navigator.clipboard.writeText(STATE.verdict.draft.body);
        const original = els.btnCopyDraft.textContent;
        els.btnCopyDraft.textContent = "Copied";
        setTimeout(() => { els.btnCopyDraft.textContent = original; }, 1200);
    } catch (e) {
        showError(`Copy failed: ${e.message}`);
    }
}

// ---------- wiring ----------

// Drop zone — visual hit target with feedback. The whole document is a
// fallback drop target when no key is loaded, so a missed drop on the
// visible zone still gets received.
const dz = els.keyDropzone;

["dragenter", "dragover"].forEach((t) => {
    dz.addEventListener(t, (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (_bundle == null) dz.classList.add("dragover");
    });
});
["dragleave", "dragend"].forEach((t) => {
    dz.addEventListener(t, () => dz.classList.remove("dragover"));
});
dz.addEventListener("drop", async (e) => {
    e.preventDefault();
    e.stopPropagation();  // suppress the document-level fallback below
    dz.classList.remove("dragover");
    const file = e.dataTransfer?.files?.[0];
    if (file) await handleDroppedFile(file);
});

// Document-level fallback: drops anywhere on the page (when no key is
// loaded yet) are treated as a bundle drop. The inline <head> script
// already prevents the browser-default file-open; this adds reception.
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", async (e) => {
    e.preventDefault();
    if (_bundle != null) return;
    const file = e.dataTransfer?.files?.[0];
    if (file) await handleDroppedFile(file);
});

els.btnUseKey.addEventListener("click", handlePastedBundle);
els.keyPasteInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handlePastedBundle();
});
els.btnClearKey.addEventListener("click", clearBundle);

els.btnPaste.addEventListener("click", pasteFromClipboard);
els.btnClear.addEventListener("click", clearEmail);
els.btnAnalyze.addEventListener("click", analyze);
els.btnCopyDraft.addEventListener("click", copyDraft);

setMode("idle");
