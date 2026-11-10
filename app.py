"""
Prompt Generating AI 2.0 — Flask backend
=========================================

A small, production-ready API that turns a system prompt + user task into a
generated prompt using the Google GenAI SDK (Gemini).

Run:
    pip install flask flask-cors python-dotenv google-genai
    python app.py

Environment (.env):
    GEMINI_API_KEY=your-server-side-key      # optional, used by genai.Client()
    PORT=5000                                # optional

Endpoints:
    GET  /health   -> liveness probe
    POST /generate -> { "systemPrompt": "...", "userPrompt": "..." }
                      optional header: X-Gemini-API-Key: <user's own key>
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("prompt-generating-ai")

# Model can be overridden with GEMINI_MODEL; gemini-3.8-flash is the default.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Header the frontend uses to pass a user-supplied API key.
API_KEY_HEADER = "X-Gemini-API-Key"

# Guardrails so a huge payload can't be used to burn tokens.
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "20000"))

app = Flask(__name__, static_folder=".", static_url_path="")

# Fully open CORS, but only the headers this API actually understands.
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", API_KEY_HEADER],
    methods=["GET", "POST", "OPTIONS"],
    expose_headers=["Content-Type"],
    supports_credentials=False,
    max_age=86400,
)

# ---------------------------------------------------------------------------
# GenAI client handling
# ---------------------------------------------------------------------------

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - import-time guard
    genai = None
    genai_types = None
    log.error(
        "google-genai is not installed (%s). Run: pip install google-genai", exc
    )


def build_client(custom_api_key: str | None):
    """Return a GenAI client.

    Uses the caller's key when one is supplied in the request header,
    otherwise falls back to the server-side default client.
    """
    if genai is None:
        raise RuntimeError(
            "google-genai SDK is not installed on the server. "
            "Install it with: pip install google-genai"
        )

    if custom_api_key:
        log.info("Using caller-supplied Gemini API key.")
        return genai.Client(api_key=custom_api_key)

    log.info("Using server default Gemini client.")
    return genai.Client()


def build_contents(system_prompt: str, user_prompt: str) -> str:
    """Combine the system instructions and the user's task into one request.

    The SDK's system_instruction field is preferred when available, but we also
    embed the instructions in the content so the model keeps the framing even
    on SDK versions that ignore it.
    """
    parts = []
    if system_prompt:
        parts.append(f"SYSTEM INSTRUCTIONS:\n{system_prompt.strip()}")
    parts.append(f"USER TASK:\n{user_prompt.strip()}")
    return "\n\n".join(parts)


def extract_text(response: Any) -> str:
    """Safely pull text out of a GenAI response.

    Tries `response.text` first, then walks candidates -> content -> parts.
    `response.text` raises when there are no parts, so it is wrapped.
    """
    # 1) The convenient accessor.
    try:
        text = response.text
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception as exc:  # noqa: BLE001 - SDK raises on empty candidates
        log.debug("response.text unavailable: %s", exc)

    # 2) Fallback: concatenate every text part across all candidates.
    chunks: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text:
                chunks.append(part_text)

    if chunks:
        return "\n".join(chunks).strip()

    # 3) Nothing usable — surface the block reason if there is one.
    for candidate in getattr(response, "candidates", None) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason:
            raise ValueError(f"Gemini returned no text (finish_reason={reason})")

    raise ValueError("Gemini returned an empty response.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    """Serve the single-page frontend that lives beside this file."""
    return send_from_directory(app.root_path, "index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": DEFAULT_MODEL})


@app.post("/generate")
def generate():
    payload = request.get_json(silent=True) or {}

    system_prompt = (payload.get("systemPrompt") or "").strip()
    user_prompt = (payload.get("userPrompt") or "").strip()

    if not user_prompt:
        return jsonify({"error": "`userPrompt` is required."}), 400
    if not system_prompt:
        return jsonify({"error": "`systemPrompt` is required."}), 400
    if len(system_prompt) + len(user_prompt) > MAX_PROMPT_CHARS:
        return (
            jsonify(
                {"error": f"Prompt too long. Limit is {MAX_PROMPT_CHARS} characters."}
            ),
            413,
        )

    custom_api_key = (request.headers.get(API_KEY_HEADER) or "").strip() or None

    try:
        client = build_client(custom_api_key)

        contents = build_contents(system_prompt, user_prompt)

        config = None
        if genai_types is not None:
            try:
                config = genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.9")),
                    max_output_tokens=int(
                        os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - config is optional
                log.warning("Could not build GenerateContentConfig: %s", exc)
                config = None

        if config is not None:
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=contents,
                config=config,
            )
        else:
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=contents,
            )

        text = extract_text(response)
        return jsonify({"text": text, "model": DEFAULT_MODEL}), 200

    except Exception as exc:  # noqa: BLE001 - never leak a traceback to clients
        log.exception("Generation failed: %s", exc)
        return (
            jsonify(
                {
                    "error": "Generation failed. Please try again.",
                    "detail": str(exc),
                }
            ),
            500,
        )


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'online'}), 200