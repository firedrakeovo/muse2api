"""Offline checks for chat image input parsing and preparation."""
import base64
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HOME = tempfile.TemporaryDirectory(prefix="muse-chat-mm-")
os.environ.update(MUSE2API_HOME=HOME.name,
                  MUSE2API_HOME_DIR=HOME.name,
                  MUSE2API_PROFILE_ROOT=HOME.name,
                  MUSE2API_KEY="test-only")

import app  # noqa: E402

PNG_1PX = ("data:image/png;base64,"
           "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0l"
           "EQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=")


def check(cond, message):
    if not cond:
        raise AssertionError(message)


# Chat Completions image_url + Responses input_image are both recognized.
msg = app.ChatMessage(role="user", content=[
    {"type": "text", "text": "这张图是什么颜色？"},
    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    {"type": "input_image", "image_url": "https://example.com/b.png"},
])
refs = app._chat_image_sources([msg])
check(refs == ["https://example.com/a.png", "https://example.com/b.png"],
      f"unexpected refs: {refs}")

# Assistant images must not be treated as user attachments.
assistant = app.ChatMessage(role="assistant", content=[
    {"type": "image_url", "image_url": {"url": "https://example.com/reply.png"}},
])
check(app._chat_image_sources([assistant]) == [], "assistant image leaked into input")

# Prompt still carries a placeholder so the model gets textual context.
prompt = app.build_chat_prompt([msg])
check("[图片]" in prompt and "这张图是什么颜色？" in prompt, prompt)

# Malformed image references are rejected before any browser work.
bad = app.ChatMessage(role="user", content=[
    {"type": "image_url", "image_url": {"detail": "high"}},
])
try:
    app._chat_image_sources([bad])
except app.HTTPException as exc:
    check(exc.status_code == 400, exc.status_code)
else:
    raise AssertionError("malformed image reference was accepted")

# More than the reference limit is rejected.
many = app.ChatMessage(role="user", content=[
    {"type": "image_url", "image_url": f"https://example.com/{i}.png"}
    for i in range(app.MAX_REFERENCE_IMAGES + 1)
])
try:
    app._chat_image_sources([many])
except app.HTTPException as exc:
    check(exc.status_code == 400, exc.status_code)
else:
    raise AssertionError("too many chat images were accepted")

# Inline data URL is snapshotted to a canonical data URL.
prepared = app._prepare_chat_images([PNG_1PX])
check(len(prepared) == 1 and prepared[0].startswith("data:image/png;base64,"),
      prepared[:1])
check(base64.b64decode(prepared[0].split(",", 1)[1]), "empty image payload")

# Non-image payloads must not be uploaded through the chat image path.
try:
    app._prepare_chat_images(["data:text/plain;base64,aGVsbG8="])
except app.HTTPException as exc:
    check(exc.status_code == 400, exc.status_code)
else:
    raise AssertionError("text payload was accepted as a chat image")

# Responses top-level input_image is normalized into a user message.
req = app.ResponsesRequest(input=[{
    "type": "input_image", "image_url": "https://example.com/responses.png"}])
msgs = app._responses_messages(req)
check(app._chat_image_sources(msgs) == ["https://example.com/responses.png"],
      msgs)

print("chat multimodal checks passed")
