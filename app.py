"""
FastAPI backend for the FinSolve internal assistant.

Wraps the exact same pipeline the CLI uses (rag_chain + rbac + guardrails).
The only web-specific concern here is session handling: on login the server
issues an opaque token and remembers token -> username server-side, so the
role used for RBAC is ALWAYS decided by the server, never trusted from the
client. Sending role in the request body would make RBAC bypassable.
"""

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rbac import DEMO_USERS, get_allowed_departments
from rag_chain import retrieve_with_scores, generate_answer
from guardrails.scope import is_out_of_scope, OUT_OF_SCOPE_MESSAGE
from guardrails.pii import redact_pii

BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="FinSolve Internal Assistant")

# In-memory session store: token -> username. Cleared on server restart.
# A real deployment would use signed cookies / JWT / a session backend.
_sessions: dict[str, str] = {}


class LoginRequest(BaseModel):
    username: str


class ChatRequest(BaseModel):
    token: str
    question: str


def _resolve_role(token: str) -> tuple[str, str]:
    """Return (username, role) for a valid token, else 401."""
    username = _sessions.get(token)
    if username is None:
        raise HTTPException(status_code=401, detail="Not logged in or session expired.")
    return username, DEMO_USERS[username]["role"]


@app.get("/api/users")
def list_users():
    """Public: the login screen needs the list of demo users to choose from."""
    return [
        {"username": username, **{k: v for k, v in user.items()}}
        for username, user in DEMO_USERS.items()
    ]


@app.post("/api/login")
def login(req: LoginRequest):
    if req.username not in DEMO_USERS:
        raise HTTPException(status_code=404, detail="Unknown user.")
    token = uuid.uuid4().hex
    _sessions[token] = req.username

    user = DEMO_USERS[req.username]
    role = user["role"]
    return {
        "token": token,
        "user": {"username": req.username, **user},
        "allowed_departments": get_allowed_departments(role),
    }


@app.post("/api/logout")
def logout(req: ChatRequest):
    _sessions.pop(req.token, None)
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatRequest):
    _username, role = _resolve_role(req.token)

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")

    results = retrieve_with_scores(question, role)

    if is_out_of_scope(results):
        return {"answer": OUT_OF_SCOPE_MESSAGE, "sources": [], "out_of_scope": True}

    docs = [doc for doc, _ in results]
    try:
        answer = generate_answer(question, docs)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    answer = redact_pii(answer, role)
    sources = sorted({doc.metadata.get("source") for doc in docs})
    return {"answer": answer, "sources": sources, "out_of_scope": False}


# Serve the frontend. Mounted last so it doesn't shadow the /api routes.
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
