import os, json, time, secrets, asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Set

import httpx
import jwt
from jwt import PyJWKClient

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastmcp import FastMCP
from fastmcp.server.auth import AuthProvider, AccessToken

STATE_PATH = os.environ.get("STATE_PATH", "./broker_state.json")
ENROLLMENT_KEY = os.environ.get("AGENTBRIDGE_ENROLLMENT_KEY", "")

OIDC_ISSUER = os.environ.get("OIDC_ISSUER", "").rstrip("/")
OIDC_AUDIENCE = os.environ.get("OIDC_AUDIENCE", "")
OIDC_REQUIRED_SCOPE = os.environ.get("OIDC_REQUIRED_SCOPE", "agentbridge")
ENFORCE_SCOPES = os.environ.get("ENFORCE_SCOPES", "1") != "0"

def _now() -> int:
    return int(time.time())

def _load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"devices": {}, "pending_pairings": {}, "defaults": {"device_id": None}}

def _save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_PATH)

STATE = _load_state()

async def _discover_oidc() -> Dict[str, Any]:
    if not OIDC_ISSUER:
        raise RuntimeError("OIDC_ISSUER not set")
    url = f"{OIDC_ISSUER}/.well-known/openid-configuration"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()

class OIDCAuth(AuthProvider):
    def __init__(self):
        self._jwks_client: Optional[PyJWKClient] = None
        self._issuer: Optional[str] = None
        self._audience: str = OIDC_AUDIENCE
        self._required_scope: str = OIDC_REQUIRED_SCOPE

    async def _init(self):
        if self._jwks_client:
            return
        doc = await _discover_oidc()
        jwks_uri = doc.get("jwks_uri")
        self._issuer = doc.get("issuer")
        if not jwks_uri:
            raise RuntimeError("OIDC discovery missing jwks_uri")
        self._jwks_client = PyJWKClient(jwks_uri)

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        await self._init()
        assert self._jwks_client is not None
        assert self._issuer is not None

        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token).key
            decoded = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256", "ES256", "PS256"],
                audience=self._audience or None,
                issuer=self._issuer,
                options={"require": ["exp", "iat"]},
            )
        except Exception:
            return None

        scopes: Set[str] = set()
        if isinstance(decoded.get("scope"), str):
            scopes = set(decoded["scope"].split())
        elif isinstance(decoded.get("scp"), list):
            scopes = set(decoded["scp"])

        if ENFORCE_SCOPES and self._required_scope and self._required_scope not in scopes:
            return None

        sub = decoded.get("sub", "user")
        exp = int(decoded.get("exp", _now() + 300))
        return AccessToken(subject=sub, expires_at=exp, scopes=list(scopes), raw_claims=decoded)

auth = OIDCAuth()

@dataclass
class Conn:
    device_id: str
    ws: WebSocket
    connected_at: int

class DeviceManager:
    def __init__(self):
        self.conns: Dict[str, Conn] = {}
        self._lock = asyncio.Lock()
        self._pending: Dict[str, asyncio.Future] = {}

    async def register(self, device_id: str, ws: WebSocket):
        async with self._lock:
            self.conns[device_id] = Conn(device_id=device_id, ws=ws, connected_at=_now())

    async def unregister(self, device_id: str):
        async with self._lock:
            self.conns.pop(device_id, None)

    async def send_tool(self, device_id: str, tool: str, args: Dict[str, Any], timeout_sec: int = 60) -> Any:
        async with self._lock:
            conn = self.conns.get(device_id)
            if not conn:
                raise RuntimeError(f"Device not connected: {device_id}")
            req_id = secrets.token_hex(12)
            fut = asyncio.get_event_loop().create_future()
            self._pending[req_id] = fut
            await conn.ws.send_json({"type": "tool_req", "id": req_id, "tool": tool, "args": args})

        try:
            return await asyncio.wait_for(fut, timeout=timeout_sec)
        finally:
            async with self._lock:
                self._pending.pop(req_id, None)

    async def handle_message(self, msg: Dict[str, Any]):
        if msg.get("type") != "tool_res":
            return
        req_id = msg.get("id")
        async with self._lock:
            fut = self._pending.get(req_id)
        if not fut:
            return
        if msg.get("ok"):
            fut.set_result(msg.get("result"))
        else:
            fut.set_exception(RuntimeError(msg.get("error", "Unknown device error")))

DEVICES = DeviceManager()

api = FastAPI(title="AgentBridge Level 3 Broker")
mcp = FastMCP("AgentBridge Broker MCP (OAuth)", auth=auth)

def _get_default_device() -> str:
    did = STATE.get("defaults", {}).get("device_id")
    if not did:
        raise RuntimeError("No default device set. Pair a device, then device_set_default().")
    return did

@mcp.tool
def devices_list() -> List[Dict[str, Any]]:
    return [{
        "device_id": did,
        "label": info.get("label"),
        "last_seen": info.get("last_seen"),
        "connected": did in DEVICES.conns,
    } for did, info in STATE.get("devices", {}).items()]

@mcp.tool
def device_set_default(device_id: str) -> Dict[str, Any]:
    if device_id not in STATE.get("devices", {}):
        raise ValueError("Unknown device_id")
    STATE["defaults"]["device_id"] = device_id
    _save_state(STATE)
    return {"default_device_id": device_id}

@mcp.tool
def pairing_approve(code: str) -> Dict[str, Any]:
    pending = STATE.get("pending_pairings", {}).get(code)
    if not pending:
        raise ValueError("Invalid pairing code")
    if pending["expires_at"] < _now():
        STATE["pending_pairings"].pop(code, None)
        _save_state(STATE)
        raise ValueError("Pairing code expired")

    device_id = pending["device_id"]
    token = secrets.token_urlsafe(48)

    dev = STATE["devices"].get(device_id) or {"device_id": device_id}
    dev["token"] = token
    dev["last_seen"] = _now()
    dev.setdefault("label", f"My Mac ({device_id[:6]})")
    STATE["devices"][device_id] = dev
    STATE["pending_pairings"].pop(code, None)

    if not STATE.get("defaults", {}).get("device_id"):
        STATE["defaults"]["device_id"] = device_id

    _save_state(STATE)

    conn = DEVICES.conns.get(device_id)
    if conn:
        asyncio.create_task(conn.ws.send_json({"type": "pairing_ok", "device_token": token}))

    return {"paired_device_id": device_id, "default_device_id": STATE["defaults"]["device_id"]}

@mcp.tool
async def shell_exec(command: str, device_id: Optional[str] = None, cwd: Optional[str] = None, timeout_sec: int = 45) -> Dict[str, Any]:
    did = device_id or _get_default_device()
    return await DEVICES.send_tool(
        did,
        "shell_exec",
        {"command": command, "cwd": cwd, "timeout_sec": timeout_sec},
        timeout_sec=timeout_sec + 15,
    )

@mcp.tool
async def fs_list(path: str = ".", device_id: Optional[str] = None) -> Any:
    did = device_id or _get_default_device()
    return await DEVICES.send_tool(did, "fs_list", {"path": path})

@mcp.tool
async def fs_read(path: str, device_id: Optional[str] = None, max_bytes: int = 2_000_000) -> Any:
    did = device_id or _get_default_device()
    return await DEVICES.send_tool(did, "fs_read", {"path": path, "max_bytes": max_bytes})

@mcp.tool
async def fs_write(path: str, text: str, device_id: Optional[str] = None, create_dirs: bool = True) -> Any:
    did = device_id or _get_default_device()
    return await DEVICES.send_tool(did, "fs_write", {"path": path, "text": text, "create_dirs": create_dirs})

@mcp.tool
async def fs_mkdir(path: str, device_id: Optional[str] = None) -> Any:
    did = device_id or _get_default_device()
    return await DEVICES.send_tool(did, "fs_mkdir", {"path": path})

@mcp.tool
async def fs_delete(path: str, device_id: Optional[str] = None, recursive: bool = False) -> Any:
    did = device_id or _get_default_device()
    return await DEVICES.send_tool(did, "fs_delete", {"path": path, "recursive": recursive})

@mcp.tool
async def fs_pull(path: str, device_id: Optional[str] = None, offset: int = 0, chunk_bytes: int = 512_000) -> Any:
    did = device_id or _get_default_device()
    return await DEVICES.send_tool(did, "fs_pull", {"path": path, "offset": offset, "chunk_bytes": chunk_bytes})

@mcp.tool
async def fs_push(path: str, b64: str, device_id: Optional[str] = None, offset: int = 0, truncate: bool = False) -> Any:
    did = device_id or _get_default_device()
    return await DEVICES.send_tool(did, "fs_push", {"path": path, "b64": b64, "offset": offset, "truncate": truncate})

api.mount("/mcp", mcp.http_app(path="/"))

@api.get("/health")
def health():
    return {"status": "ok", "devices_connected": len(DEVICES.conns)}

@api.websocket("/agent/ws")
async def agent_ws(ws: WebSocket):
    await ws.accept()
    device_id = None
    try:
        hello = await ws.receive_json()
        if hello.get("type") != "hello":
            await ws.close(code=1008)
            return

        device_id = hello.get("device_id")
        mode = hello.get("mode")
        if not device_id:
            await ws.close(code=1008)
            return

        if mode == "enroll":
            if not ENROLLMENT_KEY or hello.get("enrollment_key") != ENROLLMENT_KEY:
                await ws.send_json({"type": "error", "error": "Bad enrollment key or enrollment disabled."})
                await ws.close(code=1008)
                return

            code = "PAIR-" + secrets.token_hex(3).upper()
            STATE.setdefault("pending_pairings", {})[code] = {"device_id": device_id, "expires_at": _now() + 600}
            STATE.setdefault("devices", {}).setdefault(device_id, {"device_id": device_id, "label": hello.get("label")})
            _save_state(STATE)

            await DEVICES.register(device_id, ws)
            await ws.send_json({"type": "pairing_code", "code": code, "expires_in_sec": 600})

        else:
            dev = STATE.get("devices", {}).get(device_id)
            token = hello.get("device_token")
            if not dev or not dev.get("token") or token != dev.get("token"):
                await ws.send_json({"type": "error", "error": "Unauthorized device (bad/missing token). Pair first."})
                await ws.close(code=1008)
                return

            dev["last_seen"] = _now()
            _save_state(STATE)

            await DEVICES.register(device_id, ws)
            await ws.send_json({"type": "ok", "message": "connected"})

        while True:
            msg = await ws.receive_json()
            await DEVICES.handle_message(msg)

    except WebSocketDisconnect:
        pass
    finally:
        if device_id:
            await DEVICES.unregister(device_id)
