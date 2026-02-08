# main.py
# FRANZ - Narrative Desktop Agent (Windows 11 + Python 3.12)
# Standard library only, stateless API, story-driven memory

from __future__ import annotations
import argparse
import base64
import ctypes
import ctypes.wintypes as w
import json
import struct
import threading
import time
import urllib.request
import zlib
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen3-vl-2b-instruct-1m"

RES_W, RES_H = 1536, 864
STORY_MAX = 1400
MIN_STORY_LEN = 200

SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.85,
    "top_k": 30,
    "max_tokens": 600,
    "presence_penalty": 0.8,
    "frequency_penalty": 0.3,
    "repeat_penalty": 1.15,
}

INIT_STORY = (
    "[IDENTITY] FRANZ desktop agent v1.0\n"
    "[PAST] Initialized in sandbox environment.\n"
    "[NOW] Awaiting task instructions.\n"
    "[WHERE] Windows desktop, no active windows.\n"
    "[DELTA] System idle.\n"
    "[NEXT] Wait for explicit task.\n"
    "[CONF] 0.00"
)

SYSTEM_PROMPT = (
    "You are FRANZ, a Windows desktop controller. Use only provided tools. "
    "Every action sequence must end with commit_story(story). "
    "Rewrite the full story each time, preserving [IDENTITY] and structure. "
    "If uncertain about task, use wait(3000) then commit_story."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Left click at x,y (normalized 0-1000)",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "y": {"type": "integer", "minimum": 0, "maximum": 1000}
                },
                "required": ["x", "y"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "right_click",
            "description": "Right click at x,y",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "y": {"type": "integer", "minimum": 0, "maximum": 1000}
                },
                "required": ["x", "y"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": "Double click at x,y",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "y": {"type": "integer", "minimum": 0, "maximum": 1000}
                },
                "required": ["x", "y"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text string",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll wheel, dy in [-100,100]",
            "parameters": {
                "type": "object",
                "properties": {"dy": {"type": "integer", "minimum": -100, "maximum": 100}},
                "required": ["dy"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait milliseconds (1-10000), use when no confident task",
            "parameters": {
                "type": "object",
                "properties": {"ms": {"type": "integer", "minimum": 1, "maximum": 10000}},
                "required": ["ms"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "commit_story",
            "description": "REQUIRED final call. Rewrite full story with all tags preserved.",
            "parameters": {
                "type": "object",
                "properties": {"story": {"type": "string"}},
                "required": ["story"],
                "additionalProperties": False
            }
        }
    }
]

try:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
except OSError as e:
    raise RuntimeError(f"Failed to load Windows DLLs: {e}")

try:
    ctypes.WinDLL("Shcore").SetProcessDpiAwareness(2)
except:
    pass

# Define ctypes structures
class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", w.DWORD), ("biWidth", w.LONG), ("biHeight", w.LONG),
        ("biPlanes", w.WORD), ("biBitCount", w.WORD), ("biCompression", w.DWORD),
        ("biSizeImage", w.DWORD), ("biXPelsPerMeter", w.LONG),
        ("biYPelsPerMeter", w.LONG), ("biClrUsed", w.DWORD), ("biClrImportant", w.DWORD)
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", w.DWORD * 3)]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", w.LONG), ("dy", w.LONG), ("mouseData", w.DWORD),
        ("dwFlags", w.DWORD), ("time", w.DWORD), ("dwExtraInfo", ctypes.c_size_t)
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", w.WORD), ("wScan", w.WORD), ("dwFlags", w.DWORD),
        ("time", w.DWORD), ("dwExtraInfo", ctypes.c_size_t)
    ]

class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", w.DWORD), ("union", _INPUTunion)]

# Set argtypes and restype for GDI/user32 functions to prevent overflows
gdi32.CreateCompatibleDC.argtypes = [w.HDC]
gdi32.CreateCompatibleDC.restype = w.HDC

gdi32.CreateDIBSection.argtypes = [
    w.HDC, ctypes.POINTER(BITMAPINFO), ctypes.c_uint,
    ctypes.POINTER(ctypes.c_void_p), w.HANDLE, w.DWORD
]
gdi32.CreateDIBSection.restype = w.HBITMAP

gdi32.SelectObject.argtypes = [w.HDC, w.HGDIOBJ]
gdi32.SelectObject.restype = w.HGDIOBJ

gdi32.BitBlt.argtypes = [w.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          w.HDC, ctypes.c_int, ctypes.c_int, w.DWORD]
gdi32.BitBlt.restype = w.BOOL

gdi32.StretchBlt.argtypes = [w.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             w.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.DWORD]
gdi32.StretchBlt.restype = w.BOOL

gdi32.SetStretchBltMode.argtypes = [w.HDC, ctypes.c_int]
gdi32.SetStretchBltMode.restype = w.BOOL

gdi32.SetBrushOrgEx.argtypes = [w.HDC, ctypes.c_int, ctypes.c_int, ctypes.POINTER(w.POINT)]
gdi32.SetBrushOrgEx.restype = w.BOOL

gdi32.DeleteObject.argtypes = [w.HGDIOBJ]
gdi32.DeleteObject.restype = w.BOOL

gdi32.DeleteDC.argtypes = [w.HDC]
gdi32.DeleteDC.restype = w.BOOL

user32.GetDC.argtypes = [w.HWND]
user32.GetDC.restype = w.HDC

user32.ReleaseDC.argtypes = [w.HWND, w.HDC]
user32.ReleaseDC.restype = ctypes.c_int

user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int

user32.SendInput.argtypes = [w.UINT, ctypes.c_void_p, ctypes.c_int]
user32.SendInput.restype = w.UINT

def capture_and_encode(sw: int, sh: int, dw: int, dh: int) -> bytes:
    sdc = user32.GetDC(0)
    if not sdc:
        raise RuntimeError("GetDC failed")
    mdc = gdi32.CreateCompatibleDC(sdc)
    if not mdc:
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("CreateCompatibleDC failed")
    
    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = sw
    bmi.bmiHeader.biHeight = -sh
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    
    bits = ctypes.c_void_p()
    hbmp = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    if not hbmp:
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("CreateDIBSection failed")
    
    old = gdi32.SelectObject(mdc, hbmp)
    if not old:
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("SelectObject failed")
    
    if not gdi32.BitBlt(mdc, 0, 0, sw, sh, sdc, 0, 0, 0x40CC0020):
        gdi32.SelectObject(mdc, old)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("BitBlt failed")
    
    bgra = ctypes.string_at(bits, sw * sh * 4)
    
    gdi32.SelectObject(mdc, old)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(mdc)
    user32.ReleaseDC(0, sdc)
    
    if (sw, sh) != (dw, dh):
        bgra = downsample(bgra, sw, sh, dw, dh)
        sw, sh = dw, dh
    
    rgba = bytearray(len(bgra))
    for i in range(0, len(bgra), 4):
        rgba[i:i+4] = [bgra[i+2], bgra[i+1], bgra[i], 255]
    
    raw = bytearray()
    stride = sw * 4
    for y in range(sh):
        raw.append(0)
        raw.extend(rgba[y*stride:(y+1)*stride])
    
    ihdr = struct.pack(">IIBBBBB", sw, sh, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 6)
    
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag+data) & 0xFFFFFFFF)
    
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

def downsample(src: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes:
    sdc = user32.GetDC(0)
    if not sdc:
        raise RuntimeError("GetDC failed")
    src_dc = gdi32.CreateCompatibleDC(sdc)
    if not src_dc:
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("CreateCompatibleDC (src) failed")
    dst_dc = gdi32.CreateCompatibleDC(sdc)
    if not dst_dc:
        gdi32.DeleteDC(src_dc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("CreateCompatibleDC (dst) failed")
    
    bmi_src = BITMAPINFO()
    bmi_src.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi_src.bmiHeader.biWidth = sw
    bmi_src.bmiHeader.biHeight = -sh
    bmi_src.bmiHeader.biPlanes = 1
    bmi_src.bmiHeader.biBitCount = 32
    
    src_bits = ctypes.c_void_p()
    src_bmp = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi_src), 0, ctypes.byref(src_bits), None, 0)
    if not src_bmp:
        gdi32.DeleteDC(dst_dc)
        gdi32.DeleteDC(src_dc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("CreateDIBSection (src) failed")
    
    if len(src) != sw * sh * 4:
        gdi32.DeleteObject(src_bmp)
        gdi32.DeleteDC(dst_dc)
        gdi32.DeleteDC(src_dc)
        user32.ReleaseDC(0, sdc)
        raise ValueError("Source buffer size mismatch")
    
    ctypes.memmove(src_bits, src, len(src))
    old_src = gdi32.SelectObject(src_dc, src_bmp)
    if not old_src:
        gdi32.DeleteObject(src_bmp)
        gdi32.DeleteDC(dst_dc)
        gdi32.DeleteDC(src_dc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("SelectObject (src) failed")
    
    bmi_dst = BITMAPINFO()
    bmi_dst.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi_dst.bmiHeader.biWidth = dw
    bmi_dst.bmiHeader.biHeight = -dh
    bmi_dst.bmiHeader.biPlanes = 1
    bmi_dst.bmiHeader.biBitCount = 32
    
    dst_bits = ctypes.c_void_p()
    dst_bmp = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi_dst), 0, ctypes.byref(dst_bits), None, 0)
    if not dst_bmp:
        gdi32.SelectObject(src_dc, old_src)
        gdi32.DeleteObject(src_bmp)
        gdi32.DeleteDC(dst_dc)
        gdi32.DeleteDC(src_dc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("CreateDIBSection (dst) failed")
    
    old_dst = gdi32.SelectObject(dst_dc, dst_bmp)
    if not old_dst:
        gdi32.DeleteObject(dst_bmp)
        gdi32.SelectObject(src_dc, old_src)
        gdi32.DeleteObject(src_bmp)
        gdi32.DeleteDC(dst_dc)
        gdi32.DeleteDC(src_dc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("SelectObject (dst) failed")
    
    if not gdi32.SetStretchBltMode(dst_dc, 4):  # HALFTONE
        gdi32.SelectObject(dst_dc, old_dst)
        gdi32.DeleteObject(dst_bmp)
        gdi32.SelectObject(src_dc, old_src)
        gdi32.DeleteObject(src_bmp)
        gdi32.DeleteDC(dst_dc)
        gdi32.DeleteDC(src_dc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("SetStretchBltMode failed")
    
    if not gdi32.SetBrushOrgEx(dst_dc, 0, 0, None):
        gdi32.SelectObject(dst_dc, old_dst)
        gdi32.DeleteObject(dst_bmp)
        gdi32.SelectObject(src_dc, old_src)
        gdi32.DeleteObject(src_bmp)
        gdi32.DeleteDC(dst_dc)
        gdi32.DeleteDC(src_dc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("SetBrushOrgEx failed")
    
    if not gdi32.StretchBlt(dst_dc, 0, 0, dw, dh, src_dc, 0, 0, sw, sh, 0x00CC0020):
        gdi32.SelectObject(dst_dc, old_dst)
        gdi32.DeleteObject(dst_bmp)
        gdi32.SelectObject(src_dc, old_src)
        gdi32.DeleteObject(src_bmp)
        gdi32.DeleteDC(dst_dc)
        gdi32.DeleteDC(src_dc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("StretchBlt failed")
    
    out = ctypes.string_at(dst_bits, dw * dh * 4)
    
    gdi32.SelectObject(src_dc, old_src)
    gdi32.SelectObject(dst_dc, old_dst)
    gdi32.DeleteObject(src_bmp)
    gdi32.DeleteObject(dst_bmp)
    gdi32.DeleteDC(src_dc)
    gdi32.DeleteDC(dst_dc)
    user32.ReleaseDC(0, sdc)
    
    return out

def to_screen(xn: int, yn: int, sw: int, sh: int) -> tuple[int, int]:
    x = int(max(0, min(1000, xn)) / 1000 * (sw - 1))
    y = int(max(0, min(1000, yn)) / 1000 * (sh - 1))
    return x, y

def to_abs(x: int, y: int, sw: int, sh: int) -> tuple[int, int]:
    return int(x * 65535 / max(1, sw-1)), int(y * 65535 / max(1, sh-1))

def send_input(inp: INPUT) -> None:
    arr = (INPUT * 1)(inp)
    if user32.SendInput(1, ctypes.byref(arr), ctypes.sizeof(INPUT)) != 1:
        raise RuntimeError("SendInput failed")

def mouse_move(x: int, y: int, sw: int, sh: int) -> None:
    ax, ay = to_abs(x, y, sw, sh)
    send_input(INPUT(0, _INPUTunion(mi=MOUSEINPUT(ax, ay, 0, 0x8001, 0, 0))))

def mouse_click(x: int, y: int, sw: int, sh: int) -> None:
    mouse_move(x, y, sw, sh)
    send_input(INPUT(0, _INPUTunion(mi=MOUSEINPUT(0, 0, 0, 0x0002, 0, 0))))
    send_input(INPUT(0, _INPUTunion(mi=MOUSEINPUT(0, 0, 0, 0x0004, 0, 0))))

def mouse_right(x: int, y: int, sw: int, sh: int) -> None:
    mouse_move(x, y, sw, sh)
    send_input(INPUT(0, _INPUTunion(mi=MOUSEINPUT(0, 0, 0, 0x0008, 0, 0))))
    send_input(INPUT(0, _INPUTunion(mi=MOUSEINPUT(0, 0, 0, 0x0010, 0, 0))))

def mouse_double(x: int, y: int, sw: int, sh: int) -> None:
    mouse_click(x, y, sw, sh)
    time.sleep(0.05)
    mouse_click(x, y, sw, sh)

def type_text(text: str) -> None:
    utf16 = text.encode("utf-16le", errors="surrogatepass")
    for i in range(0, len(utf16), 2):
        code = utf16[i] | (utf16[i+1] << 8)
        send_input(INPUT(1, _INPUTunion(ki=KEYBDINPUT(0, code, 0x0004, 0, 0))))
        send_input(INPUT(1, _INPUTunion(ki=KEYBDINPUT(0, code, 0x0006, 0, 0))))

def scroll(dy: int) -> None:
    md = max(-100, min(100, dy)) * 120
    send_input(INPUT(0, _INPUTunion(mi=MOUSEINPUT(0, 0, md, 0x0800, 0, 0))))

@dataclass
class State:
    story: str
    paused: bool
    stop: bool
    step: int
    lock: threading.Lock
    
    def get_story(self) -> str:
        with self.lock:
            return self.story
    
    def set_story(self, s: str) -> None:
        with self.lock:
            self.story = s
    
    def is_paused(self) -> bool:
        with self.lock:
            return self.paused
    
    def set_paused(self, p: bool) -> None:
        with self.lock:
            self.paused = p
    
    def should_stop(self) -> bool:
        with self.lock:
            return self.stop
    
    def request_stop(self) -> None:
        with self.lock:
            self.stop = True
    
    def get_step(self) -> int:
        with self.lock:
            return self.step
    
    def inc_step(self) -> int:
        with self.lock:
            self.step += 1
            return self.step

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass
    
    def do_GET(self) -> None:
        if self.path == "/":
            with self.server.state.lock:
                data = json.dumps({
                    "step": self.server.state.step,
                    "paused": self.server.state.paused,
                    "story": self.server.state.story
                }, indent=2)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self) -> None:
        if self.path == "/pause":
            self.server.state.set_paused(True)
        elif self.path == "/resume":
            self.server.state.set_paused(False)
        elif self.path == "/stop":
            self.server.state.request_stop()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()

def start_server(state: State, port: int) -> threading.Thread:
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.state = state
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return t

def validate_story(s: str) -> bool:
    tags = ["[IDENTITY]", "[PAST]", "[NOW]", "[WHERE]", "[DELTA]", "[NEXT]", "[CONF]"]
    return all(tag in s for tag in tags)

def truncate_story(s: str) -> str:
    if len(s) <= STORY_MAX:
        return s
    cut = s.rfind("\n", 0, STORY_MAX)
    if cut > STORY_MAX * 0.6:
        return s[:cut]
    return s[:STORY_MAX]

def sanitize_int(x: Any, lo: int, hi: int) -> int | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return max(lo, min(hi, x))
    if isinstance(x, float):
        return max(lo, min(hi, int(x)))
    if isinstance(x, str):
        try:
            return max(lo, min(hi, int(float(x.strip()))))
        except:
            return None
    return None

def parse_tool_calls(obj: Any) -> list[tuple[str, dict]]:
    out = []
    if not isinstance(obj, list):
        return out
    for tc in obj:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        name = fn.get("name", "").strip().lower()
        args_raw = fn.get("arguments", "")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                continue
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            continue
        if isinstance(args, dict):
            out.append((name, args))
    return out

def sanitize_args(tool: str, args: dict) -> dict | None:
    if tool in ("click", "right_click", "double_click"):
        x = sanitize_int(args.get("x"), 0, 1000)
        y = sanitize_int(args.get("y"), 0, 1000)
        if x is None or y is None:
            return None
        return {"x": x, "y": y}
    elif tool == "type_text":
        return {"text": str(args.get("text", ""))}
    elif tool == "scroll":
        dy = sanitize_int(args.get("dy"), -100, 100)
        if dy is None:
            return None
        return {"dy": dy}
    elif tool == "wait":
        ms = sanitize_int(args.get("ms"), 1, 10000)
        if ms is None:
            return None
        return {"ms": ms}
    elif tool == "commit_story":
        s = str(args.get("story", "")).strip()
        if not s or len(s) < MIN_STORY_LEN:
            return None
        return {"story": truncate_story(s)}
    return None

def execute_tool(tool: str, args: dict, sw: int, sh: int) -> None:
    if tool == "click":
        x, y = to_screen(args["x"], args["y"], sw, sh)
        mouse_click(x, y, sw, sh)
    elif tool == "right_click":
        x, y = to_screen(args["x"], args["y"], sw, sh)
        mouse_right(x, y, sw, sh)
    elif tool == "double_click":
        x, y = to_screen(args["x"], args["y"], sw, sh)
        mouse_double(x, y, sw, sh)
    elif tool == "type_text":
        type_text(args["text"])
    elif tool == "scroll":
        scroll(args["dy"])
    elif tool == "wait":
        time.sleep(args["ms"] / 1000)

def vlm_infer(api: str, model: str, png: bytes, story: str, last: str) -> tuple[list[tuple[str, dict]], str]:
    prompt = (
        f"Current story:\n{story}\n\n"
        f"Last action: {last}\n\n"
        "Rules:\n"
        "1. If no clear task: wait(3000) + commit_story\n"
        "2. If clear task: 1-3 actions + commit_story\n"
        "3. commit_story MUST preserve all 7 tags and similar length\n"
        "4. Never interact with browser pages showing JSON story"
    )
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(png).decode()}"}},
                ]
            }
        ],
        "tools": TOOLS,
        "tool_choice": "auto"
    }
    payload.update(SAMPLING)
    
    for attempt in range(3):  # Retries
        try:
            req = urllib.request.Request(api, json.dumps(payload).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as f:
                resp = json.load(f)
            msg = resp.get("choices", [{}])[0].get("message", {})
            calls = parse_tool_calls(msg.get("tool_calls", []))
            content = msg.get("content", "")
            return calls, content
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)  # Backoff

def fallback_story(api: str, model: str, png: bytes, story: str) -> str | None:
    tools = [t for t in TOOLS if t["function"]["name"] == "commit_story"]
    prompt = f"Rewrite this story now:\n{story}\n\nPreserve all tags and length."
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(png).decode()}"}},
                ]
            }
        ],
        "tools": tools,
        "tool_choice": "required"
    }
    payload.update(SAMPLING)
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(api, json.dumps(payload).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as f:
                resp = json.load(f)
            msg = resp.get("choices", [{}])[0].get("message", {})
            calls = parse_tool_calls(msg.get("tool_calls", []))
            for tool, args in calls:
                if tool == "commit_story":
                    clean = sanitize_args(tool, args)
                    if clean:
                        return clean["story"]
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            if attempt == 2:
                return None
            time.sleep(2 ** attempt)
    return None

def log_event(path: Path, obj: dict) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except (PermissionError, OSError) as e:
        print(f"Log write failed: {e}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--min-step-sec", type=float, default=2.0)
    args = parser.parse_args()
    
    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)
    
    dump = Path("dump") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    try:
        dump.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        print(f"Dump dir creation failed: {e}")
        dump = Path(".")
    
    log_path = dump / "events.jsonl"
    
    state = State(INIT_STORY, True, False, 0, threading.Lock())
    start_server(state, args.port)
    
    print(f"FRANZ | Screen {sw}x{sh} -> Model {RES_W}x{RES_H}")
    print(f"API: {args.api_url}")
    print(f"Log: {dump}")
    print(f"HUD: http://localhost:{args.port}/")
    print("PAUSED - POST /resume to start")
    
    story = state.get_story()
    last_action = "init"
    last_crc = 0
    idle = 0
    
    try:
        while not state.should_stop():
            while state.is_paused() and not state.should_stop():
                time.sleep(0.1)
            if state.should_stop():
                break
            
            start_time = time.time()
            step = state.inc_step()
            ts = datetime.now().strftime("%H:%M:%S")
            
            story = state.get_story()
            
            png = capture_and_encode(sw, sh, RES_W, RES_H)
            crc = zlib.crc32(png) & 0xFFFFFFFF
            changed = crc != last_crc
            last_crc = crc
            
            if not changed and idle > 2:
                print(f"[{ts}] {step:03d} | SKIP unchanged screen")
                time.sleep(args.min_step_sec)
                idle += 1
                continue
            
            try:
                (dump / f"step{step:03d}.png").write_bytes(png)
            except (PermissionError, OSError) as e:
                print(f"PNG write failed: {e}")
            
            log_event(log_path, {"ts": ts, "step": step, "event": "capture", "crc": f"{crc:08x}", "changed": changed})
            
            try:
                calls, content = vlm_infer(args.api_url, args.model, png, story, last_action)
            except Exception as e:
                print(f"[{ts}] {step:03d} | VLM ERROR: {e}")
                log_event(log_path, {"ts": ts, "step": step, "event": "vlm_error", "error": str(e)})
                time.sleep(1)
                continue
            
            if not calls:
                print(f"[{ts}] {step:03d} | NO CALLS | content: {content[:100]}")
                log_event(log_path, {"ts": ts, "step": step, "event": "no_calls", "content": content[:200]})
                fb = fallback_story(args.api_url, args.model, png, story)
                if fb and validate_story(fb):
                    story = fb
                    state.set_story(story)
                    try:
                        (dump / "story.txt").write_text(story, encoding="utf-8")
                    except (PermissionError, OSError):
                        pass
                time.sleep(1)
                continue
            
            valid_calls = []
            story_candidate = None
            
            for tool, raw_args in calls[:5]:
                clean = sanitize_args(tool, raw_args)
                if clean is None:
                    log_event(log_path, {"ts": ts, "step": step, "event": "invalid_args", "tool": tool, "args": raw_args})
                    continue
                valid_calls.append((tool, clean))
                if tool == "commit_story":
                    story_candidate = clean["story"]
            
            if not valid_calls:
                print(f"[{ts}] {step:03d} | ALL INVALID")
                time.sleep(1)
                continue
            
            any_action = False
            for idx, (tool, tool_args) in enumerate(valid_calls, 1):
                if state.should_stop() or state.is_paused():
                    break
                
                summary = f"{tool}({','.join(f'{k}={v}' for k,v in tool_args.items())})"
                print(f"[{ts}] {step:03d}.{idx} | {summary}")
                
                try:
                    execute_tool(tool, tool_args, sw, sh)
                    log_event(log_path, {"ts": ts, "step": step, "idx": idx, "tool": tool, "args": tool_args, "ok": True})
                    if tool not in ("wait", "commit_story"):
                        any_action = True
                        last_action = summary
                except Exception as e:
                    print(f"[{ts}] {step:03d}.{idx} | ERROR: {e}")
                    log_event(log_path, {"ts": ts, "step": step, "idx": idx, "tool": tool, "error": str(e), "ok": False})
                
                time.sleep(0.05)
            
            if story_candidate and validate_story(story_candidate):
                story = story_candidate
                state.set_story(story)
                try:
                    (dump / "story.txt").write_text(story, encoding="utf-8")
                except (PermissionError, OSError):
                    pass
                log_event(log_path, {"ts": ts, "step": step, "event": "story_commit", "len": len(story)})
            else:
                fb = fallback_story(args.api_url, args.model, png, story)
                if fb and validate_story(fb):
                    story = fb
                    state.set_story(story)
                    try:
                        (dump / "story.txt").write_text(story, encoding="utf-8")
                    except (PermissionError, OSError):
                        pass
                    log_event(log_path, {"ts": ts, "step": step, "event": "story_fallback", "len": len(story)})
            
            if not changed and not any_action:
                idle += 1
            else:
                idle = 0
            
            elapsed = time.time() - start_time
            delay = args.min_step_sec * (1 + 0.3 * min(idle, 5))
            if elapsed < delay:
                time.sleep(delay - elapsed)
    
    except KeyboardInterrupt:
        print("\nStopped by user")

if __name__ == "__main__":
    main()