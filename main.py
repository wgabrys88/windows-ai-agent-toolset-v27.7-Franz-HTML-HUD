# main.py
# FRANZ - Narrative Desktop Agent (Windows 11 + Python 3.12)
# Standard library only. Stateless API. Story is the only durable state.

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


# ctypes.wintypes does not always define ULONG_PTR (common on some Python builds).
# Define it safely based on pointer size.
try:
    ULONG_PTR = w.ULONG_PTR  # type: ignore[attr-defined]
except AttributeError:
    ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32




API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen3-vl-2b-instruct-1m"

# Model vision resolution (sent to the API)
RES_W, RES_H = 1536, 864

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
    "[PAST] Initialized in isolated environment.\n"
    "[NOW] Awaiting task instructions.\n"
    "[WHERE] Windows desktop.\n"
    "[DELTA] System idle.\n"
    "[NEXT] Observe and act.\n"
    "[CONF] 0.00"
)

SYSTEM_PROMPT = (
    "You are FRANZ, a Windows desktop controller. Use only the provided tools. "
    "The story is the only durable state across steps. "
    "Every response MUST end with commit_story(story). "
    "Rewrite the FULL story each time, preserving the 7 tags: "
    "[IDENTITY] [PAST] [NOW] [WHERE] [DELTA] [NEXT] [CONF]. "
    "The story MUST ALWAYS be between 1000 and 1400 tokens. "
    "All click coordinates are normalized integers in range 0..1000 "
    "(0,0 is top-left of the captured screen; 1000,1000 is bottom-right). "
    "If uncertain, still commit a story update describing uncertainty and next observation."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Left click at x,y (normalized 0-1000)",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "right_click",
            "description": "Right click at x,y",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": "Double click at x,y",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
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
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll wheel, dy integer (positive down, negative up)",
            "parameters": {
                "type": "object",
                "properties": {"dy": {"type": "integer"}},
                "required": ["dy"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait milliseconds",
            "parameters": {
                "type": "object",
                "properties": {"ms": {"type": "integer"}},
                "required": ["ms"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_story",
            "description": "Commit the full updated story text. Must contain all 7 tags.",
            "parameters": {
                "type": "object",
                "properties": {"story": {"type": "string"}},
                "required": ["story"],
                "additionalProperties": False,
            },
        },
    },
]


# --- Win32 / GDI / Input ---

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

SM_CXSCREEN = 0
SM_CYSCREEN = 1

BI_RGB = 0
DIB_RGB_COLORS = 0
SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", w.DWORD),
        ("biWidth", w.LONG),
        ("biHeight", w.LONG),
        ("biPlanes", w.WORD),
        ("biBitCount", w.WORD),
        ("biCompression", w.DWORD),
        ("biSizeImage", w.DWORD),
        ("biXPelsPerMeter", w.LONG),
        ("biYPelsPerMeter", w.LONG),
        ("biClrUsed", w.DWORD),
        ("biClrImportant", w.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", w.DWORD * 3)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", w.LONG),
        ("dy", w.LONG),
        ("mouseData", w.DWORD),
        ("dwFlags", w.DWORD),
        ("time", w.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", w.WORD),
        ("wScan", w.WORD),
        ("dwFlags", w.DWORD),
        ("time", w.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", w.DWORD), ("u", _INPUTunion)]


send_input = user32.SendInput
send_input.argtypes = (w.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
send_input.restype = w.UINT

user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
user32.GetSystemMetrics.restype = ctypes.c_int


def capture_rgba(sw: int, sh: int) -> bytes:
    sdc = user32.GetDC(0)
    if not sdc:
        raise RuntimeError("GetDC failed")

    memdc = gdi32.CreateCompatibleDC(sdc)
    if not memdc:
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("CreateCompatibleDC failed")

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = sw
    bmi.bmiHeader.biHeight = -sh
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = BI_RGB

    bits = ctypes.c_void_p()
    hbmp = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi), DIB_RGB_COLORS, ctypes.byref(bits), None, 0)
    if not hbmp:
        gdi32.DeleteDC(memdc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("CreateDIBSection failed")

    old = gdi32.SelectObject(memdc, hbmp)
    if not old:
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(memdc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("SelectObject failed")

    ok = gdi32.BitBlt(memdc, 0, 0, sw, sh, sdc, 0, 0, SRCCOPY | CAPTUREBLT)
    if not ok:
        gdi32.SelectObject(memdc, old)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(memdc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("BitBlt failed")

    size = sw * sh * 4
    buf = (ctypes.c_ubyte * size).from_address(bits.value)
    raw = bytes(buf)

    gdi32.SelectObject(memdc, old)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(memdc)
    user32.ReleaseDC(0, sdc)
    return raw


def rgba_to_png(rgba: bytes, sw: int, sh: int) -> bytes:
    # Minimal PNG (RGBA) encoder using zlib from stdlib.
    raw = bytearray()
    stride = sw * 4
    for y in range(sh):
        raw.append(0)  # filter type 0
        raw.extend(rgba[y * stride : (y + 1) * stride])

    ihdr = struct.pack(">IIBBBBB", sw, sh, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        # PNG chunks require CRC32; this is part of the file format.
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

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
        gdi32.SelectObject(src_dc, old_src)
        gdi32.DeleteObject(dst_bmp)
        gdi32.DeleteObject(src_bmp)
        gdi32.DeleteDC(dst_dc)
        gdi32.DeleteDC(src_dc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("SelectObject (dst) failed")

    gdi32.SetStretchBltMode(dst_dc, 4)  # HALFTONE
    ok = gdi32.StretchBlt(dst_dc, 0, 0, dw, dh, src_dc, 0, 0, sw, sh, SRCCOPY)
    if not ok:
        gdi32.SelectObject(dst_dc, old_dst)
        gdi32.SelectObject(src_dc, old_src)
        gdi32.DeleteObject(dst_bmp)
        gdi32.DeleteObject(src_bmp)
        gdi32.DeleteDC(dst_dc)
        gdi32.DeleteDC(src_dc)
        user32.ReleaseDC(0, sdc)
        raise RuntimeError("StretchBlt failed")

    out = (ctypes.c_ubyte * (dw * dh * 4)).from_address(dst_bits.value)
    out_bytes = bytes(out)

    gdi32.SelectObject(dst_dc, old_dst)
    gdi32.SelectObject(src_dc, old_src)
    gdi32.DeleteObject(dst_bmp)
    gdi32.DeleteObject(src_bmp)
    gdi32.DeleteDC(dst_dc)
    gdi32.DeleteDC(src_dc)
    user32.ReleaseDC(0, sdc)
    return out_bytes


def capture_and_encode(sw: int, sh: int, dw: int, dh: int) -> bytes:
    rgba = capture_rgba(sw, sh)
    if sw != dw or sh != dh:
        rgba = downsample(rgba, sw, sh, dw, dh)
        sw, sh = dw, dh
    return rgba_to_png(rgba, sw, sh)


def to_screen(xn: int, yn: int, sw: int, sh: int) -> tuple[int, int]:
    # No clamping: model owns the coordinate contract via prompt.
    x = int((xn / 1000) * (sw - 1))
    y = int((yn / 1000) * (sh - 1))
    return x, y


def to_abs(x: int, y: int, sw: int, sh: int) -> tuple[int, int]:
    return int(x * 65535 / max(1, sw - 1)), int(y * 65535 / max(1, sh - 1))


def mouse_move(x: int, y: int, sw: int, sh: int) -> None:
    ax, ay = to_abs(x, y, sw, sh)
    send_input(
        1,
        ctypes.byref(INPUT(INPUT_MOUSE, _INPUTunion(mi=MOUSEINPUT(ax, ay, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, 0)))),
        ctypes.sizeof(INPUT),
    )


def mouse_click(x: int, y: int, sw: int, sh: int) -> None:
    mouse_move(x, y, sw, sh)
    send_input(1, ctypes.byref(INPUT(INPUT_MOUSE, _INPUTunion(mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, 0)))), ctypes.sizeof(INPUT))
    send_input(1, ctypes.byref(INPUT(INPUT_MOUSE, _INPUTunion(mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, 0)))), ctypes.sizeof(INPUT))


def mouse_right(x: int, y: int, sw: int, sh: int) -> None:
    mouse_move(x, y, sw, sh)
    send_input(1, ctypes.byref(INPUT(INPUT_MOUSE, _INPUTunion(mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_RIGHTDOWN, 0, 0)))), ctypes.sizeof(INPUT))
    send_input(1, ctypes.byref(INPUT(INPUT_MOUSE, _INPUTunion(mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_RIGHTUP, 0, 0)))), ctypes.sizeof(INPUT))


def mouse_double(x: int, y: int, sw: int, sh: int) -> None:
    mouse_click(x, y, sw, sh)
    mouse_click(x, y, sw, sh)


def type_text(text: str) -> None:
    utf16 = text.encode("utf-16le", errors="surrogatepass")
    for i in range(0, len(utf16), 2):
        code = utf16[i] | (utf16[i + 1] << 8)
        send_input(1, ctypes.byref(INPUT(INPUT_KEYBOARD, _INPUTunion(ki=KEYBDINPUT(0, code, KEYEVENTF_UNICODE, 0, 0)))), ctypes.sizeof(INPUT))
        send_input(1, ctypes.byref(INPUT(INPUT_KEYBOARD, _INPUTunion(ki=KEYBDINPUT(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0)))), ctypes.sizeof(INPUT))


def scroll(dy: int) -> None:
    send_input(1, ctypes.byref(INPUT(INPUT_MOUSE, _INPUTunion(mi=MOUSEINPUT(0, 0, dy * 120, MOUSEEVENTF_WHEEL, 0, 0)))), ctypes.sizeof(INPUT))


# --- Agent state + HTTP ---

@dataclass
class State:
    story: str
    step: int
    lock: threading.Lock

    def get_story(self) -> str:
        with self.lock:
            return self.story

    def set_story(self, s: str) -> None:
        with self.lock:
            self.story = s

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
                data = json.dumps({"step": self.server.state.step, "story": self.server.state.story}, indent=2)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data.encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def start_server(state: State, port: int) -> threading.Thread:
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.state = state
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return t


# --- Model I/O ---

def parse_tool_calls(tool_calls: list[dict]) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for tc in tool_calls or []:
        fn = (tc.get("function") or {})
        name = fn.get("name")
        arg_s = fn.get("arguments") or "{}"
        try:
            args = json.loads(arg_s) if isinstance(arg_s, str) else dict(arg_s)
        except Exception:
            args = {}
        if name:
            out.append((str(name), args if isinstance(args, dict) else {}))
    return out


def vlm_infer(api: str, model: str, png: bytes, story: str, last: str) -> tuple[list[tuple[str, dict]], str]:
    prompt = (
        f"Current story:\n{story}\n\n"
        f"Last action: {last}\n\n"
        "Rules:\n"
        "1. Always end with commit_story(story).\n"
        "2. Story must keep all 7 tags and be 1000-1400 tokens.\n"
        "3. Coordinates are normalized 0..1000 integers.\n"
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
                ],
            },
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    payload.update(SAMPLING)

    req = urllib.request.Request(api, json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as f:
        resp = json.load(f)

    msg = resp.get("choices", [{}])[0].get("message", {})
    calls = parse_tool_calls(msg.get("tool_calls", []))
    content = msg.get("content", "") or ""
    return calls, content


def story_has_tags(s: str) -> bool:
    tags = ["[IDENTITY]", "[PAST]", "[NOW]", "[WHERE]", "[DELTA]", "[NEXT]", "[CONF]"]
    return all(tag in s for tag in tags)


def execute_tool(tool: str, args: dict, sw: int, sh: int) -> None:
    if tool == "click":
        x, y = to_screen(int(args["x"]), int(args["y"]), sw, sh)
        mouse_click(x, y, sw, sh)
    elif tool == "right_click":
        x, y = to_screen(int(args["x"]), int(args["y"]), sw, sh)
        mouse_right(x, y, sw, sh)
    elif tool == "double_click":
        x, y = to_screen(int(args["x"]), int(args["y"]), sw, sh)
        mouse_double(x, y, sw, sh)
    elif tool == "type_text":
        type_text(str(args.get("text", "")))
    elif tool == "scroll":
        scroll(int(args["dy"]))
    elif tool == "wait":
        time.sleep(int(args["ms"]) / 1000)
    elif tool == "commit_story":
        # Handled in the main loop to update state + write story.txt
        return
    else:
        raise ValueError(f"Unknown tool: {tool}")


def log_event(path: Path, obj: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    sw = user32.GetSystemMetrics(SM_CXSCREEN)
    sh = user32.GetSystemMetrics(SM_CYSCREEN)

    dump = Path("dump") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    try:
        dump.mkdir(parents=True, exist_ok=True)
    except Exception:
        dump = Path(".")

    log_path = dump / "events.jsonl"

    state = State(INIT_STORY, 0, threading.Lock())
    start_server(state, args.port)

    print(f"FRANZ | Screen {sw}x{sh} -> Model {RES_W}x{RES_H}")
    print(f"API: {args.api_url}")
    print(f"Log: {dump}")
    print(f"Story endpoint: http://localhost:{args.port}/")

    story = state.get_story()
    last_action = "init"

    try:
        while True:
            step = state.inc_step()
            ts = datetime.now().strftime("%H:%M:%S")

            story = state.get_story()
            png = capture_and_encode(sw, sh, RES_W, RES_H)

            try:
                (dump / f"step{step:06d}.png").write_bytes(png)
            except Exception:
                pass

            log_event(log_path, {"ts": ts, "step": step, "event": "capture"})

            try:
                calls, content = vlm_infer(args.api_url, args.model, png, story, last_action)
            except Exception as e:
                log_event(log_path, {"ts": ts, "step": step, "event": "vlm_error", "error": str(e)})
                continue
            if not calls:
                log_event(log_path, {"ts": ts, "step": step, "event": "no_calls", "content": content[:200]})
                continue

            story_candidate: str | None = None

            for idx, (tool, tool_args) in enumerate(calls, 1):
                summary = f"{tool}({','.join(f'{k}={v}' for k,v in (tool_args or {}).items())})"
                print(f"[{ts}] {step:06d}.{idx} {summary}")

                if tool == "commit_story":
                    story_candidate = str((tool_args or {}).get("story", ""))
                    continue

                try:
                    execute_tool(tool, tool_args or {}, sw, sh)
                    if tool not in ("wait",):
                        last_action = summary
                    log_event(log_path, {"ts": ts, "step": step, "idx": idx, "tool": tool, "args": tool_args, "ok": True})
                except Exception as e:
                    log_event(log_path, {"ts": ts, "step": step, "idx": idx, "tool": tool, "args": tool_args, "ok": False, "error": str(e)})

            if story_candidate is not None:
                story = story_candidate
                state.set_story(story)
                try:
                    (dump / "story.txt").write_text(story, encoding="utf-8")
                except Exception:
                    pass
                log_event(log_path, {"ts": ts, "step": step, "event": "story_commit", "len": len(story)})
                if not story_has_tags(story):
                    log_event(log_path, {"ts": ts, "step": step, "event": "story_missing_tags"})
            else:
                log_event(log_path, {"ts": ts, "step": step, "event": "story_missing"})

    except KeyboardInterrupt:
        print("\nStopped by user")


if __name__ == "__main__":
    main()
