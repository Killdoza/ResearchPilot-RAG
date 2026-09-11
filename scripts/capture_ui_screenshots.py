"""Capture deterministic multi-viewport Streamlit screenshots through CDP."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import websocket


@dataclass(frozen=True)
class Viewport:
    name: str
    width: int
    height: int
    device_scale_factor: float = 1.0
    mobile: bool = False


VIEWPORTS = (
    Viewport("mobile_390x844", 390, 844, mobile=True),
    Viewport("tablet_768x1024", 768, 1024, mobile=True),
    Viewport("desktop_1366x900", 1366, 900),
    Viewport("desktop_1440x900_scale125", 1440, 900, 1.25),
    Viewport("desktop_1920x1080", 1920, 1080),
)


def _json_get(url: str) -> object:
    with urllib.request.urlopen(url, timeout=3) as response:  # noqa: S310 - localhost CDP only
        return json.loads(response.read().decode("utf-8"))


def _wait_for_debugger(port: int, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            pages = _json_get(f"http://127.0.0.1:{port}/json/list")
            if isinstance(pages, list):
                page = next((item for item in pages if item.get("type") == "page"), None)
                if page and isinstance(page.get("webSocketDebuggerUrl"), str):
                    return page["webSocketDebuggerUrl"]
        except (OSError, ValueError):
            pass
        time.sleep(0.25)
    raise TimeoutError("Chrome DevTools debugger did not become ready.")


class CdpClient:
    def __init__(self, websocket_url: str) -> None:
        self._socket = websocket.create_connection(websocket_url, timeout=10)
        self._identifier = 0

    def close(self) -> None:
        self._socket.close()

    def call(self, method: str, params: dict[str, object] | None = None) -> dict[str, object]:
        self._identifier += 1
        identifier = self._identifier
        self._socket.send(json.dumps({"id": identifier, "method": method, "params": params or {}}))
        while True:
            payload = json.loads(self._socket.recv())
            if payload.get("id") == identifier:
                if "error" in payload:
                    raise RuntimeError(f"CDP {method} failed: {payload['error']}")
                result = payload.get("result", {})
                return result if isinstance(result, dict) else {}


def _wait_for_streamlit(client: CdpClient, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    expression = """
(() => {
  const app = document.querySelector('[data-testid="stAppViewContainer"]');
  const text = document.body.innerText || '';
  return {
    ready: Boolean(
      app && document.title.includes('ResearchPilot') && text.trim().length > 100
    ),
    title: document.title,
    hasApp: Boolean(app),
    textLength: text.trim().length,
    textPreview: text.trim().slice(0, 160)
  };
})()
"""
    observation: object = None
    while time.monotonic() < deadline:
        result = client.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        payload = result.get("result", {})
        observation = payload.get("value") if isinstance(payload, dict) else None
        if isinstance(observation, dict) and observation.get("ready") is True:
            return
        time.sleep(0.5)
    raise TimeoutError(
        f"Streamlit UI did not become ready before screenshot timeout. Last DOM observation: {observation!r}"
    )


def capture(*, chrome: Path, url: str, output_dir: Path, port: int, ready_timeout: float = 90) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = output_dir / "chrome_profile"
    command = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--disable-default-apps",
        "--remote-allow-origins=*",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        "about:blank",
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    process = subprocess.Popen(  # noqa: S603 - exact local Chrome executable supplied by caller
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    paths: list[Path] = []
    try:
        client = CdpClient(_wait_for_debugger(port, 15))
        try:
            client.call("Page.enable")
            for viewport in VIEWPORTS:
                client.call(
                    "Emulation.setDeviceMetricsOverride",
                    {
                        "width": viewport.width,
                        "height": viewport.height,
                        "deviceScaleFactor": viewport.device_scale_factor,
                        "mobile": viewport.mobile,
                    },
                )
                client.call("Page.navigate", {"url": url})
                _wait_for_streamlit(client, ready_timeout)
                time.sleep(1)
                result = client.call(
                    "Page.captureScreenshot",
                    {"format": "png", "captureBeyondViewport": False, "fromSurface": True},
                )
                encoded = result.get("data")
                if not isinstance(encoded, str):
                    raise RuntimeError("Chrome did not return screenshot data.")
                path = output_dir / f"{viewport.name}.png"
                path.write_bytes(base64.b64decode(encoded))
                paths.append(path)
        finally:
            client.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--debug-port", type=int, default=9227)
    parser.add_argument("--ready-timeout", type=float, default=90)
    parser.add_argument(
        "--chrome",
        type=Path,
        default=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    )
    args = parser.parse_args()
    if not args.chrome.is_file():
        raise FileNotFoundError(f"Chrome executable not found: {args.chrome}")
    for path in capture(
        chrome=args.chrome,
        url=args.url,
        output_dir=args.output_dir,
        port=args.debug_port,
        ready_timeout=args.ready_timeout,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
