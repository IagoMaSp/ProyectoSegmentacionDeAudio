import argparse
import json
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from src.config import BASE_DIR, SAMPLE_RATE
from src.preprocessing import scale_pcm16
from src.segmenter import load_inference_artifacts, merge_timeline_intervals, segment_signal

INDEX_HTML = (BASE_DIR / "web" / "index.html").read_bytes()
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
SCALER, MODEL = load_inference_artifacts()


def decode_audio(data: bytes) -> np.ndarray:
    # ffmpeg acepta mp3, ogg, flac, m4a, opus, wav con cualquier frecuencia, etc. y lo lleva a 16 kHz mono
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(data)
        tmp.flush()
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", tmp.name, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
            capture_output=True,
        )
    if result.returncode != 0:
        raise ValueError("No se pudo decodificar el audio: " + result.stderr.decode(errors="replace").strip())
    return scale_pcm16(np.frombuffer(result.stdout, dtype=np.int16))


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        else:
            self._send(404, b"", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/segment":
            return self._send(404, b"", "text/plain")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            return self._json(400, {"error": "Archivo vacio o mayor a 200 MB."})
        try:
            signal = decode_audio(self.rfile.read(length))
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        timeline = segment_signal(signal, SCALER, MODEL)
        self._json(200, {"duration": round(signal.size / SAMPLE_RATE, 2), "segments": merge_timeline_intervals(timeline)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Frontend web para segmentar audio.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Abrir http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
