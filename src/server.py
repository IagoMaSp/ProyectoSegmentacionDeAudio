"""Servidor local de la interfaz web. Solo biblioteca estandar + el pipeline del proyecto.

Uso:  python -m src.server [--port 8000] [--modelo models/modelo.joblib]

Rutas
  GET  /                       interfaz (web/index.html, app.js, style.css)
  GET  /api/estado             modelo cargado y parametros
  POST /api/segmentar          cuerpo = archivo; query: salto, suavizado, penalidad, minimo
  GET  /api/explicar           id, inicio, fin -> datos del panel "por que"
  GET  /api/audio              id, version=original|acondicionado -> WAV 16 kHz
  GET  /api/filtros            respuesta en frecuencia, polos/ceros, retardo de grupo
  GET  /api/aliasing           tono, fs, modo=tono|barrido -> espectros
  GET  /api/aliasing.wav       mismos parametros + variante=original|sin|con
  GET  /api/metricas           metricas de entrenamiento, robustez y grabaciones reales
  POST /api/vivo               sesion, fs; cuerpo = float32 little endian
  POST /api/vivo/fin           sesion
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from scipy import signal as sps

from src import config as C
from src import filters as F
from src.features import FRAME_FEATURE_NAMES
from src.live import LiveSession, spectrogram_columns
from src.preprocessing import decode_bytes, write_wav_bytes
from src.segmenter import analyze_signal, load_bundle

WEB_DIR = C.BASE_DIR / "web"
STATIC = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css", "/captura.js": "captura.js"}
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_ANALYSES = 4

STATE: dict[str, Any] = {"bundle": None, "bundle_error": None}
ANALYSES: "OrderedDict[str, dict]" = OrderedDict()
LIVE: dict[str, LiveSession] = {}
LOCK = threading.Lock()


def _bundle() -> dict:
    if STATE["bundle"] is None:
        raise RuntimeError(STATE["bundle_error"] or "No hay modelo cargado.")
    return STATE["bundle"]


def _q(query: dict, key: str, default, cast=float):
    try:
        return cast(query[key][0])
    except (KeyError, IndexError, ValueError):
        return default


def _envelope(x: np.ndarray, n_bins: int = 800) -> dict[str, list[float]]:
    if x.size == 0:
        return {"min": [], "max": []}
    n_bins = min(n_bins, x.size)
    edges = np.linspace(0, x.size, n_bins + 1).astype(int)
    mins = [float(x[a:b].min()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    maxs = [float(x[a:b].max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    return {"min": np.round(mins, 4).tolist(), "max": np.round(maxs, 4).tolist()}


def _decimate_cols(m: np.ndarray, max_cols: int) -> np.ndarray:
    if m.shape[0] <= max_cols:
        return m
    edges = np.linspace(0, m.shape[0], max_cols + 1).astype(int)
    return np.stack([m[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def api_estado() -> dict:
    b = STATE["bundle"]
    info = {
        "modelo_cargado": b is not None,
        "error": STATE["bundle_error"],
        "parametros": {
            "fs": C.SAMPLE_RATE, "ventana_s": C.SEGMENT_DURATION, "salto_s": C.WINDOW_HOP,
            "frame_ms": 1000 * C.N_WIN / C.SAMPLE_RATE, "hop_ms": 1000 * C.HOP_LENGTH / C.SAMPLE_RATE,
            "n_fft": C.N_FFT, "suavizado": C.SMOOTHING, "penalidad": C.TRANSITION_PENALTY,
            "duracion_minima": C.MIN_SEGMENT_DURATION, "hp_fc": C.HP_CUTOFF_HZ, "hp_orden": C.HP_ORDER,
            "notch": list(C.NOTCH_FREQS_HZ), "notch_q": C.NOTCH_Q, "paso_vivo_s": C.LIVE_STEP,
        },
    }
    if b is not None:
        info.update({"modelo": b["nombre"], "descripcion": b["descripcion"], "n_caracteristicas": len(b["feature_idx"])})
    return info


def api_segmentar(body: bytes, query: dict, filename: str) -> dict:
    bundle = _bundle()
    suffix = Path(filename).suffix if filename else ""
    fs_in, raw = decode_bytes(body, suffix=suffix)
    x = F.resample(raw, fs_in, C.SAMPLE_RATE)
    hop = min(max(_q(query, "salto", C.WINDOW_HOP), 0.1), 3.0)
    smoothing = _q(query, "suavizado", C.SMOOTHING, str)
    if smoothing not in ("ninguno", "mediana", "viterbi"):
        smoothing = C.SMOOTHING
    t0 = time.time()
    res = analyze_signal(x, bundle, hop=hop, smoothing=smoothing,
                         penalty=_q(query, "penalidad", C.TRANSITION_PENALTY),
                         min_duration=max(_q(query, "minimo", C.MIN_SEGMENT_DURATION), 0.0), keep_frames=True)
    elapsed = time.time() - t0
    aid = uuid.uuid4().hex[:12]
    with LOCK:
        ANALYSES[aid] = {
            "x": x, "y": F.condition_offline(x), "X": res.pop("_X"), "starts": res.pop("_starts"),
            "frames": res.pop("_frames"), "result": res,
        }
        while len(ANALYSES) > MAX_ANALYSES:
            ANALYSES.popitem(last=False)
    res.update({"id": aid, "fs_original": fs_in, "segundos_proceso": round(elapsed, 2),
                "envolvente": _envelope(x), "modelo": bundle["descripcion"]})
    return res


def api_explicar(query: dict) -> dict:
    aid = _q(query, "id", "", str)
    with LOCK:
        a = ANALYSES.get(aid)
    if a is None:
        raise KeyError("El analisis ya no esta en memoria; volver a procesar el archivo.")
    bundle = _bundle()
    start, end = _q(query, "inicio", 0.0), _q(query, "fin", 0.0)
    fs = C.SAMPLE_RATE
    a0, a1 = int(start * fs), int(min(end, a["x"].size / fs) * fs)
    y = a["y"][a0:a1]

    # espectrograma del tramo acondicionado
    spec = spectrogram_columns(y) if y.size >= C.N_WIN else np.zeros((0, 128))
    spec = _decimate_cols(spec, 500)

    # curvas por frame
    fr = a["frames"]
    f0, f1 = a0 // C.HOP_LENGTH, max(a0 // C.HOP_LENGTH + 1, (a1 - C.N_WIN) // C.HOP_LENGTH + 1)
    seg_frames = fr[f0:f1]
    curves = {}
    for name in ("rms", "zcr", "centroide", "planitud"):
        col = seg_frames[:, FRAME_FEATURE_NAMES.index(name)]
        if name == "rms":
            col = 20 * np.log10(col + 1e-9)
        curves[name] = np.round(_decimate_cols(col[:, None], 500)[:, 0], 4).tolist()

    # ventanas cuyo centro cae en el tramo
    centers = a["starts"] + C.SEGMENT_DURATION / 2
    sel = np.flatnonzero((centers >= start) & (centers <= end))
    if sel.size == 0:
        sel = np.array([int(np.argmin(np.abs(centers - (start + end) / 2)))])
    X = a["X"][sel][:, bundle["feature_idx"]]
    model = bundle["modelo"]
    scaler = model.named_steps["escalado"]
    Z = scaler.transform(X)
    z_mean = Z.mean(axis=0)
    p_mean = model.predict_proba(X).mean(axis=0)
    pca = bundle["pca"]
    pts = pca.transform(Z)
    center = pca.transform(z_mean[None, :])[0]

    d = np.linalg.norm(bundle["muestra_Z"] - z_mean[None, :], axis=1)
    nn = np.argsort(d)[:10]
    neighbors = [{"x": float(bundle["muestra_2d"][i, 0]), "y": float(bundle["muestra_2d"][i, 1]),
                  "label": C.CATEGORIES[int(bundle["muestra_y"][i])], "dist": round(float(d[i]), 3)} for i in nn]
    rng = np.random.default_rng(0)
    cloud_idx = rng.choice(len(bundle["muestra_y"]), size=min(1200, len(bundle["muestra_y"])), replace=False)
    cloud = [[round(float(bundle["muestra_2d"][i, 0]), 3), round(float(bundle["muestra_2d"][i, 1]), 3),
              int(bundle["muestra_y"][i])] for i in cloud_idx]

    # rasgos que mas separan la clase elegida de la segunda
    order = np.argsort(p_mean)[::-1]
    k1, k2 = int(order[0]), int(order[1])
    mu = np.stack([bundle["muestra_Z"][bundle["muestra_y"] == k].mean(axis=0) for k in range(len(C.CATEGORIES))])
    contrib = (z_mean - mu[k2]) ** 2 - (z_mean - mu[k1]) ** 2
    top = np.argsort(contrib)[::-1][:6]
    raw_mean = X.mean(axis=0)
    rasgos = [{"nombre": bundle["feature_names"][i], "valor": float(raw_mean[i]), "z": round(float(z_mean[i]), 2),
               "z_clase": round(float(mu[k1][i]), 2), "z_alternativa": round(float(mu[k2][i]), 2),
               "aporte": round(float(contrib[i]), 2)} for i in top]

    return {
        "inicio": start, "fin": end,
        "envolvente": _envelope(a["x"][a0:a1], 600),
        "espectrograma": {"db": np.round(spec, 1).tolist(), "fmax": 128 * 62.5},
        "curvas": curves,
        "probabilidades": {c: round(float(v), 4) for c, v in zip(C.CATEGORIES, p_mean)},
        "ventanas": {"inicio": np.round(a["starts"][sel], 2).tolist(),
                     "probs": np.round(model.predict_proba(X), 3).tolist()},
        "pca": {"punto": [float(center[0]), float(center[1])], "ventanas": np.round(pts, 3).tolist(),
                "vecinos": neighbors, "nube": cloud,
                "varianza": np.round(pca.explained_variance_ratio_, 3).tolist()},
        "rasgos": rasgos, "clase": C.CATEGORIES[k1], "alternativa": C.CATEGORIES[k2],
    }


def api_audio(query: dict) -> bytes:
    aid = _q(query, "id", "", str)
    with LOCK:
        a = ANALYSES.get(aid)
    if a is None:
        raise KeyError("Analisis no encontrado.")
    version = _q(query, "version", "original", str)
    x = a["y"] if version == "acondicionado" else a["x"]
    return write_wav_bytes(x, C.SAMPLE_RATE)


def api_filtros() -> dict:
    fs = C.SAMPLE_RATE
    freqs = np.geomspace(5, fs / 2 * 0.999, 600)
    hp = F.design_highpass(fs)
    notch = F.design_notch(fs, C.NOTCH_FREQS_HZ[0])
    total = F.design_conditioning(fs)

    def resp(sos):
        _, h = F.sos_frequency_response(sos, fs, freqs=freqs)
        return h

    def db(h):
        return np.round(20 * np.log10(np.maximum(np.abs(h), 1e-12)), 3)

    h_hp, h_n, h_t = resp(hp), resp(notch), resp(total)
    gd = np.zeros(freqs.size)
    for s in total:
        _, g = sps.group_delay((s[:3], s[3:]), w=freqs, fs=fs)
        gd += g
    z, p = F.sos_poles_zeros(total)
    h_aa = F.antialias_filter(48000, fs)
    f_aa, H_aa = sps.freqz(h_aa, worN=2048, fs=48000)
    return {
        "f": np.round(freqs, 3).tolist(),
        "hp_db": db(h_hp).tolist(), "notch_db": db(h_n).tolist(), "total_db": db(h_t).tolist(),
        "total_filtfilt_db": (2 * db(h_t)).tolist(),
        "fase_grados": np.round(np.degrees(np.unwrap(np.angle(h_t))), 2).tolist(),
        "retardo_ms": np.round(1000 * gd / fs, 3).tolist(),
        "ceros": [[round(float(c.real), 6), round(float(c.imag), 6)] for c in z],
        "polos": [[round(float(c.real), 6), round(float(c.imag), 6)] for c in p],
        "estable": F.is_stable(total),
        "radio_max_polos": round(float(np.max(np.abs(p))), 6),
        "parametros": {"hp_fc": C.HP_CUTOFF_HZ, "hp_orden": C.HP_ORDER, "notch_f0": C.NOTCH_FREQS_HZ[0],
                       "notch_q": C.NOTCH_Q, "notch_bw": C.NOTCH_FREQS_HZ[0] / C.NOTCH_Q, "fs": fs,
                       "secciones": int(total.shape[0])},
        "antialias": {"f": np.round(f_aa[::4], 1).tolist(), "db": db(H_aa)[::4].tolist(),
                      "coeficientes": int(h_aa.size), "atenuacion_db": C.AA_ATTENUATION_DB},
    }


ALIAS_FS_IN = 48000


def _aliasing_signal(query: dict) -> tuple[np.ndarray, int, str, float]:
    mode = _q(query, "modo", "tono", str)
    fs_out = int(_q(query, "fs", 8000))
    if fs_out not in (6000, 8000, 11025, 16000):
        fs_out = 8000
    tone = float(min(max(_q(query, "tono", 5000.0), 50.0), 23500.0))
    if mode == "barrido":
        t = np.arange(int(ALIAS_FS_IN * 4)) / ALIAS_FS_IN
        x = 0.3 * sps.chirp(t, f0=100, f1=20000, t1=4, method="linear")
    else:
        t = np.arange(int(ALIAS_FS_IN * 1.5)) / ALIAS_FS_IN
        x = 0.3 * np.sin(2 * np.pi * tone * t)
        fade = np.minimum(1, np.minimum(t, t[-1] - t) / 0.02)
        x *= fade
    return x, fs_out, mode, tone


def _spec_db(x: np.ndarray, fs: int, n_pts: int = 500) -> dict:
    n = min(x.size, 1 << 15)
    w = np.hanning(n)
    X = np.abs(np.fft.rfft(x[:n] * w)) / (w.sum() / 2)
    f = np.fft.rfftfreq(n, 1 / fs)
    db = 20 * np.log10(np.maximum(X, 1e-7))
    edges = np.linspace(0, f.size, n_pts + 1).astype(int)
    return {"f": [round(float(f[a]), 1) for a in edges[:-1]],
            "db": [round(float(db[a:b].max()), 1) for a, b in zip(edges[:-1], edges[1:])]}


def api_aliasing(query: dict) -> dict:
    x, fs_out, mode, tone = _aliasing_signal(query)
    naive = F.naive_resample(x, ALIAS_FS_IN, fs_out)
    clean = F.resample(x, ALIAS_FS_IN, fs_out)
    out = {"modo": mode, "fs_in": ALIAS_FS_IN, "fs_out": fs_out, "nyquist": fs_out / 2, "tono": tone}
    if mode == "tono":
        k = round(tone / fs_out)
        out["alias_previsto"] = abs(tone - k * fs_out)
        out["espectros"] = {"original": _spec_db(x, ALIAS_FS_IN), "sin": _spec_db(naive, fs_out),
                            "con": _spec_db(clean, fs_out)}
    else:
        def sg(y, fs):
            f, t, S = sps.spectrogram(y, fs=fs, nperseg=256, noverlap=128)
            S = 10 * np.log10(S + 1e-12)
            return {"db": np.round(_decimate_cols(S.T, 200), 1).tolist(), "fmax": fs / 2}
        out["espectrogramas"] = {"sin": sg(naive, fs_out), "con": sg(clean, fs_out)}
    return out


def api_aliasing_wav(query: dict) -> bytes:
    x, fs_out, _, _ = _aliasing_signal(query)
    variant = _q(query, "variante", "original", str)
    if variant == "sin":
        return write_wav_bytes(F.naive_resample(x, ALIAS_FS_IN, fs_out), fs_out)
    if variant == "con":
        return write_wav_bytes(F.resample(x, ALIAS_FS_IN, fs_out), fs_out)
    return write_wav_bytes(x, ALIAS_FS_IN)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def api_metricas() -> dict:
    return {
        "entrenamiento": _read_json(C.METRICS_PATH),
        "base_anterior": _read_json(C.LEGACY_DIR / "metricas_legacy.json"),
        "robustez": _read_json(C.REPORTS_DIR / "robustez.json"),
        "reales": _read_json(C.REPORTS_DIR / "eval_real.json"),
    }


def api_vivo(body: bytes, query: dict) -> dict:
    bundle = _bundle()
    sid = _q(query, "sesion", "", str)
    fs_in = int(_q(query, "fs", 48000))
    if not 8000 <= fs_in <= 192000:
        raise ValueError("Frecuencia de muestreo no soportada.")
    chunk = np.frombuffer(body, dtype="<f4").astype(np.float64)
    with LOCK:
        now = time.time()
        for k in [k for k, s in LIVE.items() if now - s.last_seen > 120]:
            del LIVE[k]
        sess = LIVE.get(sid)
        if sess is None:
            sess = LIVE[sid] = LiveSession(fs_in, bundle)
    return sess.push(chunk)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "Segmentador/2"

    def log_message(self, fmt, *args):  # consola mas limpia
        if "/api/vivo" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise ValueError("Archivo vacio o mayor a 200 MB.")
        return self.rfile.read(length)

    def _guard(self, fn):
        try:
            fn()
        except (ValueError, KeyError) as exc:
            self._json(400, {"error": str(exc).strip("'\"")})
        except RuntimeError as exc:
            self._json(503, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            self._json(500, {"error": f"Error interno: {exc}"})

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in STATIC:
            p = WEB_DIR / STATIC[url.path]
            return self._send(200, p.read_bytes(), MIME[p.suffix])
        routes = {
            "/api/estado": lambda: self._json(200, api_estado()),
            "/api/explicar": lambda: self._json(200, api_explicar(q)),
            "/api/audio": lambda: self._send(200, api_audio(q), "audio/wav"),
            "/api/filtros": lambda: self._json(200, api_filtros()),
            "/api/aliasing": lambda: self._json(200, api_aliasing(q)),
            "/api/aliasing.wav": lambda: self._send(200, api_aliasing_wav(q), "audio/wav"),
            "/api/metricas": lambda: self._json(200, api_metricas()),
        }
        if url.path in routes:
            return self._guard(routes[url.path])
        self._send(404, b"", "text/plain")

    def do_POST(self) -> None:
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/api/segmentar":
            name = unquote(self.headers.get("X-Nombre-Archivo", ""))
            return self._guard(lambda: self._json(200, api_segmentar(self._body(), q, name)))
        if url.path == "/api/vivo":
            return self._guard(lambda: self._json(200, api_vivo(self._body(), q)))
        if url.path == "/api/vivo/fin":
            with LOCK:
                LIVE.pop(_q(q, "sesion", "", str), None)
            return self._json(200, {"ok": True})
        self._send(404, b"", "text/plain")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interfaz web del segmentador.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--modelo", default=str(C.MODEL_BUNDLE_PATH))
    args = parser.parse_args()
    try:
        STATE["bundle"] = load_bundle(args.modelo)
        print(f"Modelo: {STATE['bundle']['descripcion']}")
    except FileNotFoundError as exc:
        STATE["bundle_error"] = str(exc)
        print(f"Aviso: {exc} La pestana de acondicionamiento y la demo de aliasing funcionan igual.")
    print(f"Abrir http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
