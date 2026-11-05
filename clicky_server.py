# Local lip-reading server for Clicky's silent mode.
#
# Keeps the camera open, records while Clicky's push-to-talk is held, and
# returns the top-10 lip-reading hypotheses. Uses the settings picked by
# tune/sweep.py: 1280x720 frames resampled to the model's 25 fps,
# ctc_weight=0.1, lm_weight=0.1.
#
#   POST /start  -> begin recording
#   POST /stop   -> stop, decode, return {"hypotheses": [...], "seconds": ...}
#   GET  /health -> {"ok": true}
#
# Run from the chaplin repo root:
#   uv run --with-requirements requirements.txt --python 3.12 clicky_server.py
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

from pipelines.model import get_beam_search_decoder
from pipelines.pipeline import InferencePipeline
from tune.sweep import CONFIG, LM_CONF, LM_PATH, MODEL_FPS, decode, emulate_capture, encode

PORT = 8765
NBEST = 10
CTC_WEIGHT, LM_WEIGHT = 0.1, 0.1
MIN_SECONDS = 0.8


class LipReader:
    def __init__(self):
        self.pipeline = InferencePipeline(CONFIG, device=torch.device("cpu"), detector="mediapipe", face_track=True)
        self.decoder = get_beam_search_decoder(
            self.pipeline.model.model, self.pipeline.model.token_list,
            LM_PATH, LM_CONF, 0.0, CTC_WEIGHT, LM_WEIGHT, 40).eval()
        self.lock = threading.Lock()
        self.recording = False
        self.frames, self.timestamps = [], []

        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        # read continuously so the camera never serves stale buffered frames
        while True:
            ok, frame = self.camera.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                if self.recording:
                    self.frames.append(frame)
                    self.timestamps.append(time.time())

    def start(self):
        with self.lock:
            self.frames, self.timestamps = [], []
            self.recording = True
        print("● recording", flush=True)

    def stop(self):
        with self.lock:
            self.recording = False
            frames, timestamps = self.frames, np.array(self.timestamps)
        if len(frames) < 2 or timestamps[-1] - timestamps[0] < MIN_SECONDS:
            print("■ too short, ignored", flush=True)
            return []

        started = time.time()
        path = os.path.join(tempfile.mkdtemp(), "clip.mp4")
        emulate_capture(frames, timestamps - timestamps[0], MODEL_FPS, 1280, 720, None, path)
        try:
            hypotheses = decode(self.decoder, self.pipeline.model.token_list,
                                encode(self.pipeline, path, 1), nbest=NBEST)
        except Exception as error:  # usually: no face found
            print(f"■ decode failed: {error}", flush=True)
            return []
        finally:
            os.remove(path)
        print(f"■ {timestamps[-1] - timestamps[0]:.1f}s clip decoded in {time.time() - started:.1f}s", flush=True)
        for rank, text in enumerate(hypotheses, 1):
            print(f"   {rank:2d}. {text}", flush=True)
        return hypotheses


def make_handler(reader):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._reply({"ok": True} if self.path == "/health" else {"error": "not found"},
                        200 if self.path == "/health" else 404)

        def do_POST(self):
            if self.path == "/start":
                reader.start()
                self._reply({"ok": True})
            elif self.path == "/stop":
                started = time.time()
                hypotheses = reader.stop()
                self._reply({"hypotheses": hypotheses, "seconds": round(time.time() - started, 2)})
            else:
                self._reply({"error": "not found"}, 404)

        def log_message(self, *args):
            pass

    return Handler


if __name__ == "__main__":
    reader = LipReader()
    print(f"\nlip-reading server ready on http://localhost:{PORT}\n", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(reader)).serve_forever()
