# Offline tuning sweep over the clips recorded by record_clips.py.
#
# Stage 1: capture settings (fps, resolution, compression, fps bookkeeping),
#          all decoded with Chaplin's default decode weights.
# Stage 2: decode weights (language model weight, CTC weight) on the best capture.
# Stage 3: top-10 hypotheses on the best overall setting, to see how often the
#          right sentence is somewhere in the beam (what an LLM corrector can use).
#
# Run from the chaplin repo root:
#   uv run --with-requirements requirements.txt --python 3.12 tune/sweep.py
import glob
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from espnet.asr.asr_utils import parse_hypothesis  # noqa: E402
from pipelines.data.transforms import VideoTransform  # noqa: E402
from pipelines.model import get_beam_search_decoder  # noqa: E402
from pipelines.pipeline import InferencePipeline  # noqa: E402

CONFIG = "./configs/LRS3_V_WER19.1.ini"
LM_PATH = "benchmarks/LRS3/language_models/lm_en_subword/model.pth"
LM_CONF = "benchmarks/LRS3/language_models/lm_en_subword/model.json"
MODEL_FPS = 25
CLIP_DIR = os.environ.get("CLIP_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips"))
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

# name: (capture fps, width, height, jpeg quality or None, tell the model the true fps)
CAPTURES = {
    "chaplin_default": (16, 213, 120, 25, False),  # what Chaplin does today
    "fps_bookkeeping_fixed": (16, 213, 120, 25, True),
    "25fps_low_res": (25, 213, 120, 25, True),
    "25fps_640": (25, 640, 360, None, True),
    "25fps_1280": (25, 1280, 720, None, True),
}
DEFAULT_DECODE = (0.1, 0.3)  # (ctc_weight, lm_weight) from Chaplin's config
DECODE_GRID = [(ctc, lm) for ctc in (0.1, 0.3) for lm in (0.0, 0.1, 0.3, 0.5, 0.7)]


def word_errors(reference, hypothesis):
    ref = reference.lower().split()
    hyp = hypothesis.lower().replace("colour", "color").split()
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1], len(ref)


def load_clips():
    clips = []
    for meta_path in sorted(glob.glob(os.path.join(CLIP_DIR, "*.json"))):
        meta = json.load(open(meta_path))
        cap = cv2.VideoCapture(meta_path.replace(".json", ".avi"))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        n = min(len(frames), len(meta["timestamps"]))
        clips.append((os.path.basename(meta_path)[:2], meta["sentence"], frames[:n],
                      np.array(meta["timestamps"][:n]) - meta["timestamps"][0]))
    return clips


def emulate_capture(frames, timestamps, fps, width, height, jpeg_quality, path):
    # pick the camera frame nearest each tick of the target frame rate
    ticks = np.arange(0, timestamps[-1], 1.0 / fps)
    picked = [frames[int(np.argmin(np.abs(timestamps - t)))] for t in ticks]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), False)
    for frame in picked:
        gray = cv2.cvtColor(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        if jpeg_quality is not None:
            _, buffer = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            gray = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        writer.write(gray)
    writer.release()


def encode(pipeline, path, speed_rate):
    landmarks = pipeline.landmarks_detector(path)
    video = pipeline.dataloader.load_video(path)
    video = torch.tensor(pipeline.dataloader.video_process(video, landmarks))
    data = VideoTransform(speed_rate=speed_rate)(video)
    with torch.no_grad():
        return pipeline.model.model.encode(data.to(pipeline.model.device))


def decode(decoder, token_list, enc_feats, nbest=1):
    with torch.no_grad():
        hyps = decoder(enc_feats)
    texts = []
    for h in hyps[:nbest]:
        text = parse_hypothesis(h.asdict(), token_list)[0]
        texts.append(text.replace("▁", " ").replace("<eos>", "").strip())
    return texts


def main():
    clips = load_clips()
    print(f"{len(clips)} clips loaded")
    pipeline = InferencePipeline(CONFIG, device=torch.device("cpu"), detector="mediapipe", face_track=True)
    model = pipeline.model
    decoders = {}

    def decoder_for(ctc, lm):
        if (ctc, lm) not in decoders:
            decoders[(ctc, lm)] = get_beam_search_decoder(
                model.model, model.token_list, LM_PATH, LM_CONF, 0.0, ctc, lm, 40).eval()
        return decoders[(ctc, lm)]

    results = {"capture": {}, "decode": {}, "nbest": []}
    tmp_dir = tempfile.mkdtemp()

    def run_capture(name):
        fps, width, height, quality, true_fps = CAPTURES[name]
        feats = {}
        for clip_id, sentence, frames, timestamps in clips:
            path = os.path.join(tmp_dir, f"{name}_{clip_id}.mp4")
            emulate_capture(frames, timestamps, fps, width, height, quality, path)
            try:
                feats[clip_id] = encode(pipeline, path, fps / MODEL_FPS if true_fps else 1)
            except Exception as error:  # face not found, crop out of bounds, ...
                print(f"  {name} clip {clip_id}: preprocessing failed ({error})")
                feats[clip_id] = None
        return feats

    def score(feats, ctc, lm, label):
        errors = words = 0
        rows = []
        for clip_id, sentence, _, _ in clips:
            text = decode(decoder_for(ctc, lm), model.token_list, feats[clip_id])[0] if feats[clip_id] is not None else ""
            e, n = word_errors(sentence, text)
            errors, words = errors + e, words + n
            rows.append({"clip": clip_id, "sentence": sentence, "output": text, "errors": e, "words": n})
        wer = errors / words
        print(f"{label:45s} WER {wer:6.1%}  ({errors}/{words})", flush=True)
        return {"wer": wer, "rows": rows}

    print("\nStage 1: capture settings (ctc=0.1, lm=0.3)")
    capture_feats = {}
    for name in CAPTURES:
        capture_feats[name] = run_capture(name)
        results["capture"][name] = score(capture_feats[name], *DEFAULT_DECODE, name)
    best_capture = min(results["capture"], key=lambda k: results["capture"][k]["wer"])

    print(f"\nStage 2: decode weights on {best_capture}")
    for ctc, lm in DECODE_GRID:
        results["decode"][f"ctc={ctc} lm={lm}"] = score(capture_feats[best_capture], ctc, lm, f"ctc={ctc} lm={lm}")
    best_decode = min(results["decode"], key=lambda k: results["decode"][k]["wer"])
    ctc, lm = (float(part.split("=")[1]) for part in best_decode.split())

    print(f"\nStage 3: top-10 hypotheses for {best_capture}, {best_decode}")
    oracle_errors = top1_errors = words = 0
    for clip_id, sentence, _, _ in clips:
        feats = capture_feats[best_capture][clip_id]
        hyps = decode(decoder_for(ctc, lm), model.token_list, feats, nbest=10) if feats is not None else [""]
        per_hyp = [word_errors(sentence, h)[0] for h in hyps]
        n = len(sentence.split())
        top1_errors, oracle_errors, words = top1_errors + per_hyp[0], oracle_errors + min(per_hyp), words + n
        results["nbest"].append({"clip": clip_id, "sentence": sentence, "hypotheses": hyps})
    print(f"top-1 WER {top1_errors / words:.1%}   best-of-10 (oracle) WER {oracle_errors / words:.1%}")

    results["best"] = {"capture": best_capture, "decode": best_decode}
    json.dump(results, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nsaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
