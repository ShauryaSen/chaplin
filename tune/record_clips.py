# Records one high-quality clip per test sentence, so capture and decode
# settings can be compared offline on the exact same mouthing.
#
# Controls (focus the camera window):
#   SPACE  start / stop recording the current sentence
#   r      go back and redo the previous sentence
#   n      skip to the next sentence
#   q      quit
import json
import os
import time

import cv2

SENTENCES = [
    # general clicky-style questions (same as the first test)
    "how do i commit in xcode",
    "where is the color inspector",
    "what does this error mean",
    "how do i make a new folder",
    "where do i click to export this video",
    "which button do i press to save this file",
    "can you show me where the settings are",
    "what is this panel on the right side",
    "what is html and how does it work",
    "explain what this function is doing",
    "how do i color grade this clip in final cut",
    "why is my code not running",
    # onshape demo candidates
    "how do i start a new sketch",
    "where is the extrude tool",
    "how do i add a fillet to this edge",
    "how do i make this hole bigger",
    "where do i change the units",
    "how do i mirror this part",
    "what does this constraint do",
    "how do i export this as an stl",
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera: {width}x{height} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")

    index, recording, writer, timestamps = 0, False, None, []
    while index < len(SENTENCES):
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time()
        clip_name = f"{index + 1:02d}"

        if recording:
            writer.write(frame)
            timestamps.append(now)

        # overlay only on the preview, never on the saved clip
        preview = cv2.flip(frame, 1)
        status = "REC" if recording else "SPACE to record"
        cv2.putText(preview, f"{index + 1}/{len(SENTENCES)}: {SENTENCES[index]}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(preview, status, (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255) if recording else (255, 255, 255), 2)
        cv2.imshow("record clips", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and not recording:
            writer = cv2.VideoWriter(os.path.join(OUT_DIR, clip_name + ".avi"),
                                     cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height))
            timestamps, recording = [], True
        elif key == ord(" ") and recording:
            writer.release()
            recording = False
            with open(os.path.join(OUT_DIR, clip_name + ".json"), "w") as f:
                json.dump({"sentence": SENTENCES[index], "timestamps": timestamps}, f)
            duration = timestamps[-1] - timestamps[0] if timestamps else 0
            print(f"saved {clip_name}: {len(timestamps)} frames, {duration:.1f}s, "
                  f"{len(timestamps) / max(duration, 1e-6):.1f} fps  ({SENTENCES[index]})")
            index += 1
        elif key == ord("r") and not recording and index > 0:
            index -= 1
            print(f"redo {index + 1}: {SENTENCES[index]}")
        elif key == ord("n") and not recording:
            index += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"done, clips in {OUT_DIR}")


if __name__ == "__main__":
    main()
