import argparse, glob, json, math, os, queue, tempfile, threading, time
from collections import defaultdict, deque

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from ultralytics import YOLO

CLASS_IDS = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
FRAMES_PER_SEC = 6   # evenly spaced frames checked in each second of video
AHEAD = 3            # videos decoded in the background at the same time
BATCH = 32           # frames per model call
QUEUE_BATCHES = 3    # decoded batches waiting per video (limits memory use)
EPS = 1e-6           # absorbs floating point noise at second / bucket boundaries
STALL_TIMEOUT = 300.0   # seconds without any decoded frames from a video before it is given up on
MAX_JUMP = 1200.0    # a timestamp jump bigger than this (seconds) between two frames is treated as corrupt
BASE_KW = dict(imgsz=640, conf=0.4, iou=0.7, classes=list(CLASS_IDS), verbose=False)


def load_model():
    """Loads YOLO11l and picks the fastest precision option this Ultralytics version accepts."""
    weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11l.pt")
    if not os.path.exists(weights):
        weights = "yolo11l.pt"
    model = YOLO(weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA GPU available, running on the CPU. This will be very slow.", flush=True)
    else:
        print(f"using GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}", flush=True)
    model.to(device)
    blank = [np.zeros((360, 640, 3), dtype=np.uint8)]
    options = [dict(quantize=16), dict(half=True), {}] if device == "cuda" else [{}]
    for extra in options:
        try:
            model.predict(blank, **BASE_KW, **extra)
            return model, {**BASE_KW, **extra}
        except Exception as e:
            print(f"predict option {extra} not usable: {str(e)[:100]}", flush=True)
    raise RuntimeError("could not run the model")


class Job:
    """Decodes one video in a background thread and hands frame batches to the main thread."""

    def __init__(self, vid, data):
        self.vid = vid
        self.data = data
        self.q = queue.Queue(maxsize=QUEUE_BATCHES)
        self.cancel = threading.Event()
        self.thread = threading.Thread(target=self.decode, daemon=True)

    def put(self, item):
        # wait for room in the queue, but give up if the main thread cancelled this job
        while not self.cancel.is_set():
            try:
                self.q.put(item, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False

    def decode(self):
        cap = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                tmp.write(self.data)
                tmp.flush()
                self.data = None   # free the video bytes
                cap = cv2.VideoCapture(tmp.name)
                if not cap.isOpened():
                    raise RuntimeError("could not decode video")
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps != fps or fps <= 0 or fps > 1000:
                    fps = 25.0   # only used as a fallback step if timestamps are unusable
                step = 1.0 / fps
                k = FRAMES_PER_SEC
                frames, secs, i = [], [], 0
                pos0, prev_t, prev_b = None, None, None
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # timestamp of this frame in seconds, counted from the first decoded frame
                    pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if pos0 is None:
                        pos0 = pos
                    t = (pos - pos0) / 1000.0
                    if prev_t is None:
                        t = 0.0
                    elif not (prev_t < t <= prev_t + MAX_JUMP):
                        # timestamps must increase and not jump absurdly far (this also catches NaN);
                        # otherwise fall back to counting frames
                        t = prev_t + step
                    prev_t = t
                    b = math.floor(t * k + EPS)
                    if b != prev_b:         # first frame of each 1/k second slot
                        frames.append(frame)
                        secs.append(int(t + EPS))
                    prev_b = b
                    i += 1
                    if len(frames) == BATCH:
                        if not self.put(("batch", secs, frames)):
                            return
                        frames, secs = [], []
                if frames and not self.put(("batch", secs, frames)):
                    return
                self.put(("done", i))
        except Exception as e:
            self.put(("error", str(e)[:200]))
        finally:
            if cap is not None:
                cap.release()


def build_lines(vid, per_sec):
    lines = []
    if not per_sec:
        return lines
    for sec in range(max(per_sec) + 1):
        items = per_sec.get(sec, [])
        n = len(items)
        if n == 0:
            lines.append({"video": vid, "second": sec, "objects": [], "people": 0})
            continue
        objs = [o for o in CLASS_IDS.values() if sum(o in s for s, _ in items) * 2 >= n]
        counts = sorted(c for _, c in items)
        people = counts[n // 2] if n % 2 else (counts[n // 2 - 1] + counts[n // 2]) // 2
        lines.append({"video": vid, "second": sec, "objects": objs, "people": people})
    return lines


def consume(job, model, predict_kw):
    """Runs the model on the decoded batches of one video. Returns (lines, frames, error)."""
    per_sec = defaultdict(list)
    n_total, err = 0, None
    last = time.time()   # when this video last made progress
    try:
        while True:
            try:
                msg = job.q.get(timeout=1)
            except queue.Empty:
                if not job.thread.is_alive() and job.q.empty():
                    err = "decoder stopped unexpectedly"
                    break
                if time.time() - last > STALL_TIMEOUT:
                    err = f"decoder stalled for more than {int(STALL_TIMEOUT)}s, video skipped"
                    break
                continue
            kind = msg[0]
            if kind == "batch":
                results = model.predict(msg[2], **predict_kw)
                for sec, r in zip(msg[1], results):
                    cls = r.boxes.cls.int().tolist()
                    per_sec[sec].append(({CLASS_IDS[c] for c in cls}, cls.count(0)))
                last = time.time()
            elif kind == "done":
                n_total = msg[1]
                break
            else:
                err = msg[1]
                break
        if err is None and n_total == 0:
            err = "no frames decoded"
    except Exception as e:
        err = str(e)[:200]
    finally:
        job.cancel.set()
    return build_lines(job.vid, per_sec), n_total, err


def load_resume(path):
    """Prepares an existing output file for --resume and returns the set of finished videos.

    A crash can leave a half-written last line and an incomplete last video, so the file is
    rewritten without that line and without any lines of the last video (it is processed again).
    """
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append((json.loads(line)["video"], line if line.endswith("\n") else line + "\n"))
            except Exception:
                pass   # half-written line
    if rows:
        last = rows[-1][0]
        rows = [(v, l) for v, l in rows if v != last]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(l for _, l in rows)
    os.replace(tmp, path)
    return {v for v, _ in rows}


def find_files(path):
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))


def iter_videos(files, done):
    """Yields (video id, mp4 bytes, error) for every video that is not already done."""
    for fp in files:
        name = os.path.basename(fp)
        try:
            pf = pq.ParquetFile(fp)
            total = pf.metadata.num_rows
            batches = pf.iter_batches(batch_size=1, columns=["mp4"])
        except Exception as e:
            print(f"{name}: could not open file ({e})", flush=True)
            continue
        for row in range(total):
            vid = f"{name}:{row}"
            try:
                batch = next(batches)
            except Exception as e:
                # reading broke here: report this video and the rest of the file
                for r in range(row, total):
                    if f"{name}:{r}" not in done:
                        yield f"{name}:{r}", None, f"could not read video ({str(e)[:100]})"
                break
            if vid in done:
                continue
            try:
                data = batch.column(0)[0].as_py()
                if not data:
                    raise ValueError("no video data")
            except Exception as e:
                yield vid, None, str(e)[:200]
                continue
            yield vid, data, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of .parquet files (or one file)")
    ap.add_argument("--output", required=True, help="results .jsonl file (overwritten)")
    ap.add_argument("--resume", action="store_true",
                    help="keep the existing output file and skip videos already in it")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    done = set()
    if args.resume and os.path.exists(args.output):
        done = load_resume(args.output)

    model, predict_kw = load_model()
    files = find_files(args.input)
    print(f"{len(files)} parquet files, {len(done)} videos already done", flush=True)

    videos = iter_videos(files, done)
    active = deque()
    exhausted = False
    t_start = time.time()
    total_frames = 0

    with open(args.output, "a" if args.resume else "w") as out:
        while True:
            # keep AHEAD videos decoding in the background
            while not exhausted and len(active) < AHEAD:
                try:
                    vid, data, err = next(videos)
                except StopIteration:
                    exhausted = True
                    break
                if err is not None:
                    out.write(json.dumps({"video": vid, "error": err}) + "\n")
                    out.flush()
                    print(f"{vid}: error ({err})", flush=True)
                    continue
                job = Job(vid, data)
                data = None
                job.thread.start()
                active.append(job)
            if not active:
                break

            job = active.popleft()
            t0 = time.time()
            lines, n, err = consume(job, model, predict_kw)
            for l in lines:
                out.write(json.dumps(l) + "\n")
            if err is not None:
                out.write(json.dumps({"video": job.vid, "error": err}) + "\n")
            out.flush()
            total_frames += n
            print(f"{job.vid}: {n} frames, {time.time() - t0:.1f}s" + (f", error: {err}" if err else ""),
                  flush=True)

    total = time.time() - t_start
    print(f"DONE. total time {total:.1f}s, video frames {total_frames}")


if __name__ == "__main__":
    main()