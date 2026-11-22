# Video indexing with YOLO11l

For every second of every video in a folder of FineVideo Parquet files, this finds
person, car, dog, laptop and cell phone with YOLO11l and counts the people.

Tested on: RTX 4090, Python 3.11, NVIDIA driver 595 (CUDA 13.2 capable), Ubuntu 24.04.
Needs: Ubuntu, Python 3.11, an NVIDIA GPU with a driver that can run the CUDA 12.6 build of PyTorch.

## Install

```
sudo apt-get install -y libgl1 libglib2.0-0
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Preparation (needs internet, run once from the repo folder)

This downloads the official `yolo11l.pt` into the repo folder, next to `run.py`:

```
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

## Run (no internet needed)

```
python run.py --input <folder of .parquet files> --output results.jsonl
```

Optional: add `--resume` to continue an interrupted run; it keeps the existing output
and skips videos that are already finished. Without it the output file is overwritten.

## Notes

- Every video gets lines for each second, or one error line if it could not be processed.
- 6 evenly spaced frames per second are checked, in FP16, on the GPU.
- Result for the 50 training files (1,650 videos, 128 hours) is in `results.jsonl.gz`.
  Full run time on the RTX 4090: 5,532 seconds (about 1 h 32 min). See HARDWARE.md.