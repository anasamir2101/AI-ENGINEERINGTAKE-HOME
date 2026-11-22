# Hardware

The full 50 files were run on a rented RunPod Pod (On-Demand).

- GPU: NVIDIA GeForce RTX 4090, 24 GB memory (24,564 MiB), driver 595.112.07
- CPU: AMD EPYC 7B13 (64-core host CPU). The Pod was limited to about 18 vCPUs
  (CPU quota 17.85), although the container shows 128 threads.
- RAM: about 77 GB (Pod memory limit). The host has about 503 GB, shared.
- Disk: NVMe SSD (80 GB container disk)
- OS: Ubuntu 24.04.5 LTS. Python 3.11 in a virtual environment.

Run: started 07:25:32 UTC on 21 Nov 2026, finished about 08:57 UTC.
Total time from start to finish: 5,532.4 seconds (about 1 h 32 min), 1,650 videos,
12,981,615 video frames.