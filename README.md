# 🔬 Hardware-Accelerated AI Object Detection on PYNQ-Z2 FPGA

<!-- Platform -->
![Board](https://img.shields.io/badge/board-PYNQ--Z2-1a3a8f?style=flat-square&labelColor=162d6e)
![SoC](https://img.shields.io/badge/SoC-Zynq--7020-2251c4?style=flat-square&labelColor=162d6e)
![HLS](https://img.shields.io/badge/HLS-Vitis%202020.2-7c6cf8?style=flat-square&labelColor=2e2080)
![Bitstream](https://img.shields.io/badge/bitstream-3.9%20MB-6a5ae0?style=flat-square&labelColor=2e2080)
![AXI](https://img.shields.io/badge/arch-AXI--Lite%20%2B%20AXI4-555555?style=flat-square&labelColor=3a3a3a)

<!-- Model -->
![Model](https://img.shields.io/badge/model-YOLOv3--Tiny-0d7a55?style=flat-square&labelColor=0a5940)
![BlobSize](https://img.shields.io/badge/blob-160×160-0d7a55?style=flat-square&labelColor=0a5940)
![Dataset](https://img.shields.io/badge/dataset-COCO%2080cls-0d7a55?style=flat-square&labelColor=0a5940)
![Accel](https://img.shields.io/badge/accelerator-Conv2D%20·%2064×64%20·%20int32-b36b00?style=flat-square&labelColor=7a4a00)

<!-- Results -->
![FPS](https://img.shields.io/badge/display%20FPS-8%20–%2018-2e7019?style=flat-square&labelColor=214d12)
![Confidence](https://img.shields.io/badge/confidence-96%25-2e7019?style=flat-square&labelColor=214d12)
![Output](https://img.shields.io/badge/output-640×480%20HDMI-0d7a55?style=flat-square&labelColor=0a5940)
![Pipeline](https://img.shields.io/badge/pipeline-threaded%20·%20V4L2-b33528?style=flat-square&labelColor=822018)

<!-- Status -->
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)

> Real-time YOLOv3-Tiny object detection pipeline on a Xilinx Zynq-7020 FPGA board,
> with a custom HLS convolutional accelerator, live USB camera input, and HDMI TV output.

![Detection Demo](docs/demo.jpg) <!-- Replace with your actual demo photo -->

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Hardware Setup](#hardware-setup)
- [Architecture](#architecture)
- [HLS Accelerator (Vitis HLS)](#hls-accelerator-vitis-hls)
- [Vivado Block Design](#vivado-block-design)
- [PYNQ Python Pipeline](#pynq-python-pipeline)
- [Results](#results)
- [File Structure](#file-structure)
- [Setup & Usage](#setup--usage)
- [Known Issues & Future Work](#known-issues--future-work)
- [References](#references)

---

## Project Overview

This project implements a real-time AI object detection system on the **PYNQ-Z2** development board (Xilinx Zynq-7020). It combines:

- A **custom HLS Conv2D accelerator** synthesized in Vitis HLS 2020.2 and deployed to the FPGA fabric
- A **YOLOv3-Tiny inference pipeline** running on the ARM Cortex-A9 processor
- A **live video pipeline** from USB camera → object detection → HDMI TV display

The system detects objects (e.g., *person at 96% confidence*) in real time, displaying annotated bounding boxes on a connected LG TV at **8–18 FPS**.

---

## Hardware Setup

| Component | Details |
|-----------|---------|
| **Board** | PYNQ-Z2 (Xilinx Zynq-7020, ARM Cortex-A9 + FPGA fabric) |
| **Camera** | USB camera via V4L2 (`/dev/video0`) |
| **Display** | LG TV via HDMI OUT |
| **Development** | SSH + VS Code Remote from Windows PC |
| **Power** | 5V via micro-USB or barrel jack |

---

## Architecture

```
USB Camera (V4L2)
       │
       ▼
 camera_stream.py          ← grab()/retrieve() threading
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
 YOLO Thread (ARM)                    Display Thread (ARM)
 yolo_inference.py                    main.py
 YOLOv3-Tiny @ 160×160                640×480 BGR→RGB
       │                                      │
       └──────────────► Bounding Boxes ───────┘
                                      │
                                      ▼
                         HDMI OUT (BaseOverlay)
                         np.copyto → frame buffer
                                      │
                                      ▼
                               LG TV @ 8–18 FPS
```

The **HLS Conv2D accelerator** (`conv_accel`) runs on the FPGA fabric and is mapped to AXI-Lite at address `0x43C00000`. It handles 3×3 convolutions on 64×64 int32 arrays for accelerated layer processing.

---

## HLS Accelerator (Vitis HLS)

### Design: `conv_accel.cpp`

A pipelined 3×3 Conv2D kernel operating on 64×64 `int32` arrays.

**Key design decisions:**

- `#pragma HLS PIPELINE` placed on the **inner loop** (not outer) to fix II Violation
- All AXI-Lite ports bundled into a **single `control` bundle** to resolve dual-port conflict
- AXI Master (`m_axi_gmem`) for DMA data transfers

**Register Map (confirmed via PYNQ readback):**

| Register | Offset | Description |
|----------|--------|-------------|
| `ap_ctrl` | `0x00` | ap_start / ap_done / ap_idle |
| `input_addr` | `0x10` | Input buffer base address (64-bit) |
| `kernel_addr` | `0x1C` | Kernel buffer base address (64-bit) |
| `output_addr` | `0x28` | Output buffer base address (64-bit) |

### Vitis HLS Export Bug Fix

Vitis HLS 2020.2 has a known bug where `core_revision` in `run_ippack.tcl` exceeds the allowed integer size, causing IP export to fail.

**Fix:** A PowerShell file watcher script detects `run_ippack.tcl` creation and patches the value within **~100ms**:

```powershell
# Watches for run_ippack.tcl and patches core_revision before HLS reads it
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = "<your_hls_project_path>"
$watcher.Filter = "run_ippack.tcl"
# ... patches set core_revision to a valid small integer
```

> This was a critical unblocking step — without it, `conv_accel.zip` could not be exported.

---

## Vivado Block Design

### Block Diagram Summary

```
PS (ARM)
 ├── GP0 ──► AXI Interconnect (Control) ──► conv_accel_0 (s_axi_control)
 └── HP0 ◄── AXI Interconnect (Data)    ◄── conv_accel_0 (m_axi_gmem)
```

### Address Map

| IP Block | Base Address | Size |
|----------|-------------|------|
| `conv_accel_0` | `0x43C00000` | 64K |

### Key Configuration

- `C_M_AXI_GMEM_ADDR_WIDTH = 64` — required for 64-bit address compatibility
- IP version upgraded from **v1.0 → v2.0** to resolve address width conflict with cached v1.0 in block design
- Bitstream: `design_1_wrapper.bit` (3.9 MB)

---

## PYNQ Python Pipeline

### Overlay Loading

```python
from pynq import Overlay, allocate
import numpy as np

ol = Overlay("cnn_accel.bit")   # .hwh filename must match .bit filename exactly
accel = ol.conv_accel_0
```

### Buffer Allocation & Cache Sync

```python
input_buf  = allocate(shape=(64*64,), dtype=np.int32)
kernel_buf = allocate(shape=(9,),     dtype=np.int32)
output_buf = allocate(shape=(64*64,), dtype=np.int32)

# Fill buffers ...

input_buf.flush()          # Write-back to DRAM before FPGA reads
kernel_buf.flush()

accel.register_map.input_addr  = input_buf.physical_address
accel.register_map.kernel_addr = kernel_buf.physical_address
accel.register_map.output_addr = output_buf.physical_address
accel.register_map.ap_ctrl.ap_start = 1

while not accel.register_map.ap_ctrl.ap_done:
    pass

output_buf.invalidate()    # Invalidate cache before CPU reads FPGA result
result = np.array(output_buf)
```

### YOLO Inference

```python
# yolo_inference.py
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (160, 160), swapRB=True)
net.setInput(blob)
outputs = net.forward(output_layers)
```

Blob size of **160×160** balances detection accuracy and ARM CPU speed on the Zynq.

### Display Pipeline

```python
# main.py — threaded: display thread + YOLO thread
hdmi_out = base.video.hdmi_out
hdmi_out.configure(VideoMode(640, 480, 24))
hdmi_out.start()

frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
np.copyto(hdmi_out.newframe(), frame_rgb)
hdmi_out.writeframe(frame_rgb)
```

---

## Results

| Metric | Value |
|--------|-------|
| **Display FPS** | 8–18 FPS (live, smooth) |
| **Detection confidence** | Person at **96%** ✅ |
| **YOLO blob size** | 160×160 |
| **Output resolution** | 640×480 on HDMI TV |
| **Accelerator registers** | Readback verified ✅ |

> **Note:** The FPGA Conv2D accelerator was successfully synthesized and loaded.  
> Due to a `C_M_AXI_GMEM_ADDR_WIDTH=64` incompatibility on the 32-bit ARM memory bus,  
> the YOLO inference runs on ARM CPU as a fallback. FPGA acceleration fix is in progress (see Future Work).

---

## File Structure

```
pynq-yolo-fpga/
│
├── pynq/                        # Files deployed to PYNQ board
│   ├── main.py                  # Threaded display + YOLO pipeline
│   ├── yolo_inference.py        # YOLOv3-Tiny OpenCV DNN wrapper
│   ├── camera_stream.py         # V4L2 USB camera (grab/retrieve)
│   ├── cnn_accel.bit            # FPGA bitstream (Vivado)
│   ├── cnn_accel.hwh            # Hardware handoff file (must match .bit name)
│   ├── yolov3-tiny.weights      # Pre-trained YOLO weights (not tracked by git)
│   ├── yolov3-tiny.cfg          # YOLOv3-Tiny model config
│   └── coco.names               # COCO class labels (80 classes)
│
├── hls/                         # Vitis HLS project
│   ├── conv_accel.cpp           # HLS Conv2D accelerator source
│   ├── conv_accel.h             # Header
│   └── patch_run_ippack.ps1     # PowerShell watcher — fixes HLS export bug
│
├── vivado/                      # Vivado project files
│   └── design_1_wrapper.bit     # Final exported bitstream
│
├── docs/
│   ├── demo.jpg                 # Live detection photo
│   └── block_design.png         # Vivado block design screenshot
│
└── README.md
```

> ⚠️ `yolov3-tiny.weights` is not included in this repo due to file size.  
> Download from: https://pjreddie.com/darknet/yolo/

---

## Setup & Usage

### 1. Prerequisites on PYNQ-Z2

```bash
# Stop memory-heavy services to avoid OOM
sudo systemctl stop jupyter
sudo systemctl stop smbd
sudo systemctl stop pulseaudio
```

### 2. Copy files to PYNQ board

```bash
scp pynq/* xilinx@<pynq-ip>:~/
```

### 3. Download YOLO weights

```bash
# On PYNQ board
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

### 4. Run the pipeline

```bash
ssh xilinx@<pynq-ip>
sudo -E python3 main.py
```

> Use `sudo -E` (not Jupyter) to avoid SSH session drops and permission issues with HDMI/V4L2.

### 5. Expected output

- Live video appears on HDMI-connected TV at 640×480
- Bounding boxes drawn around detected objects with class labels and confidence
- Terminal prints FPS stats

---

## Key Issues Solved

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| SSH dropping mid-run | Jupyter process manager killing orphan sessions | Switched to `sudo -E python3` directly |
| `registers: {}` empty | `.hwh` filename didn't match `.bit` filename | Renamed both files to match exactly |
| FPGA hanging on `ap_start` | `C_M_AXI_GMEM_ADDR_WIDTH=64` on 32-bit ARM bus | Identified mismatch; CPU fallback used |
| Vitis HLS export crash | `core_revision` integer overflow bug in 2020.2 | PowerShell watcher patches `run_ippack.tcl` |
| GStreamer camera warning | Default OpenCV backend | Explicitly set `cv2.CAP_V4L2` backend |
| Out-of-memory crashes | Jupyter + samba + pulseaudio consuming RAM | Stopped all non-essential services |
| IP version conflict | Old v1.0 cached in Vivado block design | Upgraded IP to v2.0, deleted v1.0 |
| HLS II Violation | `PIPELINE` pragma on outer loop | Moved pragma to innermost loop |
| Dual AXI-Lite ports | Separate bundles per argument | Bundled all ports to single `control` bundle |

---

## Known Issues & Future Work

- [ ] **Fix FPGA AXI address width** — Rebuild with `C_M_AXI_GMEM_ADDR_WIDTH=32` or upgrade to **Vitis HLS 2021.1+** which handles this correctly
- [ ] **Test AXI Stream + DMA architecture** — Replace memory-mapped AXI with streaming interface for higher throughput
- [ ] **Increase detection resolution** — Blob size 320×320 for better small-object detection (ARM permitting)
- [ ] **Add project photos** to `docs/` folder
- [ ] **Benchmark FPGA vs CPU** convolution latency once AXI issue is resolved

---

## References

- [PYNQ Documentation](http://pynq.readthedocs.io/)
- [YOLOv3 Paper — Redmon & Farhadi](https://arxiv.org/abs/1804.02767)
- [Vitis HLS User Guide (UG902)](https://docs.xilinx.com/r/en-US/ug902-vivado-high-level-synthesis)
- [Zynq-7000 TRM (UG585)](https://docs.xilinx.com/r/en-US/ug585-zynq-7000-TRM)
- [OpenCV DNN Module](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ on a PYNQ-Z2 — where ARM meets FPGA meets AI.*
