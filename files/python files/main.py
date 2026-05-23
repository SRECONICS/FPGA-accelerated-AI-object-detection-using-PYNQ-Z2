# from pynq.overlays.base import BaseOverlay
# from pynq.lib.video import VideoMode
# from yolo_inference import YOLODetector
# import cv2
# import numpy as np
# import time
# import threading

# # ==========================
# # HDMI SETUP
# # ==========================
# print("Loading base overlay...")
# base = BaseOverlay("base.bit")
# hdmi_out = base.video.hdmi_out
# mode = VideoMode(640, 480, 24)
# hdmi_out.configure(mode)
# hdmi_out.start()
# print("HDMI initialized")

# # ==========================
# # CAMERA SETUP
# # ==========================
# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
# if not cap.isOpened():
#     print("Camera NOT opened")
#     hdmi_out.stop()
#     exit()

# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# print("Camera working")

# # ==========================
# # YOLO SETUP
# # ==========================
# print("Loading YOLO...")
# yolo = YOLODetector()
# print("YOLO loaded")

# # ==========================
# # SHARED STATE
# # ==========================
# latest_frame          = None
# latest_detected_frame = None
# frame_lock            = threading.Lock()
# result_lock           = threading.Lock()
# running               = True

# # ==========================
# # YOLO THREAD (optimized)
# # - Uses 320x240 input (2x faster)
# # - Scales result back to 640x480
# # - Small sleep to reduce CPU pressure
# # ==========================
# def yolo_thread():
#     global latest_detected_frame, running
#     while running:
#         # Get latest frame
#         with frame_lock:
#             frame = latest_frame.copy() if latest_frame is not None else None

#         if frame is None:
#             time.sleep(0.05)
#             continue

#         # FIX 2: Resize to half before detection (2x faster)
#         small  = cv2.resize(frame, (320, 240))

#         # Run YOLO on smaller frame
#         result = yolo.detect(small)

#         # Scale result back to full display size
#         result = cv2.resize(result, (640, 480))

#         # Store result
#         with result_lock:
#             latest_detected_frame = result

#         # FIX 3: Small sleep to reduce CPU pressure
#         time.sleep(0.1)


# # Start YOLO thread
# t = threading.Thread(target=yolo_thread, daemon=True)
# t.start()
# print("YOLO thread started")

# # ==========================
# # MAIN DISPLAY LOOP
# # Smooth live video at 10-16 FPS
# # Boxes update from YOLO thread
# # ==========================
# frame_count = 0
# prev_time   = time.time()

# print("Starting pipeline - check your TV!")

# try:
#     while True:
#         # Grab camera frame
#         if not cap.grab():
#             continue
#         ret, frame = cap.retrieve()
#         if not ret or frame is None:
#             continue

#         frame = cv2.resize(frame, (640, 480))
#         frame_count += 1

#         # Update shared frame for YOLO thread
#         with frame_lock:
#             latest_frame = frame.copy()

#         # Use latest YOLO result if available, else raw frame
#         with result_lock:
#             display = latest_detected_frame.copy() \
#                       if latest_detected_frame is not None \
#                       else frame.copy()

#         # FPS
#         now       = time.time()
#         fps       = 1.0 / (now - prev_time) if prev_time else 0
#         prev_time = now

#         # Overlay info
#         cv2.putText(display, f"FPS: {fps:.1f}",
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                     1.0, (0, 255, 0), 2)

#         cv2.putText(display, f"Frame: {frame_count}",
#                     (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (0, 255, 255), 2)

#         cv2.putText(display,
#                     "PYNQ-Z2 FPGA | YOLOv3-Tiny Object Detection",
#                     (10, 460), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.55, (255, 255, 255), 1)

#         # BGR -> RGB for HDMI
#         display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
#         display_rgb = np.ascontiguousarray(display_rgb)

#         out_frame = hdmi_out.newframe()
#         np.copyto(out_frame, display_rgb)
#         hdmi_out.writeframe(out_frame)

#         # Terminal log every 10 frames
#         if frame_count % 10 == 0:
#             print(f"Frame {frame_count:04d} | FPS: {fps:.2f}")

# except KeyboardInterrupt:
#     print("\nStopped by user.")

# finally:
#     running = False
#     cap.release()
#     hdmi_out.stop()
#     print("Cleanup done.")

from pynq.overlays.base import BaseOverlay
from pynq.lib.video import VideoMode
from yolo_inference import YOLODetector
import cv2
import numpy as np
import time
import threading

# ==========================
# HDMI SETUP
# ==========================
print("Loading base overlay...")
base = BaseOverlay("base.bit")
hdmi_out = base.video.hdmi_out
mode = VideoMode(640, 480, 24)
hdmi_out.configure(mode)
hdmi_out.start()
print("HDMI initialized")

# ==========================
# CAMERA SETUP
# ==========================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Camera NOT opened")
    hdmi_out.stop()
    exit()

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Zoom out — use full camera FOV
cap.set(cv2.CAP_PROP_ZOOM, 0)       # minimum zoom
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # enable autofocus
print("Camera working")

# ==========================
# YOLO SETUP
# ==========================
print("Loading YOLO...")
yolo = YOLODetector()
print("YOLO loaded")

# ==========================
# SHARED STATE
# ==========================
latest_frame          = None
latest_detected_frame = None

frame_lock   = threading.Lock()
result_lock  = threading.Lock()
running      = True

# ==========================
# THREAD 1: CAMERA THREAD
# ==========================
def camera_thread():
    global latest_frame, running
    while running:
        if not cap.grab():
            continue
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue

        # Use full 640x480 — no cropping for wider FOV
        frame = cv2.resize(frame, (640, 480))

        with frame_lock:
            latest_frame = frame

# ==========================
# THREAD 2: YOLO THREAD
# ==========================
def yolo_thread():
    global latest_detected_frame, running
    while running:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.05)
            continue

        # Run on 320x240 for speed
        small  = cv2.resize(frame, (320, 240))
        result = yolo.detect(small)
        result = cv2.resize(result, (640, 480))

        with result_lock:
            latest_detected_frame = result

        time.sleep(0.2)

# ==========================
# THREAD 3: DISPLAY THREAD
# ==========================
def display_thread():
    global running
    frame_count = 0
    prev_time   = time.time()

    while running:
        with frame_lock:
            raw = latest_frame.copy() if latest_frame is not None else None

        if raw is None:
            time.sleep(0.01)
            continue

        with result_lock:
            display = latest_detected_frame.copy() \
                      if latest_detected_frame is not None \
                      else raw.copy()

        frame_count += 1

        now       = time.time()
        fps       = 1.0 / (now - prev_time) if prev_time else 0
        prev_time = now

        # Overlay text
        cv2.putText(display, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

        cv2.putText(display, f"Frame: {frame_count}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.putText(display,
                    "PYNQ-Z2 FPGA | YOLOv3-Tiny Object Detection",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

        # BGR -> RGB
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        display_rgb = np.ascontiguousarray(display_rgb)

        # Write to HDMI
        try:
            out_frame = hdmi_out.newframe()
            np.copyto(out_frame, display_rgb)
            hdmi_out.writeframe(out_frame)
        except Exception as e:
            print(f"HDMI error: {e}")
            continue

        if frame_count % 10 == 0:
            print(f"Frame {frame_count:04d} | FPS: {fps:.2f}")

# ==========================
# START ALL THREADS
# ==========================
t_camera  = threading.Thread(target=camera_thread,  daemon=True)
t_yolo    = threading.Thread(target=yolo_thread,     daemon=True)
t_display = threading.Thread(target=display_thread,  daemon=True)

t_camera.start()
t_yolo.start()
t_display.start()
print("YOLO thread started")
print("Starting pipeline - check your TV!")

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    running = False
    time.sleep(0.5)
    cap.release()
    hdmi_out.stop()
    print("Cleanup done.")
    
    
    