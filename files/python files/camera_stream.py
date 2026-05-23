import cv2

class CameraStream:

    def __init__(self, cam_id=0, width=320, height=240):

        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError("ERROR: Camera not detected. Check 'ls /dev/video*'")

        # Reduce buffer to avoid stale frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Set FPS explicitly to avoid GStreamer timing issues
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Verify resolution was accepted
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {actual_w}x{actual_h}")

    def get_frame(self):

        # Grab and retrieve separately for better reliability
        if not self.cap.grab():
            print("WARNING: Failed to grab frame.")
            return None

        ret, frame = self.cap.retrieve()

        if not ret or frame is None:
            print("WARNING: Failed to retrieve frame.")
            return None

        return frame

    def release(self):

        if self.cap.isOpened():
            self.cap.release()
            print("Camera released.")