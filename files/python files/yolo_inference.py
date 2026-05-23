# import cv2
# import numpy as np

# class YOLODetector:

#     def __init__(self):

#         self.net = cv2.dnn.readNet(
#             "yolov3-tiny.weights",
#             "yolov3-tiny.cfg"
#         )

#         with open("coco.names", "r") as f:
#             self.classes = [line.strip() for line in f.readlines()]

#         self.layer_names = self.net.getLayerNames()
#         layer_ids = self.net.getUnconnectedOutLayers()

#         # Handle both OpenCV formats
#         try:
#             layer_ids = layer_ids.flatten()
#         except:
#             pass

#         self.output_layers = [self.layer_names[i - 1] for i in layer_ids]

#     def detect(self, frame):

#         height, width, _ = frame.shape

#         blob = cv2.dnn.blobFromImage(
#             frame,
#             1/255.0,
#             (160, 160),   # ← change from 224 to 160
#             swapRB=True,
#             crop=False
#         )

#         self.net.setInput(blob)
#         outputs = self.net.forward(self.output_layers)

#         boxes = []
#         confidences = []
#         class_ids = []

#         for output in outputs:
#             for detection in output:

#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]

#                 if confidence > 0.5:

#                     center_x = int(detection[0] * width)
#                     center_y = int(detection[1] * height)
#                     w = int(detection[2] * width)
#                     h = int(detection[3] * height)

#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)

#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)

#         indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#         if len(indexes) > 0:
#             for i in indexes.flatten():

#                 x, y, w, h = boxes[i]
#                 label = self.classes[class_ids[i]]
#                 confidence = confidences[i]

#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(
#                     frame,
#                     f"{label} {confidence:.2f}",
#                     (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (0, 255, 0),
#                     2
#                 )

#         return frame

import cv2
import numpy as np

class YOLODetector:

    def __init__(self):

        self.net = cv2.dnn.readNet(
            "yolov3-tiny.weights",
            "yolov3-tiny.cfg"
        )

        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.layer_names = self.net.getLayerNames()
        layer_ids = self.net.getUnconnectedOutLayers()

        try:
            layer_ids = layer_ids.flatten()
        except:
            pass

        self.output_layers = [self.layer_names[i - 1] for i in layer_ids]

    def detect(self, frame):

        height, width, _ = frame.shape

        # Increased blob size for better accuracy (224 instead of 160)
        blob = cv2.dnn.blobFromImage(
            frame,
            1/255.0,
            (224, 224),       # ← increased from 160 to 224
            swapRB=True,
            crop=False
        )

        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes       = []
        confidences = []
        class_ids   = []

        for output in outputs:
            for detection in output:

                scores    = detection[5:]
                class_id  = np.argmax(scores)
                confidence = scores[class_id]

                # Lowered threshold from 0.5 to 0.3 for wider detection
                if confidence > 0.3:

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():

                x, y, w, h = boxes[i]
                label      = self.classes[class_ids[i]]
                confidence = confidences[i]

                # Different colors for different classes
                color = (0, 255, 0)   # green default
                if label == "person":
                    color = (0, 255, 0)    # green
                elif label == "cat":
                    color = (0, 165, 255)  # orange
                elif label == "dog":
                    color = (0, 0, 255)    # red
                elif label == "laptop":
                    color = (255, 255, 0)  # cyan
                elif label == "cell phone":
                    color = (255, 0, 255)  # magenta

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        return frame
    