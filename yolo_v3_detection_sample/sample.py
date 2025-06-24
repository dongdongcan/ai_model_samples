import cv2
import numpy as np
import torch

# Load YOLOv3 model and configuration

# download from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
model_cfg = "yolov3.cfg"
# download from https://pjreddie.com/media/files/yolov3.weights
model_weights = "yolov3.weights"
# download from https://github.com/pjreddie/darknet/blob/master/data/coco.names
class_file = "coco.names"

# Load class names
with open(class_file, "r") as f:
    classes = f.read().strip().split("\n")

# Load the YOLOv3 network
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def detect_faces(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # if confidence > 0.5 and classes[class_id] == 'person':
            if confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    objs = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            objs.append(image[y : y + h, x : x + w])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, objs


# Load image
image_path = "bike.png"
image = cv2.imread(image_path)

# Detect faces
result_image, objs = detect_faces(image)

# Display result
cv2.imshow("YOLOv3 Object Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
# cv2.imwrite('result.jpg', result_image)

# Optionally, save detected faces
# for i, obj in enumerate(objs):
#    cv2.imwrite(f'obj_{i}.jpg', obj)
