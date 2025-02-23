import cv2
import numpy as np
import time
import math

def estimate_speed(pt1, pt2, ppm, fps):
    # Calculate Euclidean distance in pixels
    d_pixels = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    # Convert pixel distance to meters using pixels per meter (ppm)
    d_meters = d_pixels / ppm
    # Convert speed to km/h (meters * fps gives m/s, multiplied by 3.6 gives km/h)
    speed = d_meters * fps * 3.6
    return speed


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


video_path = "cars.mp4"
cap = cv2.VideoCapture(video_path)


fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video FPS: {fps}, Frame Size: {frame_width}x{frame_height}")

ppm = 8.8

vehicle_tracker = {}
vehicle_speeds = {}
current_vehicle_id = 0

detect_interval = 10
frame_counter = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    result_frame = frame.copy()

    if frame_counter % detect_interval == 0:
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        current_centroids = {}

        boxes = []
        confidences = []
        centroids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] == "car":
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    w = int(detection[2] * frame_width)
                    h = int(detection[3] * frame_height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    centroids.append((center_x, center_y))


        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


        new_vehicle_tracker = {}
        for i in indexes.flatten() if len(indexes) > 0 else []:
            box = boxes[i]
            cx, cy = centroids[i]
            matched_id = None
            for vid, prev_centroid in vehicle_tracker.items():

                distance = math.hypot(cx - prev_centroid[0], cy - prev_centroid[1])
                if distance < 50:
                    matched_id = vid
                    break
            if matched_id is None:

                matched_id = current_vehicle_id
                current_vehicle_id += 1
            new_vehicle_tracker[matched_id] = (cx, cy)

            x, y, w, h = box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(result_frame, (cx, cy), 4, (0, 0, 255), -1)


        for vid, centroid in new_vehicle_tracker.items():
            if vid in vehicle_tracker:

                prev_centroid = vehicle_tracker[vid]
                speed = estimate_speed(prev_centroid, centroid, ppm, fps / detect_interval)
                vehicle_speeds[vid] = speed
                cv2.putText(result_frame, f"{int(speed)} km/h", (centroid[0], centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:

                vehicle_speeds[vid] = 0

        vehicle_tracker = new_vehicle_tracker.copy()
    else:

        for vid, centroid in vehicle_tracker.items():
            cv2.circle(result_frame, centroid, 4, (0, 0, 255), -1)
            if vid in vehicle_speeds:
                cv2.putText(result_frame, f"{int(vehicle_speeds[vid])} km/h", (centroid[0], centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    cv2.imshow("Advanced Vehicle Speed Detection", result_frame)


    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
