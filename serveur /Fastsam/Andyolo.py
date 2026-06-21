import socket
import pickle
import struct
import cv2
import torch
import numpy as np
from ultralytics import FastSAM, YOLO


# Modele
seg_model = FastSAM('FastSAM-s.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classifier = YOLO("yolo11n-cls.pt")

# Socket serveur
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 10000))
server_socket.listen(5)

print("Serveur listening on port 10000...")
conn, addr = server_socket.accept()
print("Connected to :", addr)

data = b""
payload_size = struct.calcsize("Q")

try:

    while True:
        # Reception header
        while len(data) < payload_size:
            packet = conn.recv(4 * 1024)
            if not packet:
                raise ConnectionError("Client déconnecté")
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Reception frame
        while len(data) < msg_size:
            data += conn.recv(4 * 1024)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame, send_time = pickle.loads(frame_data)

        # Segmentation FastSAM
        results = seg_model.track(
            frame,
            imgsz=1024,
            conf=0.8,
            iou=0.6,
            max_det=25,
            persist=True,
            tracker="bytetrack.yaml",
            half=True,
            device=device,
            verbose=False
        )

        annotated_frame = results[0].plot()

        img_h, img_w = frame.shape[:2]
        center_img = np.array([img_w // 2, img_h // 2])

        # centre camera (debug)
        cv2.circle(
            annotated_frame,
            tuple(center_img),
            6,
            (0, 255, 0),
            -1
        )

        selected_obj = None
        best_area = float('inf')

        # Bounding boxes FastSAM
        if results[0].boxes is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = map(int, box)

                # check if center of image is inside bbox
                if x1 <= center_img[0] <= x2 and y1 <= center_img[1] <= y2:

                    area = (x2 - x1) * (y2 - y1)

                    # select the object with the smallest area (closest to camera)
                    if area < best_area:
                        best_area = area
                        selected_obj = (i, x1, y1, x2, y2)

        # Classification objject selected
        if selected_obj is not None:

            idx, x1, y1, x2, y2 = selected_obj

            obj_center = np.array([
                (x1 + x2) // 2,
                (y1 + y2) // 2
            ])

            margin = 200

            x1m = max(0, x1 - margin)
            y1m = max(0, y1 - margin)
            x2m = min(img_w, x2 + margin)
            y2m = min(img_h, y2 + margin)

            obj_crop = frame[y1m:y2m, x1m:x2m]

            if obj_crop.size > 0:

                cls_results = classifier(
                    obj_crop,
                    device=device,
                    verbose=False
                )

                pred_id = int(cls_results[0].probs.top1)
                pred_name = cls_results[0].names[pred_id]
                confidence = float(cls_results[0].probs.top1conf)

                label = f"{pred_name} {confidence:.2f}"



                # red circle at object center
                cv2.circle(
                    annotated_frame,
                    tuple(obj_center),
                    8,
                    (0, 0, 255),
                    -1
                )

                # bbox bleu + label
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2)

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)
        _, buffer = cv2.imencode('.jpg', annotated_frame)

        frame_data_send = pickle.dumps((buffer, send_time))
        msg_size_send = struct.pack("Q", len(frame_data_send))

        conn.sendall(msg_size_send + frame_data_send)

except KeyboardInterrupt:
    print("\nMaunual stop detected (Ctrl+C)")

except Exception as e:
    print("Error :", e)

finally:
    conn.close()
    server_socket.close()
    print("Serveur closed.")