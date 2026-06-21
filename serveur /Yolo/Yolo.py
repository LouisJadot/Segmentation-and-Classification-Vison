import socket
import pickle
import struct
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Modèle YOLOv8 for detection + tracking
device = 'cuda' if torch.cuda.is_available() else 'cpu'
track_model = YOLO("yolov8s.pt")  # YOLOv8 small: bon compromis vitesse/précision

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

        # Detection + tracking YOLOv8
        results = track_model.track(
            frame,
            device=device,
            tracker="bytetrack.yaml",  
            conf=0.7,                  
            imgsz=1280,                
            half=False,                
            persist=True,
            iou=0.5,
            max_det=25
            )

        annotated_frame = results[0].plot()

        img_h, img_w = frame.shape[:2]
        center_img = np.array([img_w // 2, img_h // 2])

        # centre caméra (debug)
        cv2.circle(annotated_frame, tuple(center_img), 6, (0, 255, 0), -1)

        selected_obj = None
        best_area = float('inf')

        # Select object the nearest to the center of the image
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)

                # check if center of image is inside bbox
                if x1 <= center_img[0] <= x2 and y1 <= center_img[1] <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if area < best_area:
                        best_area = area
                        selected_obj = (i, x1, y1, x2, y2)

        # Show bbox + info object selected
        if selected_obj is not None:
            idx, x1, y1, x2, y2 = selected_obj
            obj_center = np.array([(x1 + x2) // 2, (y1 + y2) // 2])

            # bounding box + centre
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(annotated_frame, tuple(obj_center), 8, (0, 0, 255), -1)

            # name and confidence
            if results[0].boxes.cls is not None:
                cls_id = int(results[0].boxes.cls[idx].cpu())
                cls_name = results[0].names[cls_id]
                confidence = float(results[0].boxes.conf[idx].cpu())
                label = f"{cls_name} {confidence:.2f}"
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )


        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_data_send = pickle.dumps((buffer, send_time))
        msg_size_send = struct.pack("Q", len(frame_data_send))
        conn.sendall(msg_size_send + frame_data_send)

except KeyboardInterrupt:
    print("\nManual stop detected (Ctrl+C)")
except Exception as e:
    print("Error :", e)
finally:
    conn.close()
    server_socket.close()
    print("Serveur closed.")