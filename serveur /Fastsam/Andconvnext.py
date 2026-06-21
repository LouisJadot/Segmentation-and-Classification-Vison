import socket
import pickle
import struct
import cv2
import torch
import numpy as np
from ultralytics import FastSAM
from torchvision import models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights

# Modèles
model = FastSAM('FastSAM-s.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
classifier = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).to(device)
classifier.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Mapping indices → name ImageNet
imagenet_classes = ConvNeXt_Tiny_Weights.DEFAULT.meta["categories"]

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
        # reception header
        while len(data) < payload_size:
            packet = conn.recv(4*1024)
            if not packet:
                raise ConnectionError("Client déconnecté")
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # reception frame 
        while len(data) < msg_size:
            data += conn.recv(4*1024)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame, send_time = pickle.loads(frame_data)

        # Segmentation FastSAM
        results = model.track(
        frame,
        imgsz=640,          # Input image size for inference (resizes frame to 640x640)
        save=False,         # Do not save output images/videos to disk
        show=False,         # Do not display visualization window
        conf=0.8,           # Confidence threshold for detections (only keep predictions >= 0.8)
        persist=True,       # Persist tracks between frames (important for video tracking)
        tracker="bytetrack.yaml",  # Tracking algorithm configuration (ByteTrack)
        iou=0.5,            # IoU threshold for matching detections across frames
        max_det=20,         # Maximum number of detections per frame
        half=True,          # Use FP16 precision (faster inference on GPU, less memory)
        device=device       # Device to run inference on (e.g., 'cpu', 'cuda:0')
        )
        annotated_frame = results[0].plot()

        # Select the object the nearest to the center of the image

        masks_data = results[0].masks.data.cpu().numpy()  # [N,H,W]
        img_h, img_w = frame.shape[:2]
        center_img = np.array([img_w//2, img_h//2])

        closest_obj = None
        min_dist = float('inf')

        for i in range(masks_data.shape[0]):
            mask = masks_data[i].astype(bool)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            obj_center = np.array([int(xs.mean()), int(ys.mean())])
            dist = np.linalg.norm(center_img - obj_center)
            if dist < min_dist:
                min_dist = dist
                closest_obj = (i, obj_center, xs, ys, mask)

        if closest_obj is not None:
            idx, obj_center, xs, ys, mask = closest_obj

            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            obj_crop = frame[y1:y2+1, x1:x2+1].copy()

            mask_crop = mask[y1:y2+1, x1:x2+1]
            if mask_crop.sum() > 0:
                obj_crop = obj_crop * mask_crop[:, :, None]

                # Classification ConvNeXt-Tiny

                input_tensor = transform(obj_crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = classifier(input_tensor)
                    pred_class = output.argmax(dim=1).item()
                    pred_name = imagenet_classes[pred_class]

  
                cv2.circle(annotated_frame, tuple(obj_center), 8, (0,0,255), -1)
                cv2.putText(annotated_frame, pred_name, (obj_center[0]+10, obj_center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Send annotated frame back to client

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