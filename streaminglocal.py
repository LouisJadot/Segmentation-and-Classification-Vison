import socket
import cv2
import pickle
import struct
import time

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 10000))

cap = cv2.VideoCapture(2)
payload_size = struct.calcsize("Q")
data = b""

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #timestamp before send
        send_time = time.time()
        data_to_send = pickle.dumps((frame, send_time))
        message = struct.pack("Q", len(data_to_send)) + data_to_send
        client_socket.sendall(message)

        #reception frame   
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            if not packet:
                break
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4*1024)

        frame_data_recv = data[:msg_size]
        data = data[msg_size:]

        #decodage frame 
        buffer, send_time = pickle.loads(frame_data_recv)
        annotated_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        #calcul latence end-to-end
        latency_ms = (time.time() - send_time) * 1000
        cv2.putText(annotated_frame, f"Latence: {latency_ms:.0f} ms", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Segmentation + Classification Live", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()