import cv2
import socket
import numpy as np
import datetime
import time

def send_frame_in_chunks( frame, max_chunk_size=65000):
    # Encode the frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    jpg_as_text = buffer.tobytes()

    # Calculate total chunks needed for this frame
    total_chunks = len(jpg_as_text) // max_chunk_size + (1 if len(jpg_as_text) % max_chunk_size > 0 else 0)
    
    # Get the current timestamp in ISO format
    timestamp = datetime.datetime.now()

    for chunk_number in range(total_chunks):
        start = chunk_number * max_chunk_size
        end = start + max_chunk_size
        chunk = jpg_as_text[start:end]


        # Prepare and send header + chunk. Header format: frame_number,chunk_number,total_chunks;
        header = f"{timestamp},{chunk_number},{total_chunks};".encode()
        packet = header + chunk
        # Print the size of the JPEG data
        print(f"Sending frame of size {len(packet)} bytes")
        sock.sendto(packet, server_address)

# Initialize the UDP client
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# The server's address and port
server_address = ('localhost', 9000)

robot_id = "camera01"
# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        send_frame_in_chunks(frame)
        time.sleep(0.05)

finally:
    # When everything done, release the capture
    cap.release()
    sock.close()
