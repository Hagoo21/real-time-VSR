import cv2

def get_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")
    return cap
