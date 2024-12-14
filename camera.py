import cv2

cap = cv2.VideoCapture(0)  # Open the camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Adjust camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert YUYV to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
