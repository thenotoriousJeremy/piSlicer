import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 refers to the first connected camera
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully.")

    # Adjust OpenCV camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Set the width of the video feed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the height of the video feed
    cap.set(cv2.CAP_PROP_FPS, 30)            # Optionally set the frame rate (if supported)

    # Read and display frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Camera Feed", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
