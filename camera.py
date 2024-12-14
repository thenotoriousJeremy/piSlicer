import cv2

cap = cv2.VideoCapture(0)  # Open the default camera
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
