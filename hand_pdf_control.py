import cv2
import mediapipe as mp
import fitz  # PyMuPDF for PDF rendering
import numpy as np

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize PDF Viewer
class PDFManipulator:
    def __init__(self, pdf_path):
        self.pdf = fitz.open(pdf_path)
        self.page_num = 0
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.image = None
        self.load_page()

    def load_page(self):
        page = self.pdf[self.page_num]
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        self.image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    def render(self):
        h, w, _ = self.image.shape
        crop_x = int(self.pan_x * w)
        crop_y = int(self.pan_y * h)
        crop_width = min(w - crop_x, 640)
        crop_height = min(h - crop_y, 480)

        # Crop the image based on pan
        cropped_image = self.image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
        return cv2.resize(cropped_image, (640, 480))

    def zoom_in(self):
        self.zoom = min(self.zoom + 0.1, 3.0)
        self.load_page()

    def zoom_out(self):
        self.zoom = max(self.zoom - 0.1, 0.5)
        self.load_page()

    def next_page(self):
        if self.page_num < len(self.pdf) - 1:
            self.page_num += 1
            self.load_page()

    def prev_page(self):
        if self.page_num > 0:
            self.page_num -= 1
            self.load_page()

    def pan(self, dx, dy):
        self.pan_x = max(0, min(self.pan_x + dx, 1.0))
        self.pan_y = max(0, min(self.pan_y + dy, 1.0))

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize PDF Manipulator
pdf_viewer = PDFManipulator("sample.pdf")

# Mediapipe Hands configuration
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Track one hand for simplicity
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    base_pinch_distance = None
    last_hand_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert normalized Mediapipe coordinates to pixel coordinates
            h, w, _ = frame.shape
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Calculate pinch distance
            pinch_distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            # Detect pinch gesture for zooming
            if base_pinch_distance is None:
                base_pinch_distance = pinch_distance

            zoom_factor = pinch_distance / base_pinch_distance
            if zoom_factor > 1.1:
                pdf_viewer.zoom_in()
                base_pinch_distance = pinch_distance
            elif zoom_factor < 0.9:
                pdf_viewer.zoom_out()
                base_pinch_distance = pinch_distance

            # Detect swipe for page navigation
            hand_center_x = (thumb_x + index_x) // 2
            hand_center_y = (thumb_y + index_y) // 2

            if last_hand_center is not None:
                dx = hand_center_x - last_hand_center[0]
                if abs(dx) > 50:  # Swipe detection threshold
                    if dx > 0:
                        pdf_viewer.next_page()
                    else:
                        pdf_viewer.prev_page()
                    last_hand_center = None  # Reset center to prevent multiple swipes

            last_hand_center = (hand_center_x, hand_center_y)

        # Render PDF frame and overlay
        pdf_frame = pdf_viewer.render()
        cv2.imshow("PDF Viewer", pdf_frame)

        # Display the camera feed with landmarks for debugging
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Hand Tracking", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
