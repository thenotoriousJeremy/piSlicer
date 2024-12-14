from picamera2 import Picamera2
import cv2
import fitz  # PyMuPDF for PDF rendering
import mediapipe as mp
import numpy as np
from absl import logging

# Suppress TensorFlow and Mediapipe warnings
logging.set_verbosity(logging.ERROR)

# Mediapipe setup for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# PDF Manipulator Class
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

        # Crop and resize the image for display
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


# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration({"size": (320, 240)})
picam2.configure(config)
picam2.start()

# Initialize PDF Manipulator
pdf_viewer = PDFManipulator("sample.pdf")

# Mediapipe Hand Tracking
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Single-hand tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    base_pinch_distance = None
    last_hand_center = None

    while True:
        frame = picam2.capture_array()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with Mediapipe
        results = hands.process(rgb_frame)

        # Convert frame back to BGR after processing
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Pinch detection for zoom
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinch_distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

            if base_pinch_distance is None:
                base_pinch_distance = pinch_distance

            if pinch_distance > base_pinch_distance * 1.2:
                pdf_viewer.zoom_in()
                base_pinch_distance = pinch_distance
            elif pinch_distance < base_pinch_distance * 0.8:
                pdf_viewer.zoom_out()
                base_pinch_distance = pinch_distance

        # Swipe detection for page navigation
        hand_center_x = (landmarks[mp_hands.HandLandmark.WRIST].x +
                        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x) / 2
        if last_hand_center is not None:
            dx = hand_center_x - last_hand_center[0]
            if dx > 0.1:  # Swipe Right
                pdf_viewer.next_page()
            elif dx < -0.1:  # Swipe Left
                pdf_viewer.prev_page()
        last_hand_center = (hand_center_x, hand_center_y)

        # Drag detection for panning
        if last_hand_center is not None:
            dx = hand_center_x - last_hand_center[0]
            dy = hand_center_y - last_hand_center[1]
            pdf_viewer.pan(dx * 0.01, dy * 0.01)

        # Render PDF frame
        pdf_frame = pdf_viewer.render()
        cv2.imshow("PDF Viewer", pdf_frame)

        # Draw hand landmarks on the BGR frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the annotated frame
        cv2.imshow("Hand Tracking", frame_bgr)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

picam2.stop()
cv2.destroyAllWindows()
