import cv2
import mediapipe as mp
import fitz  # PyMuPDF for handling PDF files
import numpy as np

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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
        crop_width = min(w - crop_x, 800)
        crop_height = min(h - crop_y, 600)

        # Crop the image based on pan
        cropped_image = self.image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
        return cv2.resize(cropped_image, (800, 600))

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

class HandGesturePDFControl:
    def __init__(self, pdf_path):
        self.pdf = PDFManipulator(pdf_path)
        self.cap = cv2.VideoCapture(0)
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

        self.base_pinch_distance = None
        self.last_hand_center = None

    def detect_gesture(self, landmarks, w, h):
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Convert normalized Mediapipe coordinates to pixel coordinates
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

        # Calculate distance between thumb and index finger
        pinch_distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

        return (thumb_x, thumb_y, index_x, index_y, pinch_distance)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Process the frame with Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                thumb_x, thumb_y, index_x, index_y, pinch_distance = self.detect_gesture(hand_landmarks.landmark, w, h)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect pinch gesture
                if self.base_pinch_distance is None:
                    self.base_pinch_distance = pinch_distance

                zoom_factor = pinch_distance / self.base_pinch_distance
                if zoom_factor > 1.1:
                    self.pdf.zoom_in()
                    self.base_pinch_distance = pinch_distance
                elif zoom_factor < 0.9:
                    self.pdf.zoom_out()
                    self.base_pinch_distance = pinch_distance

                # Detect swipe for page navigation
                hand_center_x = (thumb_x + index_x) // 2
                hand_center_y = (thumb_y + index_y) // 2

                if self.last_hand_center is not None:
                    dx = hand_center_x - self.last_hand_center[0]
                    if abs(dx) > 50:
                        if dx > 0:
                            self.pdf.next_page()
                        else:
                            self.pdf.prev_page()
                        self.last_hand_center = None  # Reset center to prevent multiple swipes

                self.last_hand_center = (hand_center_x, hand_center_y)

            # Render the PDF and show it
            pdf_frame = self.pdf.render()
            cv2.imshow("PDF Viewer", pdf_frame)

            # Show the camera feed for debugging (optional)
            cv2.imshow("Camera", frame)

            # Break on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pdf_control = HandGesturePDFControl("sample.pdf")
    pdf_control.run()
