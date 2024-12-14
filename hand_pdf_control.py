import cv2
import mediapipe as mp
import fitz  # PyMuPDF for PDF rendering
from PyQt5 import QtWidgets, QtGui, QtCore

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 0 for the first connected camera
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame)
        return frame, results

class PDFViewer(QtWidgets.QLabel):
    def __init__(self, pdf_path):
        super().__init__()
        self.doc = fitz.open(pdf_path)
        self.page_num = 0
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.render_page()

    def render_page(self):
        page = self.doc[self.page_num]
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        img = QtGui.QImage(pix.samples, pix.width, pix.height, pix.stride, QtGui.QImage.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(img))

    def set_zoom(self, zoom_factor):
        self.zoom = max(0.5, min(3.0, zoom_factor))  # Limit zoom between 0.5x and 3x
        self.render_page()

    def pan(self, dx, dy):
        self.pan_x += dx
        self.pan_y += dy
        self.render_page()

    def next_page(self):
        if self.page_num < len(self.doc) - 1:
            self.page_num += 1
            self.render_page()

    def prev_page(self):
        if self.page_num > 0:
            self.page_num -= 1
            self.render_page()

class HandControlApp(QtWidgets.QWidget):
    def __init__(self, pdf_path):
        super().__init__()
        self.viewer = PDFViewer(pdf_path)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.viewer)
        self.setLayout(layout)
        self.tracker = HandTracker()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

    def update(self):
        frame, results = self.tracker.process_frame()
        if results and results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_index_dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

            # Pinch-to-Zoom
            if thumb_index_dist < 0.05:
                self.viewer.set_zoom(self.viewer.zoom * 1.1)  # Zoom in
            elif thumb_index_dist > 0.1:
                self.viewer.set_zoom(self.viewer.zoom * 0.9)  # Zoom out

            # Swipe detection for page navigation (simplified)
            # Implement swipe gestures for page switching here

        self.viewer.update()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = HandControlApp("sample.pdf")
    window.show()
    app.exec_()
