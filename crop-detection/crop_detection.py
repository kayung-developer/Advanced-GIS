import sys
import os
import cv2
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QFileDialog, QWidget,
    QMessageBox, QHBoxLayout, QProgressBar, QFrame, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont


class CropDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detection and Health Visualization of Crops")
        self.setGeometry(200, 200, 1200, 800)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)

        # Create a top frame for header
        self.header_frame = QFrame()
        self.header_frame.setStyleSheet("background-color: #283747; border-radius: 8px;")
        self.header_layout = QVBoxLayout(self.header_frame)
        self.main_layout.addWidget(self.header_frame)

        self.title_label = QLabel("Detection and Health Visualization of Crops")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        self.header_layout.addWidget(self.title_label)

        # Add a group box for controls
        self.controls_group = QGroupBox("Crop Detection Options")
        self.controls_group.setStyleSheet("""
                QGroupBox {
                    font-size: 16px; font-weight: bold; color: #34495E;
                    border: 2px solid #34495E; border-radius: 8px; margin-top: 10px;
                }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            """)
        self.controls_layout = QVBoxLayout(self.controls_group)
        self.main_layout.addWidget(self.controls_group)

        self.load_image_button = QPushButton("Load Image for Detection")
        self.load_image_button.setStyleSheet(self.get_button_style())
        self.load_image_button.clicked.connect(self.load_image_for_detection)
        self.controls_layout.addWidget(self.load_image_button)

        self.start_camera_button = QPushButton("Start Camera for Real-Time Detection")
        self.start_camera_button.setStyleSheet(self.get_button_style())
        self.start_camera_button.clicked.connect(self.start_camera)
        self.controls_layout.addWidget(self.start_camera_button)

        self.load_ndvi_button = QPushButton("Visualize Crop Health (NDVI)")
        self.load_ndvi_button.setStyleSheet(self.get_button_style())
        self.load_ndvi_button.clicked.connect(self.load_ndvi_file)
        self.controls_layout.addWidget(self.load_ndvi_button)

        self.export_button = QPushButton("Export Results to CSV")
        self.export_button.setStyleSheet(self.get_button_style(disabled=True))
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        self.controls_layout.addWidget(self.export_button)

        # Add a result label and progress bar
        self.result_label = QLabel("No operations performed yet.")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 14px; color: #2C3E50; padding: 10px;")
        self.controls_layout.addWidget(self.result_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #34495E; border-radius: 5px; text-align: center; height: 20px;
                }
                QProgressBar::chunk {
                    background-color: #1ABC9C; width: 20px;
                }
            """)
        self.controls_layout.addWidget(self.progress_bar)

        # Camera view label
        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet("border: 2px solid #2C3E50; border-radius: 8px;")
        self.main_layout.addWidget(self.camera_view, 1)

        # Footer
        self.footer_label = QLabel("Powered by AI and Computer Vision | © 2025")
        self.footer_label.setAlignment(Qt.AlignCenter)
        self.footer_label.setStyleSheet("color: #7F8C8D; font-size: 12px; padding: 5px;")
        self.main_layout.addWidget(self.footer_label)

        # Class attributes
        self.image_path = None
        self.detection_results = []
        self.camera = None
        self.timer = QTimer()

    def get_button_style(self, disabled=False):
        if disabled:
            return """
                    QPushButton {
                        background-color: #BDC3C7; color: white; border: none; border-radius: 8px;
                        padding: 10px; font-size: 14px;
                    }
                """
        return """
                QPushButton {
                    background-color: #3498DB; color: white; border: none; border-radius: 8px;
                    padding: 10px; font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #2980B9;
                }
            """
    def load_image_for_detection(self):
        """Load an image for crop detection."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.jpg *.png *.tif)")
        if not file_path:
            return

        try:
            self.progress_bar.setValue(10)
            self.image_path = file_path

            # Simulate detection logic (replace with ML model inference)
            self.progress_bar.setValue(50)
            image = cv2.imread(self.image_path)
            detected_crops = self.detect_crops(image)
            self.result_label.setText(f"Detected {len(detected_crops)} crops in the image.")
            self.detection_results = detected_crops
            self.export_button.setEnabled(True)

            # Show the image with detections
            image_with_boxes = self.draw_boxes(image, detected_crops)
            self.display_image(image_with_boxes)

            self.progress_bar.setValue(100)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to detect crops: {e}")
            self.progress_bar.setValue(0)

    def detect_crops(self, image):
        """Simulate crop detection logic."""
        # Replace with a trained model's inference logic.
        # For simulation, we return dummy results.
        height, width, _ = image.shape
        crops = [
            {"label": "CropA", "x1": 50, "y1": 50, "x2": 150, "y2": 150},
            {"label": "CropB", "x1": 200, "y1": 200, "x2": 300, "y2": 300}
        ]
        return crops

    def draw_boxes(self, image, crops):
        """Draw bounding boxes and labels on the image."""
        for crop in crops:
            x1, y1, x2, y2 = crop["x1"], crop["y1"], crop["x2"], crop["y2"]
            label = crop["label"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def display_image(self, image):
        """Display the image in the QLabel."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.camera_view.setPixmap(pixmap)

    def start_camera(self):
        """Start the camera for real-time detection."""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            QMessageBox.critical(self, "Error", "Failed to access the camera.")
            return

        self.timer.timeout.connect(self.update_camera_frame)
        self.timer.start(30)  # Update every 30 ms

    def update_camera_frame(self):
        """Update the camera view with real-time crop detection."""
        ret, frame = self.camera.read()
        if not ret:
            self.timer.stop()
            self.camera.release()
            QMessageBox.critical(self, "Error", "Failed to read from the camera.")
            return

        detected_crops = self.detect_crops(frame)
        frame_with_boxes = self.draw_boxes(frame, detected_crops)
        self.display_image(frame_with_boxes)

    def load_ndvi_file(self):
        """Load and visualize an NDVI file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open NDVI File", "", "GeoTIFF Files (*.tif)")
        if not file_path:
            return

        try:
            self.visualize_crop_health(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to visualize NDVI: {e}")

    def visualize_crop_health(self, ndvi_tif):
        """Visualize crop health using NDVI."""
        with rasterio.open(ndvi_tif) as src:
            ndvi_data = src.read(1)

        # Create a colormap for NDVI
        cmap = LinearSegmentedColormap.from_list('ndvi_colors', ['red', 'yellow', 'green'])

        # Plot the NDVI data
        plt.imshow(ndvi_data, cmap=cmap)
        plt.colorbar()
        plt.title("Crop Health Detection (NDVI)")

        # Show the plot
        plt.show()

    def export_results(self):
        """Export detection results to a CSV file."""
        if self.detection_results:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
            if file_path:
                df = pd.DataFrame(self.detection_results)
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", "Results exported successfully!")

    def closeEvent(self, event):
        """Handle application close events."""
        if self.camera:
            self.timer.stop()
            self.camera.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CropDetectionApp()
    window.show()
    sys.exit(app.exec_())
