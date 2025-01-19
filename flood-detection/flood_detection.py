import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QFileDialog, QWidget,
    QMessageBox, QProgressBar, QGraphicsScene, QGraphicsView, QHBoxLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QColor
import pandas as pd


class FloodDetectionThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list)

    def __init__(self, dem_file_path, elevation_threshold):
        super().__init__()
        self.dem_file_path = dem_file_path
        self.elevation_threshold = elevation_threshold

    def run(self):
        try:
            self.progress.emit(10)
            elevation_data = self.load_dem_data(self.dem_file_path)
            self.progress.emit(50)
            flood_prone_areas = self.detect_flood_prone_areas(elevation_data)
            self.progress.emit(100)
            self.result.emit(flood_prone_areas)
        except Exception as e:
            self.progress.emit(0)
            self.result.emit([])
            raise e

    def load_dem_data(self, file_path):
        with rasterio.open(file_path) as src:
            dem_data = src.read(1)
            return dem_data

    def detect_flood_prone_areas(self, elevation_data):
        flood_prone_areas = []
        rows, cols = elevation_data.shape

        for i in range(rows):
            for j in range(cols):
                if elevation_data[i, j] <= self.elevation_threshold:
                    flood_prone_areas.append({"x": j, "y": i, "elevation": elevation_data[i, j]})

        return flood_prone_areas


class FloodDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flood-Prone Area Detection")
        self.setGeometry(200, 200, 1200, 800)
        self.setStyleSheet("background-color: #f0f0f0;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.header_label = QLabel("Flood-Prone Area Detection System")
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.header_label.setStyleSheet("color: #2E86C1;")
        self.layout.addWidget(self.header_label)

        button_layout = QHBoxLayout()

        self.load_dem_button = QPushButton("Load DEM File")
        self.load_dem_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #3498DB; color: white; border-radius: 5px;")
        self.load_dem_button.clicked.connect(self.load_dem_for_detection)
        button_layout.addWidget(self.load_dem_button)

        self.visualize_button = QPushButton("Visualize DEM")
        self.visualize_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #2ECC71; color: white; border-radius: 5px;")
        self.visualize_button.clicked.connect(self.visualize_dem)
        button_layout.addWidget(self.visualize_button)

        self.layout.addLayout(button_layout)

        self.result_label = QLabel("No flood-prone areas detected yet.")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 14))
        self.result_label.setStyleSheet("color: #8E44AD;")
        self.layout.addWidget(self.result_label)

        self.export_button = QPushButton("Export Results to CSV")
        self.export_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #E74C3C; color: white; border-radius: 5px;")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        self.layout.addWidget(self.export_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar {border: 2px solid grey; border-radius: 5px;} QProgressBar::chunk {background-color: #27AE60;}")
        self.layout.addWidget(self.progress_bar)

        self.scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.scene)
        self.graphics_view.setStyleSheet("border: 1px solid #BDC3C7;")
        self.layout.addWidget(self.graphics_view)

        self.dem_file_path = None
        self.flood_results = []
        self.elevation_threshold = 10

    def load_dem_for_detection(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open DEM File", "", "GeoTIFF Files (*.tif *.tiff)")
        if not file_path:
            return

        try:
            self.progress_bar.setValue(10)
            self.dem_file_path = file_path

            self.flood_detection_thread = FloodDetectionThread(self.dem_file_path, self.elevation_threshold)
            self.flood_detection_thread.progress.connect(self.update_progress_bar)
            self.flood_detection_thread.result.connect(self.handle_results)
            self.flood_detection_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load DEM: {e}")
            self.progress_bar.setValue(0)

    def visualize_dem(self):
        if not self.dem_file_path:
            QMessageBox.warning(self, "Warning", "Please load a DEM file first!")
            return
        try:
            self.scene.clear()
            pixmap = self.generate_dem_image(self.dem_file_path)
            self.scene.addPixmap(pixmap)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to visualize DEM: {e}")

    def generate_dem_image(self, dem_tif):
        with rasterio.open(dem_tif) as src:
            dem_data = src.read(1)
        plt.figure(figsize=(10, 7))
        plt.imshow(dem_data, cmap='terrain')
        plt.axis('off')
        plt.tight_layout()
        image_path = "temp_dem_image.png"
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return QPixmap(image_path)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def handle_results(self, flood_results):
        if flood_results:
            self.flood_results = flood_results
            self.result_label.setText(f"Detected {len(flood_results)} flood-prone areas.")
            self.export_button.setEnabled(True)
        else:
            self.result_label.setText("No flood-prone areas detected.")
            self.progress_bar.setValue(0)

    def export_results(self):
        if self.flood_results:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
            if file_path:
                flood_data = [{"x": area["x"], "y": area["y"], "elevation": area["elevation"]} for area in self.flood_results]
                df = pd.DataFrame(flood_data)
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", "Results exported successfully!")

    def closeEvent(self, event):
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FloodDetectionApp()
    window.show()
    sys.exit(app.exec_())
