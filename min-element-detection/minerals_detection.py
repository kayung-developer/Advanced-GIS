from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QProgressBar, \
    QMessageBox, QHBoxLayout
from PyQt5.QtCore import Qt, QThreadPool, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QFont
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from main import MineralDetectionThread


class MineralDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mineral and Element Detection")
        self.setGeometry(200, 200, 1200, 800)

        self.setStyleSheet("background-color: #f4f7f6;")
        self.setWindowIcon(QIcon('icon.png'))  # Add a custom icon for the window

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Header Section
        self.header_label = QLabel("Mineral and Element Detection")
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.header_label.setStyleSheet("color: #2a3d6b;")
        self.layout.addWidget(self.header_label)

        self.sub_label = QLabel("Select an option for mineral detection:")
        self.sub_label.setAlignment(Qt.AlignCenter)
        self.sub_label.setFont(QFont("Arial", 14))
        self.layout.addWidget(self.sub_label)

        # Button Layouts
        self.button_layout = QVBoxLayout()

        self.load_data_button = QPushButton("Load Mineral Data (CSV/TIFF)")
        self.load_data_button.setStyleSheet(self.button_style())
        self.load_data_button.clicked.connect(self.load_mineral_data)
        self.load_data_button.setIcon(QIcon('load_icon.png'))  # Add an icon to the button
        self.load_data_button.setIconSize(Qt.QSize(30, 30))
        self.button_layout.addWidget(self.load_data_button)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 12))
        self.result_label.setStyleSheet("color: #3d3d3d;")
        self.button_layout.addWidget(self.result_label)

        self.export_button = QPushButton("Export Results to CSV")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        self.export_button.setStyleSheet(self.button_style())
        self.export_button.setIcon(QIcon('export_icon.png'))
        self.export_button.setIconSize(Qt.QSize(30, 30))
        self.button_layout.addWidget(self.export_button)

        self.visualize_button = QPushButton("Visualize Hyperspectral Minerals")
        self.visualize_button.clicked.connect(self.visualize_hyperspectral)
        self.visualize_button.setEnabled(False)
        self.visualize_button.setStyleSheet(self.button_style())
        self.visualize_button.setIcon(QIcon('visualize_icon.png'))
        self.visualize_button.setIconSize(Qt.QSize(30, 30))
        self.button_layout.addWidget(self.visualize_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar {background-color: #e6e6e6; border-radius: 10px; height: 20px;}")
        self.button_layout.addWidget(self.progress_bar)

        self.layout.addLayout(self.button_layout)

        self.mineral_data = None
        self.detection_results = []
        self.thread_pool = QThreadPool()

    def load_mineral_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Mineral Data", "",
                                                   "GeoTIFF Files (*.tif *.tiff);;CSV Files (*.csv)")
        if not file_path:
            return

        try:
            self.progress_bar.setValue(10)
            self.mineral_data = file_path

            self.mineral_detection_thread = MineralDetectionThread(self.mineral_data)
            self.mineral_detection_thread.progress.connect(self.update_progress_bar)
            self.mineral_detection_thread.result.connect(self.handle_results)
            self.thread_pool.start(self.mineral_detection_thread)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")
            self.progress_bar.setValue(0)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def handle_results(self, detected_minerals):
        if detected_minerals:
            self.detection_results = detected_minerals
            self.result_label.setText(f"Detected {len(detected_minerals)} mineral samples.")
            self.export_button.setEnabled(True)
            self.visualize_button.setEnabled(True)
        else:
            self.result_label.setText("No minerals detected.")
            self.progress_bar.setValue(0)

    def visualize_hyperspectral(self):
        if self.mineral_data.endswith(".tif") or self.mineral_data.endswith(".tiff"):
            try:
                self.visualize_minerals(self.mineral_data)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to visualize data: {e}")
        else:
            QMessageBox.warning(self, "Warning", "Visualization is only available for hyperspectral data.")

    def visualize_minerals(self, hyperspectral_tif):
        with rasterio.open(hyperspectral_tif) as src:
            hyperspectral_data = src.read(1)

        cmap = LinearSegmentedColormap.from_list('minerals', ['gray', 'green', 'blue'])

        plt.figure(figsize=(10, 7))
        plt.imshow(hyperspectral_data, cmap=cmap)
        plt.colorbar(label='Mineral Concentration')
        plt.title("Mineral Detection (Hyperspectral)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()

    def export_results(self):
        if self.detection_results:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
            if file_path:
                mineral_data = [{"index": result["index"], "pca_values": result["pca_values"]} for result in
                                self.detection_results]
                df = pd.DataFrame(mineral_data)
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", "Results exported successfully!")

    def closeEvent(self, event):
        event.accept()

    def button_style(self):
        return """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 12px;
                padding: 10px;
                font-size: 14px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #388e3c;
            }
        """
