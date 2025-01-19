from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QSlider, \
    QProgressBar, QFileDialog, QMessageBox, QFrame, QMainWindow
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import rasterio
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class SeasonDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Season Detection: Start and End Dates")
        self.setGeometry(200, 200, 1000, 750)
        self.setWindowIcon(QIcon('icon.png'))  # Optional: Set a window icon

        # Main widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # UI Components
        self.label = QLabel("Load a temporal raster dataset (GeoTIFF format):")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Arial", 14, QFont.Bold))
        self.layout.addWidget(self.label)

        self.load_button = QPushButton("Load Raster File")
        self.load_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; border-radius: 5px;")
        self.load_button.clicked.connect(self.load_raster)
        self.layout.addWidget(self.load_button)

        self.threshold_label = QLabel("Set Threshold for Season Detection:")
        self.threshold_label.setAlignment(Qt.AlignCenter)
        self.threshold_label.setFont(QFont("Arial", 14))
        self.layout.addWidget(self.threshold_label)

        # Threshold input
        self.threshold_input = QLineEdit("Enter custom threshold or leave blank")
        self.threshold_input.setStyleSheet("padding: 5px; font-size: 14px;")
        self.layout.addWidget(self.threshold_input)

        # Slider for threshold adjustment
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.setStyleSheet("QSlider {height: 10px; background-color: #f0f0f0;} QSlider::handle:horizontal {background-color: #4CAF50; width: 20px;}")
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        self.layout.addWidget(self.threshold_slider)

        self.slider_value_label = QLabel("Slider Threshold: 50")
        self.slider_value_label.setAlignment(Qt.AlignCenter)
        self.slider_value_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.slider_value_label)

        self.analyze_button = QPushButton("Analyze Data")
        self.analyze_button.setStyleSheet("background-color: #008CBA; color: white; font-size: 14px; border-radius: 5px;")
        self.analyze_button.clicked.connect(self.detect_seasons)
        self.layout.addWidget(self.analyze_button)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 14))
        self.layout.addWidget(self.result_label)

        self.export_button = QPushButton("Export Results to CSV")
        self.export_button.setStyleSheet("background-color: #FF9800; color: white; font-size: 14px; border-radius: 5px;")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        self.layout.addWidget(self.export_button)

        self.visualize_button = QPushButton("Visualize Season (Rainfall Data)")
        self.visualize_button.setStyleSheet("background-color: #9C27B0; color: white; font-size: 14px; border-radius: 5px;")
        self.visualize_button.clicked.connect(self.visualize_season)
        self.layout.addWidget(self.visualize_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar {background-color: #f0f0f0; border-radius: 5px; color: white;} QProgressBar::chunk {background-color: #4CAF50;}")
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self.canvas = FigureCanvas(plt.figure())
        self.layout.addWidget(self.canvas)

        # Class attributes
        self.data = None  # Placeholder for raster data
        self.dates = None  # Placeholder for temporal data (e.g., monthly timestamps)
        self.start_date = None
        self.end_date = None

    def load_raster(self):
        """Load the GeoTIFF file and process it."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Raster File", "", "GeoTIFF Files (*.tif)")
        if not file_path:
            return

        try:
            self.progress_bar.setValue(10)
            with rasterio.open(file_path) as src:
                self.data = src.read()  # Reading all bands (assumed temporal data)
                self.dates = src.descriptions  # Temporal descriptions (if available)

            self.result_label.setText(f"Loaded raster file: {os.path.basename(file_path)}")
            self.progress_bar.setValue(50)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load raster file: {e}")
            self.progress_bar.setValue(0)

    def detect_seasons(self):
        """Analyze the raster data to detect start and end of seasons."""
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load a valid dataset first.")
            return

        self.progress_bar.setValue(60)
        try:
            # Get custom threshold from input or slider
            custom_threshold = self.threshold_input.text()
            if custom_threshold:
                threshold = float(custom_threshold)
            else:
                threshold = self.threshold_slider.value() / 100 * np.max(self.data)

            # Example logic: Calculate mean values across all bands
            mean_values = np.mean(self.data, axis=(1, 2))  # Temporal mean per band
            time_series = np.arange(len(mean_values))  # Assume indices represent time steps

            # Detect start and end of seasons using threshold
            start_idx = next((i for i, val in enumerate(mean_values) if val > threshold), None)
            end_idx = next((i for i, val in reversed(list(enumerate(mean_values))) if val > threshold), None)

            if start_idx is not None and end_idx is not None:
                self.start_date = self.dates[start_idx] if self.dates else f"Time Step {start_idx + 1}"
                self.end_date = self.dates[end_idx] if self.dates else f"Time Step {end_idx + 1}"
                self.result_label.setText(f"Season Start: {self.start_date} | Season End: {self.end_date}")
            else:
                self.result_label.setText("No clear season detected.")

            self.plot_data(time_series, mean_values, threshold)
            self.export_button.setEnabled(True)
            self.progress_bar.setValue(100)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {e}")
            self.progress_bar.setValue(0)

    def update_threshold_label(self):
        """Update threshold label with slider value."""
        value = self.threshold_slider.value()
        self.slider_value_label.setText(f"Slider Threshold: {value}")

    def plot_data(self, time_series, mean_values, threshold):
        """Plot the seasonal data."""
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()
        ax.plot(time_series, mean_values, label="Mean Values")
        ax.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
        ax.set_title("Seasonal Data Analysis")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Mean Value")
        ax.legend()
        self.canvas.draw()

    def export_results(self):
        """Export detected season data to a CSV file."""
        if self.start_date and self.end_date:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
            if file_path:
                data = {
                    "Start Date": [self.start_date],
                    "End Date": [self.end_date],
                }
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", "Results exported successfully!")

    def visualize_season(self):
        """Visualize the season based on rainfall data."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Rainfall Raster File", "", "GeoTIFF Files (*.tif)")
        if not file_path:
            return

        try:
            with rasterio.open(file_path) as src:
                rainfall_data = src.read(1)

            # Set color map for seasons (blue for rainy, yellow for dry)
            cmap = LinearSegmentedColormap.from_list('rainy_dry', ['blue', 'yellow'])

            # Create the plot
            plt.imshow(rainfall_data, cmap=cmap)
            plt.colorbar()
            plt.title("Season Detection (Rainfall)")

            # Display the plot
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to visualize rainfall data: {e}")

if __name__ == "__main__":
    app = QApplication([])
    window = SeasonDetectionApp()
    window.show()
    app.exec_()
