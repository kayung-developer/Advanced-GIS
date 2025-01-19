import sys
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QFileDialog, QWidget,
    QMessageBox, QLineEdit, QSlider, QProgressBar, QTabWidget, QGroupBox, QHBoxLayout
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QIcon, QPixmap

class HarvestingTimeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Determination of Harvesting Time")
        self.setGeometry(200, 200, 1200, 800)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Welcome banner
        self.banner = QLabel("Determination of Harvesting Time", self)
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50; padding: 10px;")
        self.layout.addWidget(self.banner)

        # Tabs for organizing UI
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Tab: Data Loading
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "Load Data")
        self.setup_data_tab()

        # Tab: Analysis
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.setup_analysis_tab()

        # Tab: Visualization
        self.visualization_tab = QWidget()
        self.tabs.addTab(self.visualization_tab, "Visualization")
        self.setup_visualization_tab()

        # Class attributes
        self.ndvi_data = None  # Placeholder for vegetation data
        self.dates = None  # Placeholder for temporal data (e.g., monthly timestamps)
        self.harvest_time = None
        self.file_path = None

    def setup_data_tab(self):
        layout = QVBoxLayout()
        group_box = QGroupBox("Load Vegetation Index Dataset")
        group_box.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; color: #003366; }")
        group_layout = QVBoxLayout()

        label = QLabel("Choose a vegetation index dataset (GeoTIFF or CSV):")
        label.setStyleSheet("font-size: 14px; padding: 5px;")
        group_layout.addWidget(label)

        self.load_button = QPushButton("Load Data")
        self.load_button.setIcon(QIcon(QPixmap("icons/open_file.png")))
        self.load_button.setStyleSheet("padding: 10px; font-size: 14px;")
        self.load_button.clicked.connect(self.load_vegetation_data)
        group_layout.addWidget(self.load_button)

        self.result_label = QLabel("")
        self.result_label.setStyleSheet("font-size: 12px; color: #555555; padding: 5px;")
        group_layout.addWidget(self.result_label)

        group_box.setLayout(group_layout)
        layout.addWidget(group_box)
        self.data_tab.setLayout(layout)

    def setup_analysis_tab(self):
        layout = QVBoxLayout()
        group_box = QGroupBox("Threshold Adjustment and Analysis")
        group_box.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; color: #003366; }")
        group_layout = QVBoxLayout()

        threshold_label = QLabel("Set NDVI Threshold for Harvesting:")
        threshold_label.setStyleSheet("font-size: 14px; padding: 5px;")
        group_layout.addWidget(threshold_label)

        self.threshold_input = QLineEdit("Enter custom threshold (e.g., 0.5)")
        self.threshold_input.setStyleSheet("padding: 8px; font-size: 14px;")
        group_layout.addWidget(self.threshold_input)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        group_layout.addWidget(self.threshold_slider)

        self.slider_value_label = QLabel("Slider Threshold: 50 (0.50 in NDVI)")
        self.slider_value_label.setStyleSheet("font-size: 12px; color: #555555; padding: 5px;")
        group_layout.addWidget(self.slider_value_label)

        self.analyze_button = QPushButton("Analyze Harvest Time")
        self.analyze_button.setIcon(QIcon(QPixmap("icons/analyze.png")))
        self.analyze_button.setStyleSheet("padding: 10px; font-size: 14px;")
        self.analyze_button.clicked.connect(self.determine_harvest_time)
        group_layout.addWidget(self.analyze_button)

        self.export_button = QPushButton("Export Results to CSV")
        self.export_button.setIcon(QIcon(QPixmap("icons/export.png")))
        self.export_button.setStyleSheet("padding: 10px; font-size: 14px;")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        group_layout.addWidget(self.export_button)

        self.progress_bar = QProgressBar()
        group_layout.addWidget(self.progress_bar)

        group_box.setLayout(group_layout)
        layout.addWidget(group_box)
        self.analysis_tab.setLayout(layout)

    def setup_visualization_tab(self):
        layout = QVBoxLayout()
        group_box = QGroupBox("Data Visualization")
        group_box.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; color: #003366; }")
        group_layout = QVBoxLayout()

        self.visualize_button = QPushButton("Visualize Harvest Time")
        self.visualize_button.setIcon(QIcon(QPixmap("icons/visualize.png")))
        self.visualize_button.setStyleSheet("padding: 10px; font-size: 14px;")
        self.visualize_button.clicked.connect(self.visualize_harvest_time)
        self.visualize_button.setEnabled(False)
        group_layout.addWidget(self.visualize_button)

        self.canvas = FigureCanvas(plt.figure())
        group_layout.addWidget(self.canvas)

        group_box.setLayout(group_layout)
        layout.addWidget(group_box)
        self.visualization_tab.setLayout(layout)

    def load_vegetation_data(self):
        """Load the vegetation index dataset."""
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open Vegetation Data File", "", "GeoTIFF/CSV Files (*.tif *.csv)")
        if not self.file_path:
            return

        try:
            self.progress_bar.setValue(10)
            if self.file_path.endswith(".tif"):
                with rasterio.open(self.file_path) as src:
                    self.ndvi_data = src.read(1)  # Assuming single-band NDVI data
                    self.dates = src.descriptions  # Temporal descriptions (if available)
            elif self.file_path.endswith(".csv"):
                df = pd.read_csv(self.file_path)
                self.ndvi_data = df['NDVI'].values
                self.dates = df['Date'].values

            self.result_label.setText(f"Loaded vegetation data: {os.path.basename(self.file_path)}")
            self.visualize_button.setEnabled(True)
            self.progress_bar.setValue(50)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load vegetation data: {e}")
            self.progress_bar.setValue(0)

    def determine_harvest_time(self):
        """Analyze NDVI data to determine the optimal harvesting time."""
        if self.ndvi_data is None:
            QMessageBox.warning(self, "Warning", "Please load a valid dataset first.")
            return

        self.progress_bar.setValue(60)
        try:
            custom_threshold = self.threshold_input.text()
            threshold = float(custom_threshold) if custom_threshold else self.threshold_slider.value() / 100

            time_series = np.arange(len(self.ndvi_data))
            harvest_idx = next((i for i, val in enumerate(self.ndvi_data) if val >= threshold), None)

            if harvest_idx is not None:
                self.harvest_time = self.dates[harvest_idx] if self.dates else f"Time Step {harvest_idx + 1}"
                self.result_label.setText(f"Optimal Harvesting Time: {self.harvest_time}")
            else:
                self.result_label.setText("No optimal harvesting time detected.")

            self.plot_data(time_series, self.ndvi_data, threshold)
            self.export_button.setEnabled(True)
            self.progress_bar.setValue(100)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {e}")
            self.progress_bar.setValue(0)

    def update_threshold_label(self):
        """Update threshold label with slider value."""
        value = self.threshold_slider.value()
        self.slider_value_label.setText(f"Slider Threshold: {value} ({value / 100:.2f} in NDVI)")

    def plot_data(self, time_series, ndvi_values, threshold):
        """Plot the NDVI data."""
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()
        ax.plot(time_series, ndvi_values, label="NDVI Values")
        ax.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
        ax.set_title("NDVI Analysis for Harvesting")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("NDVI")
        ax.legend()
        self.canvas.draw()

    def visualize_harvest_time(self):
        """Visualize the NDVI dataset with harvest time indication."""
        if not self.file_path or not self.file_path.endswith(".tif"):
            QMessageBox.warning(self, "Warning", "Visualization is supported only for GeoTIFF files.")
            return

        try:
            with rasterio.open(self.file_path) as src:
                ndvi_data = src.read(1)

            cmap = LinearSegmentedColormap.from_list('harvest', ['green', 'yellow', 'brown'])
            plt.imshow(ndvi_data, cmap=cmap)
            plt.colorbar()
            plt.title("Harvest Time Detection (NDVI)")
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Visualization failed: {e}")

    def export_results(self):
        """Export detected harvest time to a CSV file."""
        if self.harvest_time:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
            if file_path:
                data = {
                    "Optimal Harvesting Time": [self.harvest_time],
                }
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", "Results exported successfully!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HarvestingTimeApp()
    window.show()
    sys.exit(app.exec_())
