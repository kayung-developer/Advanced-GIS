import sys
import os
import cv2
import numpy as np
import folium
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFrame, QGroupBox, QGraphicsScene, QGraphicsView, QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QWidget,
    QMessageBox, QLineEdit, QSlider, QProgressBar, QTabWidget, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QThreadPool, QSize
from PyQt5.QtGui import QImage, QFont, QIcon, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import pandas as pd
import geopy.distance
import webbrowser
from folium.plugins import MarkerCluster
from sklearn.decomposition import PCA


class MineralDetectionThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list)

    def __init__(self, data_file_path):
        super().__init__()
        self.data_file_path = data_file_path

    def run(self):
        try:
            self.progress.emit(10)
            mineral_data = self.load_mineral_data(self.data_file_path)

            self.progress.emit(50)
            detected_minerals = self.detect_minerals(mineral_data)

            self.progress.emit(100)
            self.result.emit(detected_minerals)
        except Exception as e:
            self.progress.emit(0)
            self.result.emit([])
            raise e

    def load_mineral_data(self, file_path):
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".tif") or file_path.endswith(".tiff"):
            with rasterio.open(file_path) as src:
                return src.read(1)
        else:
            raise ValueError("Unsupported file format!")

    def detect_minerals(self, mineral_data):
        detected_minerals = []

        if isinstance(mineral_data, pd.DataFrame):
            data_matrix = mineral_data.values
        else:
            data_matrix = mineral_data

        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data_matrix)

        for i, row in enumerate(reduced_data):
            if row[0] > 1 and row[1] > 1:
                detected_minerals.append({"index": i, "pca_values": row.tolist()})

        return detected_minerals
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
        self.load_data_button.setIconSize(QSize(30, 30))
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
        self.export_button.setIconSize(QSize(30, 30))
        self.button_layout.addWidget(self.export_button)

        self.visualize_button = QPushButton("Visualize Hyperspectral Minerals")
        self.visualize_button.clicked.connect(self.visualize_hyperspectral)
        self.visualize_button.setEnabled(False)
        self.visualize_button.setStyleSheet(self.button_style())
        self.visualize_button.setIcon(QIcon('visualize_icon.png'))
        self.visualize_button.setIconSize(QSize(30, 30))
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
class LiveWatchThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(pd.DataFrame)

    def __init__(self, gps_data_file, camera_active=True):
        super().__init__()
        self.gps_data_file = gps_data_file
        self.camera_active = camera_active

    def run(self):
        try:
            self.progress.emit(10)
            gps_data = self.load_gps_data(self.gps_data_file)

            if self.camera_active:
                self.progress.emit(50)
                self.process_camera_feed()

            self.progress.emit(100)
            self.result.emit(gps_data)
        except Exception as e:
            self.progress.emit(0)
            self.result.emit(pd.DataFrame())
            QMessageBox.critical(None, "Error", f"An error occurred: {e}")

    def load_gps_data(self, file_path):
        return pd.read_csv(file_path)

    def process_camera_feed(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not access the camera.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
class LiveWatchLocationsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Watch Locations (Benue State)")
        self.setGeometry(200, 200, 1200, 800)
        self.setStyleSheet("background-color: #f0f0f0;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Header Section
        header_label = QLabel("\U0001F4CD Live Watch Locations")
        header_label.setFont(QFont("Arial", 24, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("color: #003366;")
        self.layout.addWidget(header_label)

        # Instruction Section
        instruction_label = QLabel("Select GPS Data for Benue State and Camera Detection Option:")
        instruction_label.setFont(QFont("Arial", 14))
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("color: #333333;")
        self.layout.addWidget(instruction_label)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(divider)

        # Buttons and Progress Bar Section
        button_layout = QHBoxLayout()

        self.load_data_button = QPushButton("Load GPS Data (CSV)")
        self.load_data_button.setFont(QFont("Arial", 12))
        self.load_data_button.setStyleSheet("background-color: #0066cc; color: white; padding: 10px; border-radius: 5px;")
        self.load_data_button.clicked.connect(self.load_gps_data)
        button_layout.addWidget(self.load_data_button)

        self.camera_checkbox = QCheckBox("Enable Camera Detection")
        self.camera_checkbox.setFont(QFont("Arial", 12))
        self.camera_checkbox.setStyleSheet("color: #003366;")
        button_layout.addWidget(self.camera_checkbox)

        self.layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { border: 2px solid #003366; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #0066cc; }")
        self.layout.addWidget(self.progress_bar)

        # Results Section
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 12))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        # Action Buttons Section
        action_buttons_layout = QHBoxLayout()

        self.show_map_button = QPushButton("Show GPS Locations on Map")
        self.show_map_button.setFont(QFont("Arial", 12))
        self.show_map_button.setStyleSheet("background-color: #28a745; color: white; padding: 10px; border-radius: 5px;")
        self.show_map_button.clicked.connect(self.show_map)
        self.show_map_button.setEnabled(False)
        action_buttons_layout.addWidget(self.show_map_button)

        self.visualize_map_button = QPushButton("Visualize Detected Features")
        self.visualize_map_button.setFont(QFont("Arial", 12))
        self.visualize_map_button.setStyleSheet("background-color: #17a2b8; color: white; padding: 10px; border-radius: 5px;")
        self.visualize_map_button.clicked.connect(self.visualize_features)
        self.visualize_map_button.setEnabled(False)
        action_buttons_layout.addWidget(self.visualize_map_button)

        self.layout.addLayout(action_buttons_layout)

        self.gps_data = None

    def load_gps_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open GPS Data", "", "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            self.progress_bar.setValue(10)

            self.live_watch_thread = LiveWatchThread(file_path, self.camera_checkbox.isChecked())
            self.live_watch_thread.progress.connect(self.update_progress_bar)
            self.live_watch_thread.result.connect(self.handle_gps_data)
            self.live_watch_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")
            self.progress_bar.setValue(0)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def handle_gps_data(self, gps_data):
        if not gps_data.empty:
            self.gps_data = gps_data
            self.result_label.setText(f"Loaded {len(gps_data)} GPS locations.")
            self.show_map_button.setEnabled(True)
            self.visualize_map_button.setEnabled(True)

            start_coords = gps_data.iloc[0][['latitude', 'longitude']].values
            end_coords = gps_data.iloc[1][['latitude', 'longitude']].values
            distance = geopy.distance.distance(start_coords, end_coords).km
            self.result_label.setText(f"Distance between first two points: {distance:.2f} km")
        else:
            self.result_label.setText("No GPS data loaded.")
            self.progress_bar.setValue(0)

    def show_map(self):
        if self.gps_data is None:
            return

        avg_lat = self.gps_data['latitude'].mean()
        avg_lon = self.gps_data['longitude'].mean()
        folium_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=10, tiles="OpenStreetMap")

        marker_cluster = MarkerCluster().add_to(folium_map)
        for _, row in self.gps_data.iterrows():
            folium.Marker([row['latitude'], row['longitude']], popup=row['location_name']).add_to(marker_cluster)

        map_path = "benue_state_map.html"
        folium_map.save(map_path)
        QMessageBox.information(self, "Map Created", f"Map has been created at {map_path}.")
        webbrowser.open(map_path)

    def visualize_features(self):
        map_center = [7.7285, 8.6010]
        mymap = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")

        folium.Marker(
            [7.7285, 8.6010],
            popup="Flood-Prone Area",
            icon=folium.Icon(color='red', icon='cloud'),
        ).add_to(mymap)

        folium.Marker(
            [7.7300, 8.6020],
            popup="Healthy Crops",
            icon=folium.Icon(color='green', icon='leaf'),
        ).add_to(mymap)

        folium.Marker(
            [7.7305, 8.6030],
            popup="Mineral-Rich Area",
            icon=folium.Icon(color='blue', icon='info-sign'),
        ).add_to(mymap)

        folium.Marker(
            [7.7320, 8.6040],
            popup="Rainy Season Detected",
            icon=folium.Icon(color='purple', icon='tint'),
        ).add_to(mymap)

        folium.Marker(
            [7.7315, 8.6050],
            popup="Harvest-Ready Area",
            icon=folium.Icon(color='orange', icon='ok-sign'),
        ).add_to(mymap)

        map_path = "benue_detection_map.html"
        mymap.save(map_path)
        QMessageBox.information(self, "Feature Map Created", f"Feature map has been created at {map_path}.")
        webbrowser.open(map_path)

    def closeEvent(self, event):
        event.accept()
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



class HomePage(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Advanced GIS")
        self.setGeometry(100, 100, 800, 600)

        # Central widget setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout for the main window
        layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Welcome to the Agriculture and Environment App Suite")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #4CAF50; margin: 20px;")
        layout.addWidget(title_label)

        # Tabs setup
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 2px solid #4CAF50; border-radius: 6px; }
            QTabBar::tab { background: #E8F5E9; border: 1px solid #4CAF50; padding: 10px; width: 100px; }
            QTabBar::tab:selected { background: #A5D6A7; color: white; font-weight: bold; }
            """)

        # Adding tabs
        self.live_app = LiveWatchLocationsApp()
        self.tabs.addTab(self.live_app, "Live Watch")

        self.seasons_app = SeasonDetectionApp()
        self.tabs.addTab(self.seasons_app, "Seasons")

        # Add the Flood Detection App tab
        self.flood_detection_app = FloodDetectionApp()
        self.tabs.addTab(self.flood_detection_app, "Flood")
        # Add the Harvesting Time App tab
        self.harvesting_app = HarvestingTimeApp()
        self.tabs.addTab(self.harvesting_app, "Harvesting Time")

        self.minerals_app = MineralDetectionApp()
        self.tabs.addTab(self.minerals_app, "Minerals")
        # Add the Crop Detection App tab
        self.crop_detection_app = CropDetectionApp()
        self.tabs.addTab(self.crop_detection_app, "Crop")

        layout.addWidget(self.tabs)
        self.central_widget.setLayout(layout)

    def create_tab(self, title):
        """Creates individual tabs with a placeholder label."""
        tab = QWidget()
        layout = QVBoxLayout()

        label = QLabel(f"{title} Module Coming Soon!")
        label.setFont(QFont("Arial", 14))
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #757575; padding: 20px;")

        # Adding a decorative frame
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("background-color: #F1F8E9; border: 2px solid #A5D6A7; border-radius: 10px; margin: 10px;")
        frame_layout = QVBoxLayout()
        frame_layout.addWidget(label)
        frame.setLayout(frame_layout)

        layout.addWidget(frame)
        tab.setLayout(layout)
        return tab


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = HomePage()
    window.show()

    sys.exit(app.exec_())
