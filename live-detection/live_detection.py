import sys
import cv2
import numpy as np
import folium
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QWidget,
    QProgressBar, QMessageBox, QCheckBox, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont
import geopy.distance
import webbrowser
from folium.plugins import MarkerCluster

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveWatchLocationsApp()
    window.show()
    sys.exit(app.exec_())
