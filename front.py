# frontend_mejorado.py
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
    QTextEdit, QFrame, QMessageBox  # <- agregado
)
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QDateTime, QTimer
import sys

class VehicleDetectorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚔 Sistema de Detección de Vehículos Robados")
        self.setGeometry(100, 100, 1200, 700)
        self.setup_ui()
        
        # Establecer color de fondo general
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(240, 245, 250))
        self.setPalette(palette)

    def setup_ui(self):
        # --- Widgets Principales ---
        # 1. Encabezado con fecha/hora
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            background-color: #2c3e50;
            border-radius: 10px;
            padding: 15px;
        """)
        
        self.label_datetime = QLabel()
        self.label_datetime.setAlignment(Qt.AlignCenter)
        self.label_datetime.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: white;
        """)
        
        header_layout = QVBoxLayout()
        header_layout.addWidget(QLabel("<h1 style='color: white; text-align: center;'>Detección de Vehículos Robados</h1>"))
        header_layout.addWidget(self.label_datetime)
        header_frame.setLayout(header_layout)
        
        # 2. Botón para cargar imagen
        self.btn_load = QPushButton("📤 Cargar Imagen")
        self.btn_load.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 12px 25px;
                font-weight: bold;
                background-color: #3498db;
                color: white;
                border-radius: 8px;
                min-width: 220px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        # 3. Área para imagen (izquierda)
        self.label_image = QLabel()
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setStyleSheet("""
            background-color: #ecf0f1;
            border: 2px dashed #bdc3c7;
            border-radius: 10px;
            min-width: 600px;
            min-height: 450px;
        """)
        
        # 4. Área para datos (derecha)
        self.text_results = QTextEdit()
        self.text_results.setReadOnly(True)
        self.text_results.setAlignment(Qt.AlignCenter)
        self.text_results.setStyleSheet("""
            QTextEdit {
                font-family: 'Segoe UI', Arial;
                font-size: 17px;
                background-color: white;
                border: 2px solid #dfe6e9;
                border-radius: 10px;
                padding: 30px;
            }
        """)
        self.text_results.setHtml(  # <- cambiado aquí
            """
            <div style='text-align: center; color: #7f8c8d; font-size: 16px;'>
                Aquí aparecerán:<br><br>
                • Modelo del vehículo<br>
                • Placa detectada<br>
                • Estatus (Robado/No robado)<br>
                • Hora de detección
            </div>
            """
        )

        # 5. Botón para generar reporte
        self.btn_report = QPushButton("🖨️ Generar Reporte PDF")
        self.btn_report.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 12px 25px;
                font-weight: bold;
                background-color: #e74c3c;
                color: white;
                border-radius: 8px;
                min-width: 220px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.btn_report.setEnabled(False)

        # --- Layouts ---
        # Layout horizontal para imagen + datos
        h_layout = QHBoxLayout()
        h_layout.setSpacing(20)
        h_layout.addWidget(self.label_image)
        h_layout.addWidget(self.text_results)

        # Layout vertical principal
        v_layout = QVBoxLayout()
        v_layout.setSpacing(20)
        v_layout.setContentsMargins(20, 20, 20, 20)
        v_layout.addWidget(header_frame)
        v_layout.addWidget(self.btn_load, alignment=Qt.AlignCenter)
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.btn_report, alignment=Qt.AlignCenter)

        container = QWidget()
        container.setLayout(v_layout)
        self.setCentralWidget(container)

        # --- Conexiones ---
        self.btn_load.clicked.connect(self.load_mock_image)
        self.btn_report.clicked.connect(self.generate_mock_report)
        
        # Temporizador para la hora
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_datetime)
        self.timer.start(1000)
        self.update_datetime()

    def update_datetime(self):
        """Actualiza la fecha/hora en tiempo real."""
        current_datetime = QDateTime.currentDateTime().toString("dd/MM/yyyy - hh:mm:ss ap")
        self.label_datetime.setText(f"⏱️ <b>Última actualización:</b> {current_datetime}")

    def load_mock_image(self):
        """Simula la carga de una imagen con datos de ejemplo."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "", "Imágenes (*.jpg *.png)"
        )
        
        if filename:
            # Mostrar imagen
            pixmap = QPixmap(filename)
            self.label_image.setPixmap(pixmap.scaled(550, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Simular datos de detección
            mock_data = """
            <div style='text-align: center;'>
                <p style='font-size: 18px; margin-bottom: 20px;'>
                    <span style='color: #2c3e50; font-weight: bold;'>Modelo:</span> 
                    <span style='color: #3498db;'>Nissan Sentra 2022</span>
                </p>
                
                <p style='font-size: 18px; margin-bottom: 20px;'>
                    <span style='color: #2c3e50; font-weight: bold;'>Placa:</span> 
                    <span style='color: #1E90FF; font-weight: bold;'>ABC-1234</span>
                </p>
                
                <p style='font-size: 20px; margin-bottom: 20px; color: #e74c3c; font-weight: bold;'>
                    ⚠️ VEHÍCULO ROBADO
                </p>
                
                <p style='font-size: 16px; color: #7f8c8d;'>
                    Detectado el 25/10/2023 - 02:45:30 PM
                </p>
            </div>
            """
            self.text_results.setHtml(mock_data)
            self.btn_report.setEnabled(True)

    def generate_mock_report(self):
        """Simula la generación de un reporte PDF."""
        QMessageBox.information(
            self, 
            "Reporte Generado", 
            "El reporte PDF se ha generado exitosamente.\n\n"
            "Contenido:\n"
            "- Modelo: Nissan Sentra 2022\n"
            "- Placa: ABC-1234\n"
            "- Estatus: Robado"
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Establecer estilo general
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = VehicleDetectorUI()
    window.show()
    sys.exit(app.exec_())
