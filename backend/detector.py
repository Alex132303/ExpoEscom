import os
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from datetime import datetime
import logging
from typing import Dict, Optional, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_next_image_index(prefix: str) -> int:
    existing = [f for f in os.listdir("fotos-resultados") if f.startswith(prefix)]
    numbers = []
    for name in existing:
        parts = name.replace(".jpg", "").split("_")
        if parts[-1].isdigit():
            numbers.append(int(parts[-1]))
    return max(numbers, default=0) + 1

class VehicleDetector:
    def __init__(self, 
                 vehicle_model_path: str = 'modelos/yolov8m.pt',
                 plate_model_path: str = 'modelos/best_placa.pt'):
        """
        Inicializa el detector con modelos YOLO para vehículos y placas,
        y el OCR para reconocimiento de texto.
        
        Args:
            vehicle_model_path: Ruta al modelo YOLO para detección de vehículos
            plate_model_path: Ruta al modelo YOLO personalizado para placas
        """
        try:
            # Cargar modelos con verificación
            self.vehicle_model = self._load_model(vehicle_model_path)
            self.plate_model = self._load_model(plate_model_path)
            print("clases del modelo de placas:", self.plate_model.names)
            
            # Parámetros configurables
            self.vehicle_conf_thresh = 0.6  # Confianza mínima para vehículos
            self.plate_conf_thresh = 0.3    # Confianza mínima para placas
            self.target_img_size = (640, 640)  # Tamaño para redimensionamiento
            
            logger.info("Detector inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar detector: {str(e)}")
            raise

    def _load_model(self, model_path: str) -> YOLO:
        """Carga y verifica un modelo YOLO"""
        try:
            model = YOLO(model_path)
            if model is None:
                raise ValueError(f"No se pudo cargar el modelo en {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo {model_path}: {str(e)}")
            raise

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento de imagen para mejorar detección:
        - Redimensionamiento
        - Mejora de contraste
        - Normalización
        """
        try:
            # Convertir a RGB si es necesario
            if len(img.shape) == 2:  # Si es escala de grises
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:   # Si tiene canal alpha
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionar manteniendo aspect ratio
            h, w = img.shape[:2]
            scale = min(self.target_img_size[0] / w, self.target_img_size[1] / h)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Mejorar contraste (CLAHE)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return img
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {str(e)}")
            raise

    def _detect_objects(self, 
                       img: np.ndarray, 
                       model: YOLO, 
                       conf_thresh: float) -> Optional[np.ndarray]:
        """Ejecuta detección de objetos con YOLO"""
        try:
            results = model(img, conf=conf_thresh, verbose=False)
            if len(results[0].boxes) == 0:
                return None
            return results[0].boxes.xyxy.cpu().numpy()
        except Exception as e:
            logger.error(f"Error en detección: {str(e)}")
            return None

    def _extract_plate_text(self, plate_img: np.ndarray, index: int) -> str:
        try:
            # Aumentar resolución
            resized = cv2.resize(plate_img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

            # Escala de grises
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

            # Filtro para suavizar bordes y ruido
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Umbral adaptativo inverso (texto negro sobre fondo blanco)
            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                15, 12
            )

            # Dilatación ligera para engrosar texto
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dilated = cv2.dilate(binary, kernel, iterations=1)

            # Segunda dilatación (opcional pero ayuda)
            dilated = cv2.dilate(dilated, np.ones((1, 1), np.uint8), iterations=1)

            # Suavizado (reducción de ruido y bordes más redondos)
            blurred = cv2.GaussianBlur(dilated, (3, 3), 0)

            # Guardar imagen final para depurar
            index = get_next_image_index("tesseract_input")
            cv2.imwrite(f"fotos-resultados/debug_tesseract_input_{index}.jpg", blurred)

            # OCR
            config = "--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            text = pytesseract.image_to_string(blurred, config=config)

            print("Texto bruto detectado por Tesseract:", repr(text))
            return self._clean_plate_text(text)

        except Exception as e:
            print(f"ERROR OCR TESSERACT: {e}")
            return "ErrorOCR"


    def detect_vehicle(self, img_path: str) -> Dict:
        """
        Procesa una imagen y detecta vehículo + placa.
        
        Returns:
            Dict con:
            - vehicle_img: Imagen del vehículo recortada
            - plate_img: Imagen de la placa recortada
            - plate_text: Texto reconocido
            - timestamp: Fecha y hora de procesamiento
            - debug_info: Datos técnicos para diagnóstico
        """
        try:
            # Validar entrada
            if not isinstance(img_path, str):
                raise ValueError("La ruta de imagen debe ser un string")
            
            # Cargar imagen
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen en {img_path}")
            
            logger.info(f"Procesando imagen: {img_path}")

            # Preprocesamiento
            processed_img = self._preprocess_image(img)
            debug_info = {
                "original_size": img.shape,
                "processed_size": processed_img.shape
            }

            # Usar toda la imagen como área de búsqueda
            vehicle_img = processed_img
            debug_info["vehicle_box"] = (0, 0, vehicle_img.shape[1], vehicle_img.shape[0])


            # Detección de placa en el vehículo
            plate_boxes = self._detect_objects(
                vehicle_img,
                self.plate_model,
                self.plate_conf_thresh
            )
            
            if plate_boxes is None:
                return {"error": "No se detectó placa", "debug": debug_info}
            
            # Tomar la placa con mayor confianza
            first_plate = plate_boxes[0]
            px1, py1, px2, py2 = map(int, first_plate[:4])

            # 🔧 Ajustar márgenes (recorte más fino)
            margin = 5
            px1 = max(0, px1 + margin)
            py1 = max(0, py1 + margin)
            px2 = min(vehicle_img.shape[1], px2 - margin)
            py2 = min(vehicle_img.shape[0], py2 - margin)

            # Recortar la placa
            plate_img = vehicle_img[py1:py2, px1:px2]

            index = get_next_image_index("placa_recortada")
            cv2.imwrite(f"fotos-resultados/debug_placa_recortada_{index}.jpg", plate_img)

            
            debug_info["plate_box"] = (px1, py1, px2, py2)

            # Reconocimiento de texto en placa
            plate_text = self._extract_plate_text(plate_img, index)
            
            # Post-procesamiento del texto (formato mexicano)
            plate_text = self._clean_plate_text(plate_text)
            
            return {
                "vehicle_img": vehicle_img,
                "plate_img": plate_img,
                "plate_text": plate_text,
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "debug_info": debug_info
            }
            
        except Exception as e:
            logger.error(f"Error en detect_vehicle: {str(e)}")
            return {"error": f"Error de procesamiento: {str(e)}"}


    def _clean_plate_text(self, text: str) -> str:
        """Limpia y formatea el texto de la placa según estándares mexicanos"""
        # Eliminar caracteres no válidos
        valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        cleaned = ''.join(c for c in text.upper() if c in valid_chars)
        
        # Formatear según patrones comunes en México:
        # AAA-1234 o 123-ABC-45
        if len(cleaned) >= 6:
            if cleaned[:3].isalpha() and cleaned[3:].isdigit():
                return f"{cleaned[:3]}-{cleaned[3:]}"
            elif cleaned[:3].isdigit() and cleaned[3:6].isalpha():
                return f"{cleaned[:3]}-{cleaned[3:6]}-{cleaned[6:]}" if len(cleaned) > 6 else f"{cleaned[:3]}-{cleaned[3:]}"
        
        return cleaned if cleaned else "No reconocido"

    def visualize_detection(self, img: np.ndarray, results: Dict) -> np.ndarray:
        """Dibuja cajas y texto sobre la imagen para visualización"""
        try:
            # Dibujar caja del vehículo
            x1, y1, x2, y2 = results["debug_info"]["vehicle_box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibujar caja de la placa (coordenadas relativas al vehículo)
            px1, py1, px2, py2 = results["debug_info"]["plate_box"]
            abs_px1, abs_py1 = x1 + px1, y1 + py1
            abs_px2, abs_py2 = x1 + px2, y1 + py2
            cv2.rectangle(img, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 0, 255), 2)
            
            # Mostrar texto de placa
            cv2.putText(
                img, 
                results["plate_text"], 
                (abs_px1, abs_py1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            
            return img
        except Exception as e:
            logger.error(f"Error en visualización: {str(e)}")
            return img