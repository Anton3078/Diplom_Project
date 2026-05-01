import struct
import binascii
import sys
import os
import cv2
import socket
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QPushButton, QMainWindow, QApplication,
    QHBoxLayout, QVBoxLayout, QWidget, QLabel, QFileDialog,
    QGraphicsDropShadowEffect
)
from PyQt6.QtGui import QIcon, QFont, QPixmap, QImage, QFontDatabase, QColor

# ───────────────────────────────────────────────────────────
# Константы протокола (должны совпадать с сервером)
# ───────────────────────────────────────────────────────────
FACE_RESULT_SIZE = 4 * 4 + 4 + 4 + 512 * 4   # = 2072 байта
# float x1,y1,x2,y2 (4*4) + float confidence (4) + uint32 class_id (4) + float embedding[512] (512*4)

def parse_face_results(data, count):
    """
    Парсит бинарный ответ сервера.
    Возвращает список кортежей (x1, y1, x2, y2, class_id).
    """
    results = []
    for i in range(count):
        start = i * FACE_RESULT_SIZE
        chunk = data[start:start + FACE_RESULT_SIZE]
        if len(chunk) < 24:  # минимально должны быть координаты и class_id
            print(f"Ошибка: недостаточно данных для результата {i}")
            continue
        try:
            # 4 float (x1,y1,x2,y2), 1 float confidence, 1 unsigned int class_id
            x1, y1, x2, y2, confidence, class_id = struct.unpack('4f f I', chunk[:24])
            results.append((int(x1), int(y1), int(x2), int(y2), class_id))
        except struct.error as e:
            print(f"Ошибка распаковки результата {i}: {e}")
            continue
    return results

def receive_full(sock, size):
    """Читает ровно size байт из сокета."""
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            raise ConnectionError("Соединение закрыто сервером")
        data += packet
    return data

class MainWindow(QMainWindow):
    def __init__(self, host: str, port: int):
        super().__init__()
        self.coords = [] 
        
        self.host = host
        self.port = port 
        self.client_socket = None

        # ===== НАСТРОЙКИ ОКНА =====
        self.setWindowTitle("Face Detection App")
        self.setGeometry(200, 2, 480, 720)
        self.setFixedSize(480, 720)

        # Убираем рамку окна
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Прозрачный фон для скругления
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Тень
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(10)
        shadow.setColor(QColor(0, 0, 0, 80)) 
        self.setGraphicsEffect(shadow)

        # Шрифты
        base_dir = os.path.dirname(os.path.abspath(__file__))
        michroma_path = os.path.join(base_dir, "..", "font", "Michroma-Regular.ttf")
        michroma_id = QFontDatabase.addApplicationFont(michroma_path)
        if michroma_id != -1:
            self.michroma = QFontDatabase.applicationFontFamilies(michroma_id)
        else:
            self.michroma = ["Arial"]

        # Элементы управления окном
        self.close_btn = QPushButton(self)
        self.hide_btn = QPushButton(self)

        # Название приложения
        self.name_app = QLabel("Face Detection App", self)

        # Линия под названием
        self.underline_name = QLabel("", self)
        self.underline_name.setFixedSize(250, 2)
        self.underline_name.setPixmap(QPixmap(os.path.join(base_dir, "..", "img", "Line4.png")))
        self.underline_name.setScaledContents(True)
        self.underline_name.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Кнопка распознавания
        self.recognize_btn = QPushButton("Recognize", self)

        # Картинка
        self.current_img_path = None
        self.current_img = QLabel("", self)
        self.img = None

        # Layouts
        self.main_layout = QVBoxLayout()
        self.top_layout = QHBoxLayout()
        self.center_layout = QVBoxLayout()
        self.name_layout = QVBoxLayout()  

        self.center_widget = QWidget()

        self.initUI()

    def initUI(self):
        # Центральный виджет с белым фоном и скруглением
        self.center_widget.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border-radius: 20px;
            }
        """)

        # Верхняя панель
        self.name_app.setFont(QFont(self.michroma[0], 14))
        self.name_app.setStyleSheet("color: #000000; font-weight: bold; ")
        self.name_app.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Кнопка закрыть (красная)
        self.close_btn.setFixedSize(15, 15)
        self.close_btn.setStyleSheet(
            "QPushButton { background-color: #FF5F57; border-radius: 6px; } "
            "QPushButton:hover { background-color: #FF5F57; border-radius: 6px; } "
        )
        self.close_btn.clicked.connect(self.close)

        # Кнопка свернуть (зелёная)
        self.hide_btn.setFixedSize(15, 15)
        self.hide_btn.setStyleSheet(
            "QPushButton { background-color: #28C840; border-radius: 6px; } "
            "QPushButton:hover { background-color: #28C840; border-radius: 6px; } "
        )
        self.hide_btn.clicked.connect(self.showMinimized)

        # Layout для названия + линии
        self.name_layout.addWidget(self.name_app)
        self.name_layout.addWidget(self.underline_name, alignment=Qt.AlignmentFlag.AlignLeft)
        self.name_layout.setSpacing(2)
        self.name_layout.setContentsMargins(0, 0, 0, 0)

        # Верхняя панель
        self.top_layout.addLayout(self.name_layout)
        self.top_layout.addStretch()
        self.top_layout.addWidget(self.hide_btn)
        self.top_layout.addWidget(self.close_btn)
        self.top_layout.setSpacing(8)
        self.top_layout.setContentsMargins(15, 10, 15, 10)

        # Контейнер для изображения
        self.image_container = QWidget()
        self.image_container.setStyleSheet(
            "QWidget { background-color: #FFFFFF; border: 2px solid #000000; "
            "border-radius: 12px; } "
        )
        self.image_container.setFixedSize(440, 520)

        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(8, 8, 8, 8)
        self.image_container.setLayout(container_layout)

        self.current_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_img.setStyleSheet("border-radius: 8px; ")
        self.current_img.setPixmap(QPixmap(os.path.join("..", "img", "NCE &CE.png")))
        self.current_img.setScaledContents(True)

        container_layout.addWidget(self.current_img)

        # Кнопка Recognize
        self.recognize_btn.setFixedSize(440, 60)
        self.recognize_btn.setFont(QFont(self.michroma[0], 24))
        self.recognize_btn.setStyleSheet(
            "QPushButton { "
            "background-color: #E0E0E0; "
            "border-radius: 30px; "
            "color: #000000; "
            "border: none; "
            "} "
            "QPushButton:hover { "
            "background-color: #D0D0D0; "
            "border-radius: 30px; "
            "} "
            "QPushButton:pressed { "
            "background-color: #C0C0C0; "
            "border-radius: 30px; "
            "} "
        )
        self.recognize_btn.clicked.connect(self.recognize)

        # Центральная область
        self.center_layout.addWidget(self.image_container)
        self.center_layout.addSpacing(20)
        self.center_layout.addWidget(self.recognize_btn)
        self.center_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.center_layout.setContentsMargins(20, 20, 20, 30)
        self.center_layout.setSpacing(0)

        # Основной layout
        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addLayout(self.center_layout)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.center_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.center_widget)
    
    def hex_dump(self, data, label=""):
        """Отладочный вывод первых байт."""
        print(f"[CLIENT DEBUG] {label} ({len(data)} bytes):")
        print(binascii.hexlify(data[:64]).decode())

    def recognize(self):
        """
        Загружает изображение, отправляет на сервер, получает координаты и метки,
        отрисовывает рамки с class_id.
        """
        try:
            # ===== ШАГ 1: ЗАГРУЗКА ИЗОБРАЖЕНИЯ =====
            filePath, _ = QFileDialog.getOpenFileName(
                self, "Select picture", "", "*.png *.bmp *.jpg"
            )
            if not filePath:
                print("Файл не выбран")
                return

            self.current_img_path = filePath
            self.current_img.setPixmap(QPixmap(self.current_img_path))

            self.img = cv2.imread(self.current_img_path)
            if self.img is None:
                raise ValueError("Не удалось загрузить изображение")

            original_h, original_w = self.img.shape[:2]
            if original_h == 0 or original_w == 0:
                print("Изображение имеет нулевой размер.")
                return
            target_size = (224, 224)
            self.img = cv2.resize(self.img, target_size, interpolation=cv2.INTER_AREA)

            # ===== ШАГ 2: ОТПРАВКА НА СЕРВЕР =====
            success, encoded_image = cv2.imencode('.png', self.img)
            if not success:
                raise Warning("Ошибка кодирования изображения")
            image_bytes = encoded_image.tobytes()

            self.client_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            server_address = ('::1', 8080, 0, 0)
            self.client_socket.connect(server_address)
            
            # Отправляем размер (4 байта big-endian) и данные
            self.client_socket.send(struct.pack('!I', len(image_bytes)))
            self.client_socket.sendall(image_bytes)

            # --- Получение ответа ---
            face_count_data = receive_full(self.client_socket, 4)
            self.hex_dump(face_count_data, "Raw face_count bytes")
            try:
                count_le = struct.unpack('<i', face_count_data)[0]
                face_count = count_le
            except Exception as e:
                print(f"[CLIENT DEBUG] Failed to unpack count: {e}")
                return
            print(f"Получено количество лиц: {face_count}")

            results_list = []
            if face_count > 0:
                total_expected_size = face_count * FACE_RESULT_SIZE
                try:
                    raw_face_data = receive_full(self.client_socket, total_expected_size)
                    self.hex_dump(raw_face_data, f"Raw face data ({face_count} faces)")
                    results_list = parse_face_results(raw_face_data, face_count)
                except ConnectionError as e:
                    print(f"Ошибка получения данных: {e}")
                    return

            # Закрываем сокет
            self.client_socket.close()
            self.client_socket = None

            # ===== ШАГ 3: ОТРИСОВКА =====
            if results_list:
                img_with_boxes = self.img.copy()
                scale_x = original_w / target_size[0]
                scale_y = original_h / target_size[1]

                for (x1_raw, y1_raw, x2_raw, y2_raw, class_id) in results_list:
                    x1_scaled = int(x1_raw * scale_x)
                    y1_scaled = int(y1_raw * scale_y)
                    x2_scaled = int(x2_raw * scale_x)
                    y2_scaled = int(y2_raw * scale_y)

                    cv2.rectangle(img_with_boxes, (x1_scaled, y1_scaled),
                                  (x2_scaled, y2_scaled), (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"ID: {class_id}",
                                (x1_scaled, max(20, y1_scaled - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print(f"Отмечено лицо: ({x1_scaled}, {y1_scaled}, {x2_scaled}, {y2_scaled}) "
                          f"class_id={class_id}")

                pixmap = self.convert_cv_to_pixmap(img_with_boxes)
                self.current_img.setPixmap(
                    pixmap.scaled(
                        self.current_img.width() - 16,
                        self.current_img.height() - 16,
                        Qt.AspectRatioMode.KeepAspectRatio
                    )
                )
                print(f"Обнаружено и отмечено {len(results_list)} лиц(а) с метками.")
            else:
                print("Лица не обнаружены или ошибка обработки ответа сервера.")

        except Exception as ex:
            print(f"Error in recognize: {ex}")
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None

    def convert_cv_to_pixmap(self, cv_img):
        """Конвертирует OpenCV изображение в QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qt_image)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
            event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow("::1", 8080)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
