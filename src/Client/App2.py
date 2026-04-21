import struct, binascii
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

# --- Добавлены функции для разбора бинарного ответа ---
def parse_face_results(data, count):
    """
    Парсит байтовые данные в список координат (x1, y1, x2, y2).

    Args:
        data (bytes): Байтовые данные, содержащие face_result_t.
        count (int): Количество результатов для парсинга.

    Returns:
        list: Список кортежей (x1, y1, x2, y2).
    """
    face_coords = []
    offset = 0
    # Размер одного face_result_t без embedding: 4 float + 1 float + 1 uint32 = 24 байта
    # Если embedding не используется или передаётся отдельно, можно его игнорировать.
    # Проверим длину данных. Если она соответствует 24 байтам * count, будем использовать этот формат.
    expected_min_size_without_embedding = count * 24  # 4*4 + 4 + 4
    expected_full_size = count * (4*4 + 4 + 4 + 512) # 536 байт на face_result_t

    if len(data) == expected_full_size:
        # Используем полный формат, включая embedding
        for i in range(count):
            start = offset + i * 536 # 536 байт на каждый face_result_t
            chunk = data[start:start + 536]
            if len(chunk) < 24: # Проверяем, достаточно ли данных для основных полей
                print(f"Ошибка: недостаточно данных для парсинга результата {i}")
                continue
            # Распаковываем только основные поля: x1, y1, x2, y2, confidence, class_id
            # fmt = '4f f I 128f' -> '4f f I' (берём только первые 24 байта)
            try:
                x1, y1, x2, y2, confidence, class_id = struct.unpack('4f f I', chunk[:20])
            except struct.error as e:
                print(f"Ошибка распаковки полного формата для результата {i}: {e}")
                continue
            face_coords.append((int(x1), int(y1), int(x2), int(y2)))
            # print(f"Обработано лицо {i}: ({x1}, {y1}, {x2}, {y2}), conf={confidence}, class={class_id}") # Для отладки
    elif len(data) >= expected_min_size_without_embedding:
        # Используем упрощённый формат без embedding в потоке данных
        # Предположим, сервер отправляет только основные 24 байта на лицо после face_count
        for i in range(count):
            start = offset + i * 24
            chunk = data[start:start + 24]
            if len(chunk) < 24:
                print(f"Ошибка: недостаточно данных для парсинга результата {i} в упрощённом формате")
                continue
            try:
                x1, y1, x2, y2, confidence, class_id = struct.unpack('4f f I', chunk)
            except struct.error as e:
                print(f"Ошибка распаковки упрощенного формата для результата {i}: {e}")
                continue
            face_coords.append((int(x1), int(y1), int(x2), int(y2)))
            # print(f"Обработано лицо {i}: ({x1}, {y1}, {x2}, {y2}), conf={confidence}, class={class_id}") # Для отладки
    else:
        print(f"Ошибка: неожиданный размер данных ответа для {count} лиц. Длина: {len(data)}, ожидалось ~{expected_min_size_without_embedding} или ~{expected_full_size}")

    return face_coords

def receive_full(sock, size):
    """Получает ровно 'size' байт из сокета."""
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            raise ConnectionError("Соединение закрыто сервером")
        data += packet
    return data
# --- Конец добавленных функций ---

class MainWindow(QMainWindow):
    def __init__(self, host: str, port: int):
        super().__init__()
        # Координаты bbox изображения
        self.coords = [] # Не используется напрямую, результаты теперь в списке
        # Create socket
        self.host = host
        self.port = port # Исправление: добавлена строка для присвоения порта
        self.client_socket = None

        # ===== НАСТРОЙКИ ОКНА =====
        self.setWindowTitle("Face Detection App")
        self.setGeometry(200, 2, 480, 720)
        self.setFixedSize(480, 720)

        # Убираем рамку окна
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Прозрачный фон для скругления (обязательно!)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Добавляем тень для красоты
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(10)
        shadow.setColor(QColor(0, 0, 0, 80)) # Исправление: перемещена строка и добавлен префикс QColor
        self.setGraphicsEffect(shadow)

        # Загрузка шрифтов с проверкой
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Michroma шрифт
        # Исправлены пути: убраны лишние пробелы
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

        # ===== ЛИНИЯ ПОД НАЗВАНИЕМ (как в App.py) =====
        self.underline_name = QLabel("", self)
        self.underline_name.setFixedSize(250, 2)
        # Исправлены пути: убраны лишние пробелы
        self.underline_name.setPixmap(QPixmap(os.path.join(base_dir, "..", "img", "Line4.png")))
        self.underline_name.setScaledContents(True)
        self.underline_name.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Кнопка распознавания (единая кнопка)
        self.recognize_btn = QPushButton("Recognize", self)

        # Картинка
        self.current_img_path = None
        self.current_img = QLabel("", self)
        self.img = None

        # Основные layout'ы
        self.main_layout = QVBoxLayout()
        self.top_layout = QHBoxLayout()
        self.center_layout = QVBoxLayout()
        self.name_layout = QVBoxLayout()  # Layout для названия + линии

        self.center_widget = QWidget()

        self.initUI()

    def initUI(self):
        # ===== ЦЕНТРАЛЬНЫЙ ВИДЖЕТ С БЕЛЫМ ФОНОМ И СКРУГЛЕНИЕМ =====
        self.center_widget.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border-radius: 20px;
            }
        """)

        # ===== ВЕРХНЯЯ ПАНЕЛЬ =====
        # Название приложения
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

        # Кнопка свернуть (желтая)
        self.hide_btn.setFixedSize(15, 15)
        self.hide_btn.setStyleSheet(
            "QPushButton { background-color: #28C840; border-radius: 6px; } "
            "QPushButton:hover { background-color: #28C840; border-radius: 6px; } "
        )
        self.hide_btn.clicked.connect(self.showMinimized)

        # ===== Layout для названия + линии =====
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

        # ===== ЦЕНТРАЛЬНАЯ ОБЛАСТЬ С ИЗОБРАЖЕНИЕМ =====
        # Контейнер для изображения с рамкой
        self.image_container = QWidget()
        self.image_container.setStyleSheet(
            "QWidget { background-color: #FFFFFF; border: 2px solid #000000; "
            "border-radius: 12px; } "
        )
        self.image_container.setFixedSize(440, 520)

        # Layout для контейнера
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(8, 8, 8, 8)
        self.image_container.setLayout(container_layout)

        # Изображение
        self.current_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_img.setStyleSheet("border-radius: 8px; ")
        # Исправлены пути: убраны лишние пробелы
        self.current_img.setPixmap(QPixmap(os.path.join("..", "img", "NCE &CE.png")))
        self.current_img.setScaledContents(True)

        container_layout.addWidget(self.current_img)

        # ===== КНОПКА RECOGNIZE =====
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
        print(f"[CLIENT DEBUG] {label} ({len(data)} bytes):")
        print(binascii.hexlify(data[:64]).decode())

    def recognize(self):
        """
        Единая функция, которая объединяет функционал:
        1. download() - загрузка изображения
        2. send_to_server() - отправка на сервер и получение координат
        3. detection()   - отрисовка bounding box
        """
        try:
            # ===== ШАГ 1: ЗАГРУЗКА ИЗОБРАЖЕНИЯ =====
            filePath, _ = QFileDialog.getOpenFileName(
                self, "Select picture", "", "*.png *.bmp *.jpg"
            )

            if not filePath:
                print("Файл не выбран")
                return

            # Сохраняем путь и загружаем изображение
            self.current_img_path = filePath
            self.current_img.setPixmap(QPixmap(self.current_img_path))

            # Читаем изображение через OpenCV
            self.img = cv2.imread(self.current_img_path)
            if self.img is None:
                raise ValueError("Не удалось загрузить изображение")

            original_h, original_w = self.img.shape[:2]
            if original_h == 0 or original_w == 0:
                 print("Изображение имеет нулевую высоту или ширину.")
                 return
            # Изменяем размер для отправки на сервер (предполагается, что сервер ожидает 224x224)
            # !!! ВАЖНО: Убедитесь, что модель на сервере обучена на этом размере или масштабируйте результаты обратно
            target_size = (224, 224)
            self.img = cv2.resize(self.img, target_size, interpolation=cv2.INTER_AREA)


            # ===== ШАГ 2: ОТПРАВКА НА СЕРВЕР =====
            # Кодируем изображение в PNG
            success, encoded_image = cv2.imencode('.png', self.img)
            if not success:
                raise Warning("Ошибка кодирования изображения")

            image_bytes = encoded_image.tobytes()

            # Создаем сокет и подключаемся (AF_INET6 для IPv6)
            # Используем localhost для IPv6
            self.client_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            # Для localhost IPv6 адрес обычно ::1
            # Если сервер слушает на всех интерфейсах IPv6, можно указать '[::]' как хост в сервере
            # Здесь мы подключаемся к ::1 на порт 8080
            server_address = ('::1', 8080, 0, 0) # format: (host, port, flowinfo, scope_id)
            self.client_socket.connect(server_address)
            
            # --- Отправка ---
            # Отправляем размер данных (4 байта, unsigned int)
            self.client_socket.send(struct.pack('!I', len(image_bytes))) #  '!I' = big-endian unsigned int (4 bytes)
            # Отправляем само изображение
            self.client_socket.sendall(image_bytes)

            # --- Получение ---
            # 1. Получаем количество обнаруженных лиц (4 байта, signed int)
            face_count_data = receive_full(self.client_socket, 4)
            self.hex_dump(face_count_data, "Raw face_count bytes")
            try:
                count_be = struct.unpack('!i', face_count_data)[0]
                count_le = struct.unpack('<i', face_count_data)[0]
                print(f"[CLIENT DEBUG] Unpacked count (BigEndian): {count_be} | (LittleEndian): {count_le}")
                face_count = count_le
            except Exception as e:
                print(f"[CLIENT DEBUG] Failed to unpack count: {e}")
                return
            print(f"Получено количество лиц: {face_count}")

            results_list = []
            if face_count > 0:
                # 2. Получаем данные для каждого лица
                # Предположим, что сервер отправляет только основные 24 байта на лицо (x1,y1,x2,y2,conf,class_id)
                # и ли полные 536 байт. Попробуем получить минимально необходимое количество байт.
                # Лучше всего, чтобы сервер отправлял фиксированный формат, например, 24 байта на лицо.
                # Если он отправляет 536, нужно изменить расчёт здесь.
                expected_data_size_per_face = 24 # Попробуем сначала 24 байта
                total_expected_size = face_count * expected_data_size_per_face

                # Попробуем получить 24 * face_count байт
                try:
                    raw_face_data = receive_full(self.client_socket, total_expected_size)
                    self.hex_dump(raw_face_data, f"Raw face data (first {face_count} faces)")
                    results_list = parse_face_results(raw_face_data, face_count)
                except ConnectionError as e:
                    print(f"Ошибка получения основных данных: {e}. Пробуем получить полные 536 байта на лицо.")
                    # Если не получилось, попробуем получить полные 536 байт на лицо
                    expected_data_size_per_face_full = 536
                    total_expected_size_full = face_count * expected_data_size_per_face_full
                    raw_face_data_full = receive_full(self.client_socket, total_expected_size_full)
                    results_list = parse_face_results(raw_face_data_full, face_count)

            # 3. (Опционально) Получаем информацию об ошибке (1 байт), если сервер её отправляет
            # raw_error_info = self.client_socket.recv(1) # Если сервер всегда отправляет 1 байт ошибки в конце
            # if raw_error_info:
            #     err_type_code = struct.unpack('B', raw_error_info)[0] # 'B' = unsigned char (1 byte)
            #     print(f"Получен код ошибки: {err_type_code}")


            # Закрываем сокет
            self.client_socket.close()
            self.client_socket = None

            # ===== ШАГ 3: ОТРИСОВКА BOUNDING BOX =====
            if results_list:
                # Рисуем прямоугольники на ИСХОДНОМ изображении, масштабируя координаты
                # Координаты от сервера (предположительно) для размера target_size (224x224)
                # Масштабируем их обратно к размеру оригинального изображения
                img_with_boxes = self.img.copy() # Рисуем на изменённом изображении для отображения в GUI
                scale_x = original_w / target_size[0]
                scale_y = original_h / target_size[1]

                for (x1_raw, y1_raw, x2_raw, y2_raw) in results_list:
                    # Масштабирование
                    x1_scaled = int(x1_raw * scale_x)
                    y1_scaled = int(y1_raw * scale_y)
                    x2_scaled = int(x2_raw * scale_x)
                    y2_scaled = int(y2_raw * scale_y)

                    # --- ИСПРАВЛЕНИЕ: Используем масштабированные координаты для рисования на 224x224 ---
                    # Если хотим рисовать на оригинале, используем img_orig.copy() и scaled координаты
                    img_with_boxes = cv2.rectangle(
                        img_with_boxes,
                        (x1_scaled, y1_scaled), # <-- Используем масштабированные координаты
                        (x2_scaled, y2_scaled), # <-- Используем масштабированные координаты
                        thickness=2, # Толщина линии
                        color=(0, 255, 0) # Зелёный цвет
                     )
                    print(f"Отмечено лицо: ({x1_scaled}, {y1_scaled}, {x2_scaled}, {y2_scaled}) (на 224x224)")

                # Конвертируем в QPixmap для отображения
                pixmap = self.convert_cv_to_pixmap(img_with_boxes)
                self.current_img.setPixmap(
                    pixmap.scaled(
                        self.current_img.width() - 16,
                        self.current_img.height() - 16,
                        Qt.AspectRatioMode.KeepAspectRatio
                     )
                )
                print(f"Обнаружено и отмечено {len(results_list)} лиц(а) на изображении.")
            else:
                print("Лица не обнаружены или произошла ошибка обработки ответа сервера.")

        except Exception as ex:
            print(f"Error in recognize: {ex}")
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None # Убедимся, что сокет сброшен даже при исключении

    def convert_cv_to_pixmap(self, cv_img):
        """Конвертображения в QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line,
            QImage.Format.Format_RGB888
        )
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
    # host = socket.gethostname() # Не используется напрямую в соединении IPv6
    # port = 8080 # Используется напрямую в connect
    app = QApplication(sys.argv)
    window = MainWindow("::1", 8080) # Передаём IPv6 localhost
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
