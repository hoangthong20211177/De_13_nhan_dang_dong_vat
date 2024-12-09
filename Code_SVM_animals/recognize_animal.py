import cv2
import numpy as np
import joblib
from skimage.feature import hog
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO  # Import YOLOv8

# Hàm tiền xử lý ảnh (chuyển đổi kích thước)
def preprocess_image(image, size=(64, 128)):
    image = cv2.resize(image, size)  # Đổi kích thước ảnh về kích thước chuẩn
    return image

# Hàm trích xuất đặc trưng HOG từ ảnh
def extract_features_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang ảnh xám
    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return fd

# Hàm nhận diện động vật và vẽ bounding box sử dụng YOLOv8
def detect_and_draw_bounding_box(image_path, model):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể mở ảnh: {image_path}")

    # Sử dụng YOLOv8 để phát hiện đối tượng và lấy bounding box
    yolo_model = YOLO("yolov8n.pt")  # Tải mô hình YOLOv8 pre-trained (phiên bản nhỏ cho việc suy luận nhanh)
    results = yolo_model(image)  # Phát hiện đối tượng trong ảnh

    # Duyệt qua các bounding box mà YOLOv8 phát hiện được
    for box in results[0].boxes:  # Thay đổi ở đây: truy cập qua results[0].boxes
        x1, y1, x2, y2 = box.xyxy[0]  # Lấy tọa độ bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Vẽ bounding box

        # Cắt ảnh theo bounding box và thay đổi kích thước về 64x128
        cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
        cropped_image_resized = preprocess_image(cropped_image)  # Thay đổi kích thước ảnh cắt
        
        # Trích xuất đặc trưng HOG từ phần ảnh đã cắt
        fd = extract_features_from_image(cropped_image_resized)  # Trích xuất đặc trưng HOG từ ảnh đã thay đổi kích thước
        animal_name = model.predict([fd])[0]  # Dự đoán tên động vật từ mô hình SVM

        # Danh sách động vật ăn cỏ và ăn thịt
        herbivores = [
            'antelope', 'bison', 'cow', 'deer', 'elephant', 'goat', 'hippopotamus', 'horse', 'kangaroo', 'koala',
            'leopard', 'mouse', 'okapi', 'ox', 'panda', 'parrot', 'pig', 'reindeer', 'rhinoceros', 'sheep', 'zebra'
        ]
        carnivores = [
            'badger', 'bear', 'cat', 'coyote', 'crab', 'dog', 'dolphin', 'fox', 'gorilla', 'hyena', 'jellyfish',
            'lion', 'lizard', 'lobster', 'otter', 'owl', 'penguin', 'raccoon', 'rat', 'shark', 'snake', 'sparrow',
            'squid', 'tiger', 'wolf', 'wombat'
        ]

        # Xác định loại động vật (ăn cỏ hay ăn thịt)
        if animal_name in herbivores:
            diet_type = "Herbivore"
        elif animal_name in carnivores:
            diet_type = "Carnivore"
        else:
            diet_type = "Unknown"

        # Kích thước font nhỏ hơn một chút
        font_scale = 0.6  # Giảm kích thước chữ xuống một chút
        thickness = 2  # Độ dày chữ nhỏ hơn

        # Vẽ loại ăn vào trong bounding box
        cv2.putText(image, f"Diet: {diet_type}", (int(x1) + 5, int(y1) + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        # Vẽ tên động vật vào trong bounding box, một chút dưới dòng "Diet"
        cv2.putText(image, f"{animal_name}", (int(x1) + 5, int(y1) + 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    # Tạo cửa sổ có thể thay đổi kích thước
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", image)  # Hiển thị ảnh kết quả
    cv2.waitKey(0)  # Đợi người dùng nhấn phím
    cv2.destroyAllWindows()  # Đóng cửa sổ khi nhấn phím

# Hàm để mở hộp thoại chọn tệp ảnh
def select_image_file():
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    return file_path

if __name__ == "__main__":
    # Tải mô hình đã huấn luyện
    model = joblib.load('svm_model.pkl')

    # Chọn ảnh từ máy tính
    image_path = select_image_file()
    if image_path:
        # Nhận diện và vẽ bounding box
        detect_and_draw_bounding_box(image_path, model)
    else:
        print("Không có ảnh nào được chọn.")
