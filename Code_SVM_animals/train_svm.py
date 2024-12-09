import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

# Hàm tiền xử lý ảnh (chuyển đổi kích thước)
def preprocess_image(image_path, size=(64, 128)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể mở ảnh: {image_path}")
    image = cv2.resize(image, size)  # Đổi kích thước ảnh về kích thước chuẩn
    return image

# Hàm trích xuất đặc trưng HOG từ ảnh
def extract_features_from_image(image_path):
    image = preprocess_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang ảnh xám
    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return fd

# Hàm trích xuất đặc trưng từ tất cả ảnh trong thư mục
def extract_features_from_all_images(image_paths):
    features = []
    labels = []
    for image_path in image_paths:
        feature = extract_features_from_image(image_path)
        features.append(feature)
        label = image_path.split(os.path.sep)[-2]  # Giả sử tên thư mục là nhãn
        labels.append(label)
    return np.array(features), np.array(labels)

# Hàm huấn luyện mô hình SVM
def train_svm():
    # Danh sách đường dẫn đến các ảnh huấn luyện
    image_paths = []
    # Thêm đường dẫn tới các ảnh trong thư mục (cần chỉnh lại đường dẫn của bạn)
    for root, dirs, files in os.walk('animals'):
        for file in files:
            image_paths.append(os.path.join(root, file))

    features, labels = extract_features_from_all_images(image_paths)

    # Chia dữ liệu thành 80% huấn luyện và 20% kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Khởi tạo mô hình SVM
    model = svm.SVC(kernel='linear')

    # Bắt đầu tính thời gian huấn luyện
    start_time = time.time()

    # Huấn luyện mô hình SVM
    model.fit(X_train, y_train)

    # Thời gian huấn luyện
    training_time = time.time() - start_time
    print(f"Time taken for training: {training_time:.4f} seconds")

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # In ra độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # In ra các chỉ số đánh giá tổng hợp (precision, recall, F1-score trung bình)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Precision (macro average): {report['macro avg']['precision']:.4f}")
    print(f"Recall (macro average): {report['macro avg']['recall']:.4f}")
    print(f"F1-score (macro average): {report['macro avg']['f1-score']:.4f}")

    # Lưu mô hình đã huấn luyện
    joblib.dump(model, 'svm_model.pkl')
    print("Model has been trained and saved as 'svm_model.pkl'")

if __name__ == "__main__":
    train_svm()
