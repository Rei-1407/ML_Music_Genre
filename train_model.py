import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time # MỚI: Thêm để theo dõi thời gian

# Thay thế bằng đường dẫn GTZAN dataset của bạn
DATASET_PATH = "genres_original" 

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

X, y = [], []
start_time = time.time() # MỚI: Bắt đầu đếm thời gian

print("Bắt đầu trích xuất đặc trưng...")

# MỚI: Định nghĩa thời lượng cho mỗi đoạn
SECONDS_PER_SEGMENT = 30

for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):
    if dirpath is not DATASET_PATH:
        # Lấy tên thể loại từ tên thư mục
        genre = os.path.basename(dirpath)
        print(f"Đang xử lý thể loại: {genre}")
        
        for f in filenames:
            if f.endswith(".wav"):
                file_path = os.path.join(dirpath, f)
                try:
                    # MỚI: Tải toàn bộ file audio, không giới hạn 30 giây
                    y_audio, sr = librosa.load(file_path)
                    
                    samples_per_segment = SECONDS_PER_SEGMENT * sr
                    
                    # MỚI: Lặp qua từng đoạn trong file audio
                    num_segments = int(len(y_audio) / samples_per_segment)
                    for s in range(num_segments):
                        start_sample = samples_per_segment * s
                        end_sample = start_sample + samples_per_segment
                        
                        segment = y_audio[start_sample:end_sample]
                        
                        # Trích xuất MFCC cho đoạn hiện tại
                        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
                        features = np.mean(mfcc.T, axis=0)
                        
                        X.append(features)
                        y.append(genre)

                except Exception as e:
                    print(f"Lỗi xử lý file {file_path}: {e}")
                    continue

X = np.array(X)
y = np.array(y)

print(f"\nTrích xuất hoàn tất. Tổng số mẫu: {len(X)}")
print(f"Thời gian trích xuất: {time.time() - start_time:.2f} giây")

# Chia dữ liệu để huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # MỚI: stratify=y để đảm bảo tỷ lệ các lớp là như nhau

# Huấn luyện mô hình
print("Bắt đầu huấn luyện mô hình RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Đánh giá mô hình
accuracy = model.score(X_test, y_test)
print(f"Độ chính xác của mô hình trên tập kiểm thử: {accuracy:.2%}")

# Lưu mô hình
os.makedirs("model", exist_ok=True) # MỚI: Tạo thư mục model nếu chưa có
joblib.dump(model, "model/genre_classifier.pkl")
print("✅ Mô hình đã được huấn luyện và lưu vào model/genre_classifier.pkl")