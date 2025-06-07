import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
from collections import Counter # MỚI: Thêm thư viện Counter để đếm phiếu

# Load the trained model
model = joblib.load("model/genre_classifier.pkl")

# Genre labels
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Emojis for genres
genre_emojis = {
    'blues': '🎼', 'classical': '🎻', 'country': '🤠', 'disco': '🪩',
    'hiphop': '🎤', 'jazz': '🎷', 'metal': '🤘', 'pop': '🎧',
    'reggae': '🟢', 'rock': '🎸'
}

# Page config
st.set_page_config(page_title="Music Genre Classifier", layout="centered")

st.title("🎶 Music Genre Classifier")
st.markdown("Kéo & thả một file `.wav`, tôi sẽ đoán thể loại của nó bằng cách phân tích toàn bộ bài hát!") # MỚI: Cập nhật mô tả

# File uploader
file = st.file_uploader("🎵 Thả file `.wav` của bạn vào đây", type=["wav"], label_visibility="collapsed")

if file:
    with st.spinner("🔍 Phân tích toàn bộ audio... quá trình này có thể mất một lúc..."):
        # Audio preview
        st.audio(file, format='audio/wav')

        # =================================================================
        # MỚI: BẮT ĐẦU KHỐI LOGIC PHÂN TÍCH VÀ BỎ PHIẾU
        # =================================================================
        
        try:
            # Tải toàn bộ file audio
            y, sr = librosa.load(file, sr=None) # sr=None để giữ nguyên tần số lấy mẫu gốc

            # Các biến cho việc cắt đoạn
            segment_duration = 30  # 30 giây mỗi đoạn
            samples_per_segment = segment_duration * sr
            
            segment_predictions = []

            # Lặp qua từng đoạn 30 giây trong file audio
            for start_sample in range(0, len(y), samples_per_segment):
                end_sample = start_sample + samples_per_segment
                segment = y[start_sample:end_sample]

                # Chỉ xử lý các đoạn đủ dài
                if len(segment) >= sr * 10: # Yêu cầu đoạn dài ít nhất 10 giây
                    # Trích xuất đặc trưng MFCC cho đoạn này
                    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
                    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

                    # Dự đoán thể loại cho đoạn và lưu lại
                    prediction = model.predict(mfcc_mean)[0]
                    segment_predictions.append(prediction)

            # Nếu có kết quả dự đoán từ các đoạn
            if segment_predictions:
                # Bỏ phiếu: Đếm số lần xuất hiện của mỗi thể loại
                vote_counts = Counter(segment_predictions)
                # Sắp xếp theo số phiếu giảm dần
                top_votes = vote_counts.most_common()

                # Kết quả cuối cùng là thể loại có nhiều phiếu nhất
                final_genre = top_votes[0][0]
                
                # Tính toán "độ tự tin" dựa trên tỷ lệ phiếu bầu
                confidence = (top_votes[0][1] / len(segment_predictions)) * 100

                # Hiển thị kết quả
                emoji = genre_emojis.get(final_genre, '')
                st.markdown(f"### 🎯 Thể loại dự đoán: **{final_genre.upper()}** {emoji}")
                st.markdown(f"**📈 Độ tự tin (tỷ lệ phiếu bầu):** {confidence:.2f}%")
                
                # Hiển thị biểu đồ phân phối phiếu bầu
                st.markdown("### 📊 Phân phối phiếu bầu theo thể loại")
                vote_data = {
                    "Thể loại": [item[0].capitalize() for item in top_votes],
                    "Số phiếu": [item[1] for item in top_votes]
                }
                df_votes = pd.DataFrame(vote_data)
                st.bar_chart(df_votes.set_index("Thể loại"))

            else:
                st.warning("⚠️ Không thể phân tích file audio. File có thể quá ngắn.")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")

        # =================================================================
        # MỚI: KẾT THÚC KHỐI LOGIC
        # =================================================================

        # Optional waveform (giữ nguyên)
        with st.expander("📈 Hiện thị dạng sóng (toàn bộ bài hát)"):
            fig, ax = plt.subplots(figsize=(10, 2))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title('Waveform')
            st.pyplot(fig)