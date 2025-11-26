🇻🇳 Xin chào, em là An, hiện tại đang tìm vị trí Intern để tốt nghiệp. Kính mong Anh/Chị nào thấy và có thể, xin cho em cơ hội thực tập. Em cảm ơn ạ. (@annguyen3528-telegram)

🇺🇸 Hello, I'm An, currently looking for an Intern position after graduation. I hope that if anyone sees this and can, please give me an internship opportunity. Thank you. (@annguyen3528-telegram)


ResNet-FAISS-Dedup
# ResNet-FAISS-Dedup: Semantic Deduplication Pipeline 🚀

*"Biến dữ liệu thô thành vàng ròng"*

Pipeline xử lý dữ liệu ảnh hiệu năng cao kết hợp Computer Vision truyền thống và Deep Learning để lọc ảnh kém chất lượng và loại bỏ ảnh trùng lặp dựa trên ngữ nghĩa.

## ⚠️ CẢNH BÁO PHẦN CỨNG & MÔI TRƯỜNG

**Hệ thống được tối ưu hóa cho:**
- **Hệ điều hành:** macOS Sequoia (Đã test)
- **Chipset:** Apple Silicon (M1/M2/M3/M4) với MPS - Metal Performance Shaders
- **Windows/Linux:** Chưa kiểm thử - cần điều chỉnh cấu hình WORKERS và FAISS

## 📊 Hiệu Năng Thực Tế

| Thông số | Giá trị | Ghi chú |
|----------|---------|---------|
| Thiết bị | Mac Mini M4 | 24GB RAM / 256GB SSD |
| Số lượng ảnh | 116,298 ảnh | Kích thước 640x640 |
| Tổng thời gian | ~46 phút | Bao gồm I/O, Hashing, AI, Graph |
| Tốc độ xử lý | ~42 ảnh/giây | Trung bình toàn trình |
| Kết quả lọc | Loại bỏ ~13.5% | ~15.800 ảnh rác & trùng lặp |

## 🛠 Kiến Trúc Hệ Thống (The Funnel Strategy)

### 1. Tầng 1: Bộ Lọc Chất Lượng 🧹
**Mục tiêu:** Loại bỏ ảnh "rác" - mờ, quá tối/sáng
- **Độ nét:** `cv2.Laplacian` (Variance of Laplacian)
- **Độ sáng:** `np.mean` trên ảnh Grayscale

### 2. Tầng 2: Bộ Lọc Thô (Hashing Deduplication) ⚡
**Mục tiêu:** Loại bỏ ảnh trùng lặp tuyệt đối
- **SHA-256:** Trùng khớp từng bit
- **Visual Hash (dHash/pHash):** Phát hiện ảnh resize/nén

### 3. Tầng 3: Bộ Lọc Tinh (Semantic Deduplication) 🧠
**Mục tiêu:** Xử lý trùng lặp ngữ nghĩa phức tạp
- **Feature Extraction:** ResNet50-IBN (2048 chiều)
- **Similarity Search:** FAISS (IndexFlatIP) + Cosine Similarity
- **Clustering Logic:** Graph Connected Components + "Vua & Thần dân"

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### Bước 1: Chuẩn bị môi trường
```bash
conda create -n dedup python=3.9
conda activate dedup
```

### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

**Lưu ý:** 
- macOS: sử dụng `faiss-cpu`
- Linux/Windows với NVIDIA GPU: cài `faiss-gpu`

### Bước 3: Tải Weights & Config
Đặt file pre-trained weights (`vehicleid_bot_R50-ibn.pth`) vào thư mục `configs/`

### Bước 4: Cấu hình & Chạy
```python
# Trong app.py
INPUT_FOLDER = '/path/to/your/dataset'
TEST = False  # Chuyển False để chạy thật
```

```bash
python app.py
```

## 📂 Dataset Tham Khảo

Dataset gồm 116.000+ ảnh xe cộ từ video giao thông thực tế với đa dạng điều kiện ánh sáng.

👉 **Tải tại:** [Link tới Kaggle Dataset - NẾU CÓ]

## 💡 Hỏi & Đáp (Technical Deep Dive)

### ❓ Tại sao dùng cv2.Laplacian mà không dùng AI để lọc ảnh mờ?
**Trả lời:** Tốc độ. Laplacian (0.001s/ảnh) hoạt động như "người gác cổng" cực nhanh. Dùng AI ở bước này là "dùng dao mổ trâu giết gà".

### ❓ Tại sao kết hợp Hashing và Deep Learning?
**Trả lời:** Hashing (nhanh) loại bỏ ảnh rác, giảm tải cho Deep Learning (hiểu ngữ nghĩa nhưng chậm hơn).

### ❓ Tại sao dùng FAISS mà không so sánh vector thủ công?
**Trả lời:** So sánh thủ công (O(N²)) với 100k ảnh → 5 tỷ phép tính. FAISS tìm kiếm trong vài giây.

### ❓ Tại sao WORKERS=0 trên Mac M4?
**Trả lời:** Tránh lỗi malloc do Multiprocessing của PyTorch trên macOS. Chip M4 đủ mạnh để GPU chạy 100% công suất dù đơn luồng.

## 🤝 Đóng góp

Mọi ý kiến đóng góp, báo lỗi hoặc Pull Request đều được hoan nghênh! Đặc biệt cần hỗ trợ chạy trên Windows và Linux.

## 📜 License

Dự án thuộc quyền sở hữu của [Tên Bạn]. Phân phối dưới giấy phép MIT License.

---

*"Biến dữ liệu thô thành vàng ròng" - ResNet-FAISS-Dedup*
