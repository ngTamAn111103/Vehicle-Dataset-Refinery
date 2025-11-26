# Thư viện
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
import cv2
import faiss
import numpy as np
import time
import os
import shutil
import random
from tqdm import tqdm
import hashlib
from typing import Dict, Set, List, Tuple, Optional
import imagehash
from PIL import Image
import networkx as nx 
import json
# --- FIX LỖI MALLOC TRÊN MAC ---
# Ép FAISS chỉ dùng 1 luồng để tránh xung đột bộ nhớ OpenMP
faiss.omp_set_num_threads(1) 
# ------------------------------

# Cấu hình
TEST = True
SAMPLE_SIZE = 500

# ___Ngưỡng lọc ảnh___
# Độ nét
BLUR_THRESHOLD = 90.0
# Độ tối
DARK_THRESHOLD = 30.0
# Độ sáng
BRIGHT_THRESHOLD = 220.0
# Ngưỡng giống nhau của Deep Learning
THRESHOLD_FAISS = 0.9

# ___Tốc độ___
BATCH_SIZE = 256
if TEST:
    WORKERS = 1
else:
    WORKERS = 0

# ___Đường dẫn___
# Folder cha chứa ảnh (Có triển khai đệ quy)
INPUT_FOLDER = '/Volumes/MICRON/raw_dataset_v1.1'
# Folder chứa tất cả kết quả đầu ra (Để nếu chạy trên đám mây, chỉ cần zip lại rồi tải về)
OUTPUT_BASE = '/Users/nguyentaman/Downloads/Vehicle-Dataset-Refinery/results'
# File weight (Kinh nghiệm) được cấu hình theo mô hình mạng
WEIGHTS_PATH = "configs/vehicle_weights.pth"
# File cấu hình thông số kỹ thuật (Mạng gì, size ảnh, số lượng class, ...)
CONFIG_FILE = "configs/vehicle_config.yaml"
# Đầu ra file báo cáo .html
REPORT_FILE = 'cleaning_report.html'
# Các đuôi file ảnh 
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp")
# Các folder phân loại
FOLDERS = ["blur", "dark", "bright", "duplicates", "similar", "output_features"]

# ___Thiết bị___
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def setup_folders():
    """
    Kiểm tra và xoá folder results cũ
    Tạo các folder [FOLDERS] mới
    """
    # Kiểm tra và xoá folder results cũ
    if os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)

    # Tạo lại toàn bộ folder results
    for folder in FOLDERS:
        os.makedirs(os.path.join(OUTPUT_BASE, folder), exist_ok=True)

def get_image_paths():
    """
    Lấy danh sách đường dẫn tất cả ảnh trong folder
    Có sử dụng đệ quy để quét toàn bộ các file con nếu có

    Returns:
        List[str]: Mỗi phần tử là đường dẫn tuyệt đối
    """

    # Khởi tạo danh sách đường dẫn tuyệt đối tới ảnh
    all_files = []

    # Kiểm tra folder input có tồn tại hay không
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder không tồn tại: {INPUT_FOLDER}")
        return []
    
    # Dùng os.walk để quét đệ quy (recursive) cả thư mục con
    for root, _, files in os.walk(INPUT_FOLDER):
        # Duyệt tất cả các file lấy được 
        for file in files:
            # lower() tên file, kiểm tra xem đuôi file có nằm trong IMAGE_EXTENSIONS không
            # Tên không được bắt đầu bằng '.'
            if file.lower().endswith(IMAGE_EXTENSIONS) and not file.startswith('.'):
                all_files.append(os.path.abspath(os.path.join(root, file)))

    # Nếu ở chế độ TEST -> Lấy ngẫu nhiên SAMPLE_SIZE
    if TEST and len(all_files) > SAMPLE_SIZE:
        print(f"⚠️ Chế độ TEST: Lấy ngẫu nhiên {SAMPLE_SIZE} ảnh.")
        return random.sample(all_files, SAMPLE_SIZE)
    return sorted(all_files)

def scan_and_filter_quality(all_images_path: List[str] = None) -> Tuple[List[str], List[Dict]]:
    """
    Quét toàn bộ danh sách ảnh và lọc bỏ các ảnh kém chất lượng (mờ, quá sáng, quá tối).

    Hàm này thực hiện các bước:
    1. Kiểm tra độ nét (Laplacian) và độ sáng trung bình.
    2. Nếu ảnh ĐẠT chuẩn: Giữ lại trong danh sách trả về.
    3. Nếu ảnh KHÔNG đạt chuẩn: Di chuyển (hoặc copy nếu TEST=True) sang thư mục phân loại 
       tương ứng (blur, dark, bright) và ghi log.

    Args:
        all_images_path (List[str]): Danh sách chứa đường dẫn tuyệt đối của các file ảnh.
                                     Mặc định là None.

    Returns:
        Tuple[List[str], List[Dict]]: Một tuple chứa 2 phần tử:
            - clean_images (List[str]): Danh sách đường dẫn các ảnh đạt chuẩn.
            - quality_log (List[Dict]): Danh sách nhật ký các ảnh bị loại. Mỗi phần tử là 
              một dict chứa keys: 'name', 'path', 'reason', 'score'.
    """
    # Khai báo danh sách ảnh đủ điều kiện
    clean_images = []
    # Khai báo LOGS để tạo file báo cáo
    quality_log = []
    
    print("\n🧹 [Bước 1] Kiểm tra chất lượng ảnh...")
    for filepath in tqdm(all_images_path, desc="Quality Check"):
        # Kiểm tra độ nét/sáng/tối
        _, status, score = check_image_quality(filepath)
        
        # Nếu ảnh đủ điều kiện
        if status == 'ok':
            clean_images.append(filepath)
        # Nếu ảnh không đủ điều kiện & ảnh không bị lỗi
        elif status != 'error':
            try:
                # Đường dẫn tới Folder đích
                target_folder = os.path.join(OUTPUT_BASE, status)
                # Tên file
                filename = os.path.basename(filepath)
                # Ghép tên file và folder đích
                target_path = os.path.join(target_folder, filename)
                # Hành động: Di chuyển ảnh qua folder phân loại
                shutil.move(filepath, target_path)

                # GHI LOG
                # filename: Tên ảnh
                # target_path: Đường dẫn đã bị di chuyển tới
                # score: kết hợp với status.upper() cho ra điểm số của điều kiện bị loại
                quality_log.append({'name': filename, 'path': target_path, 'reason': status.upper(), 'score': score})
            except Exception as e:
                print(f"Lỗi file {filepath}: {e}")
                
    return clean_images, quality_log

def check_image_quality(image_path: str = "") -> Tuple[str, str, float]:
    """
    Đánh giá chất lượng một bức ảnh dựa trên độ nét (Laplacian) và độ sáng trung bình.

    Hàm này đọc ảnh ở chế độ Grayscale để tối ưu hiệu năng.

    Args:
        image_path (str): Đường dẫn tuyệt đối tới file ảnh.

    Returns:
        Tuple[str, str, float]: Bộ 3 giá trị gồm:
            - image_path (str): Đường dẫn gốc (trả lại để tiện xử lý theo luồng).
            - status (str): Trạng thái phân loại, bao gồm:
                * 'ok': Ảnh đạt chuẩn.
                * 'blur': Ảnh bị mờ (dưới ngưỡng BLUR_THRESHOLD).
                * 'dark': Ảnh quá tối (dưới ngưỡng DARK_THRESHOLD).
                * 'bright': Ảnh quá sáng (trên ngưỡng BRIGHT_THRESHOLD).
                * 'error': Lỗi không đọc được file.
            - score (float): Điểm số tương ứng (Blur score hoặc Brightness mean).
    """
    try:
        # Đọc ảnh theo mode sáng tối (đen trắng)
        # Giá trị pixel từ từ 0 đến 255 để mô tả độ sáng tôi
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Trường hợp đọc file không được -> error
        if img_gray is None: 
            return image_path, 'error', 0.0

        # Tính toán độ blur/nét của ảnh bằng phương pháp Variance of Laplacian
        # 1. cv2.Laplacian: "Vẽ" lại các đường viền/cạnh của vật thể trong ảnh.
        # 2. cv2.CV_64F: Dùng số thực để giữ lại cả các viền Âm (viền tối), tránh mất dữ liệu.
        # 3. .var(): Tính phương sai (độ gắt). Giá trị càng cao -> Ảnh càng nhiều viền sắc nét -> Ảnh nét.
        blur_score = cv2.Laplacian(img_gray, cv2.CV_64F).var()

        # Độ nét thấp hơn ngưỡng BLUR_THRESHOLD -> blur
        if blur_score < BLUR_THRESHOLD: 
            return image_path, 'blur', blur_score

        # Độ sáng trung bình của ảnh
        mean_brightness = np.mean(img_gray)
        # Trả về nếu quá tối/quá sáng -> dark/bright
        if mean_brightness < DARK_THRESHOLD: 
            return image_path, 'dark', mean_brightness
        if mean_brightness > BRIGHT_THRESHOLD: 
            return image_path, 'bright', mean_brightness

        # Ảnh đủ điều kiện
        return image_path, 'ok', blur_score
    except: 
        return image_path, 'error', 0.0

def calculate_file_hash(filepath: str, method: str = 'sha256') -> str:
    """
    Tính toán mã băm (Hash) của một file để làm 'dấu vân tay số'.

    Hàm đọc file theo chế độ nhị phân (binary) và xử lý theo từng khối (chunk) 
    64KB để tối ưu bộ nhớ RAM, đảm bảo hoạt động tốt với cả file dung lượng lớn.

    Args:
        filepath (str): Đường dẫn tuyệt đối tới file cần tính hash.
        method (str, optional): Thuật toán băm cần dùng. 
                                Hỗ trợ 'sha256' (mặc định - an toàn cao) hoặc 'md5' (nhanh hơn).

    Returns:
        str: Chuỗi mã hash dạng Hexadecimal (ví dụ: '5d41402abc4b2a76...').
             Trả về None nếu không đọc được file.
    """
    hasher = hashlib.sha256() if method == 'sha256' else hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except:
        return None

def find_duplicates_by_hashing(image_paths: List[str]) -> Tuple[Set[str], List[Dict]]:
    """
    Quét và phát hiện ảnh trùng lặp bằng chiến lược Hashing đa tầng (Multi-stage Hashing).

    Hàm thực hiện lọc qua 2 giai đoạn nối tiếp:
    1. **Lọc trùng tuyệt đối (SHA-256):** Tìm các file giống hệt nhau từng bit (do copy-paste).
    2. **Lọc trùng nội dung (Visual Hash):**
       - **dHash (Difference Hash):** Nhạy với cấu trúc gradient, phát hiện ảnh bị resize/nén nhẹ.
       - **pHash (Perceptual Hash):** Nhạy với tần số ảnh, phát hiện ảnh bị biến đổi màu sắc/ánh sáng nhẹ.

    **Cơ chế an toàn:**
    Trong quá trình quét Visual Hash, hàm có logic kiểm tra sự tồn tại của file (os.path.exists) 
    để tránh lỗi "File ma" (tham chiếu đến một file trong từ điển hash nhưng file đó 
    đã bị xóa bởi một thuật toán hash khác ngay trước đó).

    Args:
        image_paths (List[str]): Danh sách chứa đường dẫn tuyệt đối của các ảnh đầu vào 
                                 (đã qua bước lọc chất lượng).

    Returns:
        Tuple[Set[str], List[Dict]]: Bộ giá trị trả về gồm:
            - **deleted** (Set[str]): Tập hợp đường dẫn các file bị đánh dấu là trùng (cần xóa/di chuyển).
            - **dup_log** (List[Dict]): Danh sách nhật ký chi tiết. Mỗi phần tử chứa thông tin:
                * 'kept_path', 'kept_score': Ảnh được giữ lại.
                * 'del_path', 'del_score': Ảnh bị loại bỏ.
                * 'reason': Thuật toán phát hiện ('SHA-256', 'dHash', 'pHash').
    """
    hashes_sha, hashes_d, hashes_p = {}, {}, {}
    deleted = set()
    dup_log = []
    
    print("\n⚡ [Bước 2] Quét trùng lặp Hashing...")
    
    # 1. SHA256
    for f in tqdm(image_paths, desc="SHA-256"):
        if not os.path.exists(f): # (Hầu như không bao giờ)
            continue
        # Tính SHA-256 của ảnh
        h = calculate_file_hash(f)
        # Nếu mã SHA-256 này đã tồn tại trong hashes_sha -> Ảnh này bị lặp lại 
        if h in hashes_sha:
            # Tính độ nét của 2 ảnh và trả ra file có điểm thấp hơn/đã bị xoá (thường giống nhau)
            del_path = process_duplicate_pair(hashes_sha[h], f, dup_log, "SHA-256")
            if del_path: 
                deleted.add(del_path)
        else: 
            hashes_sha[h] = f

    # Lọc bỏ những ảnh deleted trong image_paths đầu vào
    remaining = [f for f in image_paths if f not in deleted]

    # 2. Visual Hash
    for f in tqdm(remaining, desc="Visual Hash"):
        if f in deleted or not os.path.exists(f): 
            continue
        try:
            # Đọc ảnh
            img = Image.open(f)
            
            # --- XỬ LÝ dHASH ---
            dh = str(imagehash.dhash(img))
            # Nếu ảnh có "dh" đã tồn tại
            if dh in hashes_d:
                # Lấy ảnh đã tồn tại trước
                existing_path = hashes_d[dh]
                # Kiểm tra xem file cũ có còn tồn tại không?
                # Vì có thể nó đã bị xoá bởi pHash ở vòng lặp trước hoặc SHA256 (thường không/rất ít có trường hợp này)
                if not os.path.exists(existing_path):
                    hashes_d[dh] = f # File cũ chết rồi, tôn file này lên làm chủ
                else:
                    # Lấy file đã tồn tại + file hiện tại
                    # Kiểm tra file nào điểm thấp hơn -> Di chuyển vào folder phân loại
                    # Trả ra file bị xoá
                    del_path = process_duplicate_pair(existing_path, f, dup_log, "dHash")
                    if del_path: 
                        deleted.add(del_path)
                        # Nếu file bị xoá là file cũ -> Cập nhật lại ảnh với cái dh đó
                        if del_path == existing_path: 
                            hashes_d[dh] = f
                        continue # Đã xoá thì bỏ qua pHash
            else: 
                hashes_d[dh] = f
            
            # --- XỬ LÝ pHASH ---
            ph = str(imagehash.phash(img))
            # Nếu ảnh có "ph" đã tồn tại
            if ph in hashes_p:
                # Lấy ảnh đã tồn tại trước
                existing_path = hashes_p[ph]
                
                # Kiểm tra file cũ còn sống không?
                # Có thể nó vừa bị xoá bởi dHash ở vài dòng code trên 
                if not os.path.exists(existing_path):
                    hashes_p[ph] = f # File cũ chết rồi, tôn file này lên làm chủ
                else:
                    # Lấy file đã tồn tại + file hiện tại
                    # Kiểm tra file nào điểm thấp hơn -> Di chuyển vào folder phân loại
                    # Trả ra file bị xoá
                    del_path = process_duplicate_pair(existing_path, f, dup_log, "pHash")
                    if del_path: 
                        deleted.add(del_path)
                        # Nếu file bị xoá là file cũ -> Cập nhật lại ảnh với cái ph đó
                        if del_path == existing_path: 
                            hashes_p[ph] = f
            else: 
                hashes_p[ph] = f
                
        except Exception as e: 
            print(f"Error processing {f}: {e}")
            continue
        
    return deleted, dup_log

def calculate_sharpness(image_path):
    """
    Tính toán độ nét của ảnh bằng phương pháp Variance of Laplacian.

    Hàm đọc ảnh dưới dạng Grayscale để tối ưu hiệu năng, sau đó áp dụng bộ lọc 
    Laplacian để tìm cạnh và tính phương sai (variance) của các cạnh đó.
    Giá trị càng cao chứng tỏ ảnh càng nhiều chi tiết sắc nét.

    Args:
        image_path (str): Đường dẫn tuyệt đối tới file ảnh cần tính toán.

    Returns:
        float: Điểm số độ nét (Sharpness Score). 
               Trả về 0.0 nếu không đọc được file hoặc xảy ra lỗi.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Đọc gray luôn cho nhanh
        if img is None: 
            return 0.0
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except: 
        return 0.0

def process_duplicate_pair(path_a: str, path_b: str, duplicate_log: list, reason: str) -> str:
    """
    Xử lý một cặp ảnh được xác định là trùng lặp: So sánh độ nét, giữ ảnh tốt hơn
    và di chuyển ảnh kém hơn vào thư mục rác tương ứng.

    Hàm thực hiện các bước:
    1. Tính điểm độ nét (Sharpness Score) của cả 2 ảnh.
    2. So sánh: Ảnh nào nét hơn sẽ được giữ lại (Keeper).
    3. Ảnh kém hơn (Deleted) sẽ bị di chuyển (move) sang thư mục 'duplicates' hoặc 'similar'
       tùy thuộc vào lý do trùng lặp.
    4. Ghi lại thông tin chi tiết vào nhật ký (duplicate_log).

    Args:
        path_a (str): Đường dẫn tuyệt đối của ảnh thứ nhất.
        path_b (str): Đường dẫn tuyệt đối của ảnh thứ hai.
        duplicate_log (list): Danh sách chứa dict log để ghi lại lịch sử xóa.
        reason (str): Lý do trùng lặp (ví dụ: 'SHA-256', 'dHash', 'pHash').
                      Dùng để quyết định folder đích ('duplicates' cho SHA-256, 'similar' cho còn lại).

    Returns:
        str: Đường dẫn gốc của file bị xóa (để cập nhật vào danh sách deleted bên ngoài).
             Trả về None nếu có lỗi xảy ra (ví dụ file không tồn tại).
    """
    # Kiểm tra tồn tại file (tránh lỗi nếu file đã bị xóa bởi quy trình trước đó)
    if not os.path.exists(path_a) or not os.path.exists(path_b): 
        return None
    
    # 1. Tính điểm độ nét
    score_a = calculate_sharpness(path_a)
    score_b = calculate_sharpness(path_b)
    
    # 2. Quyết định giữ/xóa (Ưu tiên giữ ảnh nét hơn)
    if score_a >= score_b:
        keep, delete, score_del = path_a, path_b, score_b
        score_keep = score_a
    else:
        keep, delete, score_del = path_b, path_a, score_a
        score_keep = score_b
        
    # 3. Xác định thư mục đích
    # Nếu trùng SHA-256 (giống hệt nhau) -> folder 'duplicates'
    # Nếu trùng Hash/AI (giống tương đối) -> folder 'similar'
    folder = 'duplicates' if reason == "SHA-256" else 'similar'
    target_path = os.path.join(OUTPUT_BASE, folder, os.path.basename(delete))
    
    try:
        # 4. Di chuyển file bị loại
        shutil.move(delete, target_path)
        
        # 5. Ghi log
        duplicate_log.append({
            'kept_path': keep, 
            'kept_name': os.path.basename(keep), 
            'kept_score': score_keep,
            'del_path': target_path, 
            'del_name': os.path.basename(delete), 
            'del_score': score_del,
            'reason': reason, 
            'del_origin': delete  # Quan trọng để truy vết dây chuyền (A trùng B, B trùng C)
        })
        return delete  # Trả về đường dẫn file đã bị xóa
    except Exception as e: 
        print(f"Lỗi khi di chuyển file {delete}: {e}")
        return None

class VehicleDataset(Dataset):
    """
    Lớp Dataset tùy chỉnh để nạp và tiền xử lý ảnh xe cộ cho mô hình Deep Learning.

    Lớp này kế thừa từ torch.utils.data.Dataset, chịu trách nhiệm:
    1. Đọc ảnh từ đường dẫn file.
    2. Chuyển đổi hệ màu sang RGB (để tránh lỗi ảnh xám/PNG 4 kênh).
    3. Resize và Chuẩn hóa (Normalize) dữ liệu theo chuẩn ImageNet.
    4. Xử lý lỗi: Nếu ảnh hỏng, trả về None để DataLoader lọc bỏ sau.

    Args:
        image_paths (List[str]): Danh sách các đường dẫn tuyệt đối tới file ảnh.
    """

    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths
        
        # Pipeline biến đổi ảnh (Preprocessing)
        self.transform = T.Compose([
            # Resize về kích thước cố định mà Model yêu cầu (256x256)
            T.Resize((256, 256)),
            
            # Chuyển ảnh từ dạng PIL [0, 255] sang Tensor [0.0, 1.0]
            # Đồng thời đổi chiều từ (H, W, C) sang (C, H, W) để PyTorch hiểu
            T.ToTensor(),
            
            # Chuẩn hóa màu sắc theo thống kê của bộ dữ liệu ImageNet
            # Công thức: input[channel] = (input[channel] - mean[channel]) / std[channel]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        """Trả về tổng số lượng ảnh trong dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Lấy một mẫu dữ liệu tại vị trí index `idx`.

        Returns:
            Tuple[torch.Tensor, str]: 
                - Tensor ảnh đã qua xử lý (C, H, W).
                - Đường dẫn gốc của ảnh.
                - Trả về (None, path) nếu đọc lỗi.
        """
        path = self.image_paths[idx]
        try:
            # Mở ảnh và ép kiểu sang RGB (Quan trọng!)
            img = Image.open(path).convert("RGB")
            
            # Áp dụng các bước transform đã định nghĩa ở __init__
            return self.transform(img), path
        except Exception as e:
            print(f"Lỗi đọc ảnh {path}: {e}")
            # Trả về None để hàm collate_fn lọc bỏ sau này
            return None, path
        
def collate_fn(batch: List[Optional[Tuple[torch.Tensor, str]]]) -> Tuple[Optional[torch.Tensor], Optional[List[str]]]:
    """
    Hàm gom nhóm (collate) tùy chỉnh dùng cho DataLoader để xử lý các mẫu dữ liệu lỗi (None).

    Hàm này đóng vai trò như một bộ lọc cuối cùng trước khi dữ liệu vào Model:
    1. Duyệt qua danh sách `batch` thô và loại bỏ các phần tử là `None` (do lỗi đọc file ở Dataset).
    2. Nếu sau khi lọc không còn phần tử nào, trả về (None, None).
    3. Nếu còn dữ liệu, sử dụng `default_collate` của PyTorch để đóng gói các Tensor lẻ thành một Batch Tensor.

    Args:
        batch (List): Danh sách các mẫu dữ liệu trả về từ `VehicleDataset.__getitem__`. 
                      Mỗi phần tử là một tuple `(image_tensor, image_path)` hoặc `None`.

    Returns:
        Tuple: Một bộ 2 giá trị gồm:
            - batch_tensors (torch.Tensor): Tensor 4 chiều (Batch_Size, C, H, W).
            - batch_paths (List[str]): Danh sách đường dẫn ảnh tương ứng.
            - Trả về (None, None) nếu toàn bộ batch bị lỗi.
    """
    # ... (Implementation)
    # Lọc bỏ các mẫu bị None (lỗi đọc ảnh)
    batch = list(filter(lambda x: x[0] is not None, batch))
    # Nếu cả batch bị lỗi hết -> Trả về None
    if not batch: 
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

def setup_fastreid_model() -> torch.nn.Module:
    """
    Khởi tạo và cấu hình mô hình FastReID từ file config và weights đã chuẩn bị.

    Quy trình khởi tạo được thiết kế đặc biệt để tương thích với macOS (Apple Silicon):
    1. Nạp cấu hình mặc định và ghi đè bằng file YAML tùy chỉnh (vehicle_config.yaml).
    2. Ép mô hình khởi tạo trên CPU trước để vượt qua cơ chế kiểm tra CUDA của FastReID.
    3. Xây dựng kiến trúc mạng (Backbone + Head).
    4. Nạp trọng số (Weights) đã được huấn luyện sẵn (.pth).
    5. Chuyển mô hình sang chế độ đánh giá (Eval) và đẩy sang thiết bị tăng tốc (MPS/GPU).

    Returns:
        torch.nn.Module: Mô hình Deep Learning đã sẵn sàng để trích xuất đặc trưng.
                         (Sẽ tự động thoát chương trình nếu không tìm thấy file weights).
    """
    # 1. Lấy một bản cấu hình "trắng" chứa hàng trăm tham số mặc định của thư viện FastReID.
    cfg = get_cfg()
    
    # 2. Đọc file CONFIG_FILE và ghi đè lên bản mặc định.
    # Bước này nạp các tham số như: ResNet50, IBN=True, Input=256x256...
    cfg.merge_from_file(CONFIG_FILE)
    
    # Mẹo: Đặt device='cpu' trong config để đánh lừa FastReID bỏ qua kiểm tra CUDA
    # WORKAROUND: Force CPU build to bypass CUDA check on Mac.
    cfg.MODEL.DEVICE = "cpu"
    
    # 3. Xây dựng khung model (kiến trúc) dựa trên config
    model = build_model(cfg)
    
    # 4. Nạp "kiến thức" (Weights) từ file .pth vào khung model
    if os.path.exists(WEIGHTS_PATH):
        Checkpointer(model).load(WEIGHTS_PATH)
    else:
        print(f"❌ LỖI: Không tìm thấy file weights.")
        exit()
        
    # 5. Chuyển sang chế độ đánh giá (Eval mode)
    # Tắt các lớp Dropout, Batch Norm dynamic để kết quả cố định
    model.eval()
    
    # 6. Đẩy toàn bộ model sang thiết bị thực tế (Mac MPS hoặc GPU)
    model.to(DEVICE) 
    return model

def extract_features(clean_images: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Trích xuất vector đặc trưng (Embeddings) từ danh sách ảnh sử dụng mô hình Deep Learning (FastReID).
    (Đã tối ưu bộ nhớ bằng cách cấp phát trước ma trận kết quả)

    Quy trình thực hiện:
    1. Sắp xếp và lọc trùng danh sách đầu vào để đảm bảo chỉ số (Index) cố định.
    2. Chạy mô hình theo cơ chế Batch Processing (xử lý hàng loạt) để tối ưu tốc độ.
    3. Flatten (làm phẳng) các tensor đầu ra về dạng 2D (N, D).
    4. Thực hiện chuẩn hóa L2 (L2 Normalization) ngay lập tức (Bước quan trọng để tính Cosine Similarity).
    5. Lưu trữ backup 2 file `features.npy` và `paths.npy` xuống ổ cứng.

    Args:
        clean_images (List[str]): Danh sách đường dẫn tuyệt đối của các ảnh cần trích xuất.

    Returns:
        Tuple[np.ndarray, List[str]]: Bộ giá trị gồm:
            - final_feats (np.ndarray): Ma trận các vector đặc trưng (float32), kích thước (Số ảnh, Số chiều).
            - all_paths (List[str]): Danh sách đường dẫn ảnh tương ứng 1-1 với các dòng trong ma trận.
            Trả về (None, None) nếu quá trình trích xuất thất bại hoặc không có ảnh.
    """
    print(f'✨ [Bước 3] Trích xuất đặc trưng Deep Learning ({len(clean_images)} ảnh)...')
    
    # Sắp xếp lại ds đầu vào để các lần chạy là 1 kết quả 
    clean_images = sorted(list(set(clean_images)))
    num_images = len(clean_images) # (Code mới: Cần số lượng để tạo ma trận rỗng)
    
    model = setup_fastreid_model()
    # Chứa danh sách các đường dẫn ảnh (clean_images) và quy trình xử lý từng ảnh lẻ (đọc ảnh -> resize -> normalize). Nhưng lúc này nó chưa làm gì cả, chỉ đứng yên chờ lệnh.
    dataset = VehicleDataset(clean_images)
    
    # DataLoader điều WORKERS nhân viên chạy vào Kho (Dataset), lấy ra BATCH_SIZE ảnh theo đúng thứ tự danh sách. Nếu gặp ảnh lỗi, hàm collate_fn sẽ loại bỏ nó. Sau đó đóng gói lại và chuyển vào Model để xử lý.
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # Đang trích xuất đặc trứng --> Không xáo trộn, đọc tuần tự
        num_workers=WORKERS, 
        collate_fn=collate_fn # collate_fn: Đóng gói ảnh thành batch, nếu None -> loại bỏ ra
    )
    
    # --- [CODE MỚI] TỐI ƯU BỘ NHỚ: Cấp phát trước vùng nhớ cố định ---
    # Tại sao? Code cũ dùng list.append() gây phân mảnh RAM khi dữ liệu lớn (100k ảnh).
    # Giải pháp: Tạo sẵn cái thùng chứa vừa khít (Pre-allocation).
    # np.zeros: Tạo ma trận toàn số 0.
    # (num_images, 2048): Kích thước (số ảnh, độ dài vector ResNet50).
    # dtype='float32': Định dạng số thực nhẹ, chuẩn cho FAISS.
    features_matrix = np.zeros((num_images, 2048), dtype='float32')
    all_paths = []
    
    print("--> Đang chạy Model...")
    start_idx = 0 # Con trỏ đánh dấu vị trí bắt đầu điền dữ liệu
    
    with torch.no_grad():
        # Duyệt từng lô ảnh
        for imgs, paths in tqdm(dataloader, desc="Embedding"):
            # Vì collate_fn có thể trả về (None, None) nên phải check kỹ
            if imgs is None: continue
                
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            
            if len(feats.shape) > 2: 
                # Ép dẹp (Flatten) khối dữ liệu thừa chiều về dạng chuẩn 2 chiều
                # FAISS ở bước sau chỉ hiểu ma trận 2 chiều -> Phải ép dẹp nó về (số ảnh, số chiều vector)
                # feats.size(0): Giữ nguyên số lượng ảnh (batch_size)
                feats = feats.view(feats.size(0), -1)
            
            # Khi tính toán, biến feats đang nằm trên VRAM của GPU
            # .cpu(): copy dữ liệu đó từ card đồ họa về lại RAM hệ thống (CPU) để chuẩn bị lưu trữ.
            # .numpy(): Đổi định dạng Tensor sang Numpy --> để lưu file .npy --> cho FAISS đọc hiểu
            batch_feats = feats.cpu().numpy()
            
            # Lấy kích thước thực tế của batch hiện tại (thường là 128, nhưng batch cuối có thể ít hơn)
            batch_size = batch_feats.shape[0]
            
            # --- [CODE MỚI] ĐIỀN TRỰC TIẾP VÀO MA TRẬN LỚN ---
            # Tính toán vị trí kết thúc: Từ start_idx đến end_idx
            end_idx = start_idx + batch_size
            
            # Gán dữ liệu batch vào đúng vị trí trong ma trận lớn đã tạo sẵn
            # Thay vì nối đuôi (append) tốn kém, ta điền vào chỗ trống
            features_matrix[start_idx:end_idx, :] = batch_feats
            
            # Lưu paths (List string nhẹ nên append được, không cần tối ưu như ma trận số)
            all_paths.extend(paths)
            
            # Cập nhật con trỏ bắt đầu cho vòng lặp sau
            start_idx = end_idx
            
    # [CODE MỚI] Kiểm tra xem có ảnh nào bị lỗi (None) dẫn đến ma trận bị thừa dòng 0 ở cuối không
    # Nếu số lượng path thực tế ít hơn số lượng ảnh ban đầu (do có ảnh lỗi), ta cắt bớt phần thừa của ma trận
    if len(all_paths) < num_images:
        features_matrix = features_matrix[:len(all_paths)]
    
    if len(all_paths) == 0: 
        # features, paths trả ra là None None
        return None, None
    
    # [CODE CŨ - Giữ nguyên logic]
    # Chuyển đổi độ dài của các vector về 1, giữ nguyên hướng
    # Lúc này so sánh 2 ảnh bằng góc Vector, góc càng nhỏ --> 2 ảnh càng giống nhau
    # Lưu ý: faiss.normalize_L2 làm việc trực tiếp trên bộ nhớ (In-place), không tạo bản copy mới -> Tiết kiệm RAM
    faiss.normalize_L2(features_matrix)
    
    out_dir = os.path.join(OUTPUT_BASE, "output_features")
    os.makedirs(out_dir, exist_ok=True) # Thêm dòng này cho an toàn
    # Lưu Ma trận số học chứa các vector đặc trưng
    np.save(os.path.join(out_dir, "features.npy"), features_matrix)
    # Một danh sách (List) các đường dẫn file ảnh
    np.save(os.path.join(out_dir, "paths.npy"), all_paths)
    
    print(f"✅ Đã lưu features.npy ({features_matrix.shape}) vào {out_dir}")
    
    return features_matrix, all_paths

def cluster_and_filter_faiss(features: np.ndarray, paths: List[str], duplicate_log: List[Dict]) -> int:
    """
    Phân cụm và lọc ảnh trùng lặp sử dụng AI (FAISS) kết hợp Lý thuyết đồ thị và Kiểm tra trực tiếp.

    Chiến lược hoạt động: "Gom nhóm rộng, Kiểm tra chặt".
    1. Dùng FAISS để tìm tất cả các cặp ảnh có nét tương đồng (Range Search).
    2. Dùng Đồ thị (Graph) để gom các cặp rời rạc thành các nhóm liên thông (Connected Components).
    3. Trong mỗi nhóm, chọn ra ảnh nét nhất làm "Keeper" (Ảnh gốc).
    4. **Kiểm tra trực tiếp (Direct Check):** Tính lại độ giống nhau giữa Keeper và từng ảnh thành viên.
       Chỉ xóa ảnh thành viên nếu nó thực sự giống Keeper trên ngưỡng quy định. Điều này giúp tránh
       lỗi "bắc cầu" (A giống B, B giống C, nhưng A khác C).

    Args:
        features (np.ndarray): Ma trận vector đặc trưng đã chuẩn hóa L2 (Shape: N x 2048).
        paths (List[str]): Danh sách đường dẫn ảnh tương ứng.
        duplicate_log (List[Dict]): Danh sách để ghi nhật ký các file bị xóa.

    Returns:
        int: Số lượng ảnh đã bị di chuyển sang thư mục 'similar'.
    """
    print(f"\n✨ [Bước 5] Gom nhóm ảnh trùng bằng FAISS (Threshold={THRESHOLD_FAISS} - Aggressive Mode)...")
    
    # 1. Đưa toàn bộ vector vào cấu trúc dữ liệu của FAISS để chuẩn bị tìm kiếm.
    # shape[1]: Là độ dài của vector đặc trưng (ví dụ: 2048 con số).
    # Mục đích: Để khai báo độ dài cho FAISS
    d = features.shape[1]

    # Sử dụng IndexFlatIP. 
    # faiss: Thư viện
    # Index: cấu trúc dữ liệu
    # Flat: phẳng (Lưu trữ nguyên bản). So sánh cần tìm với tất cả vector còn lại
    # IP: Tích vô hướng == Độ tương đồng Cosine (Góc).
    # Vì vector đã chuẩn hóa L2, tích vô hướng chính là Cosine Similarity (Độ tương đồng góc).
    index = faiss.IndexFlatIP(d)
    # add toàn bộ vector features vào index đã tạo
    index.add(features)
    
    # 2. Range Search: 3 mảng 1 chiều nén (Compressed): lims, D, I.
    # lims(Limits): Là mục lục để biết ảnh thứ i nằm từ đâu đến đâu
    # D(Distances): Chứa toàn bộ điểm số tương đồng (Cosine Similarity) của tất cả các cặp tìm thấy, được nối đuôi nhau.
    # I(Indices): Chứa ID (Index) của những ảnh tìm thấy, tương ứng song song với mảng D.
    # ==> Với mỗi ảnh, FAISS trả về một danh sách các "hàng xóm" (những ảnh khác giống nó).
    lims, D, I = index.range_search(features, THRESHOLD_FAISS)
    # ______ Khúc này hết hiểu rồi _______
    # 3. Xây dựng đồ thị
    # Tạo ra một đồ thị rỗng.
    G = nx.Graph()
    # Rải lên đó 100.000 cái Chấm tròn (Node). Mỗi chấm đại diện cho 1 bức ảnh (từ 0 đến 99.999).
    G.add_nodes_from(range(len(paths)))

    # Duyệt qua từng ảnh (gọi là ảnh A)
    for i in tqdm(range(len(paths)), desc="Building Graph"):
        # 1. Tra mục lục để tìm phạm vi kết quả của ảnh A
        start = lims[i]
        end = lims[i+1]
        
        # 2. Duyệt qua các kết quả tìm thấy trong phạm vi đó
        for j in range(start, end):
            # I[j] chính là ID của ảnh hàng xóm (gọi là ảnh B)
            
            if i != I[j]: # Nếu A khác B (không tự nối với chính mình)
                
                # 3. Vẽ một đường thẳng nối giữa A và B
                G.add_edge(i, I[j])

    # 4. Xử lý nhóm (Logic: Direct Check với Keeper)
    components = list(nx.connected_components(G))
    duplicate_groups = [c for c in components if len(c) > 1]
    
    deleted_count = 0
    sharpness_cache = {} 
    def get_sharpness(idx):
        if idx not in sharpness_cache:
            sharpness_cache[idx] = calculate_sharpness(paths[idx])
        return sharpness_cache[idx]

    # Dùng tqdm
    for component in tqdm(duplicate_groups, desc="Cleaning"):
        comp_list = list(component)
        
        # Tìm Vua (Keeper) - Ảnh nét nhất trong cả đám
        comp_list.sort(key=lambda x: get_sharpness(x), reverse=True)
        keeper_idx = comp_list[0]
        keeper_vec = features[keeper_idx]
        keeper_path = paths[keeper_idx]
        keeper_score = get_sharpness(keeper_idx)
        
        # Duyệt qua các thần dân (Candidates)
        for candidate_idx in comp_list[1:]:
            # --- SO GĂNG TRỰC TIẾP ---
            # Tính lại độ giống nhau giữa Vua và Thần dân
            candidate_vec = features[candidate_idx]
            sim = np.dot(keeper_vec, candidate_vec)
            
            # Nếu độ giống nhau lớn hơn ngưỡng -> XÓA
            if sim >= THRESHOLD_FAISS:
                del_path = paths[candidate_idx]
                target_path = os.path.join(OUTPUT_BASE, "similar", os.path.basename(del_path))
                
                try:
                    shutil.move(del_path, target_path)
                    
                    sim_percent = f"{sim * 100:.2f}%"
                    duplicate_log.append({
                        'kept_path': keeper_path, 
                        'kept_name': os.path.basename(keeper_path), 
                        'kept_score': keeper_score,
                        'del_path': target_path, 
                        'del_name': os.path.basename(del_path), 
                        'del_score': get_sharpness(candidate_idx),
                        'reason': f"AI: {sim_percent}", 
                        'del_origin': del_path
                    })
                    deleted_count += 1
                except: pass
            else:
                # Trường hợp: A giống B (0.85), B giống C (0.85) => A,B,C vào 1 nhóm
                # Nhưng A chỉ giống C (0.75) => KHÔNG XÓA C.
                # C được giữ lại (sẽ trở thành Keeper của một nhóm khác hoặc đứng độc lập)
                pass

    return deleted_count

def calculate_detail_score(image_path: str) -> float:
    """
    Tính điểm "Độ chi tiết" (Detail Density) bằng thuật toán Canny Edge Detection.
    
    Nguyên lý: Đếm số lượng điểm ảnh là cạnh (Edge Pixels). 
    - Ảnh trơn (sườn xe): Ít cạnh -> Điểm thấp.
    - Ảnh chi tiết (biển số, lưới tản nhiệt): Nhiều cạnh -> Điểm cao (ví dụ 10.000 - 50.000).

    Args:
        image_path (str): Đường dẫn file ảnh.

    Returns:
        float: Số lượng pixel cạnh tìm thấy.
    """
    try:
        # Đọc ảnh xám
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return 0.0
        
        # Dùng Canny để tìm cạnh
        # Ngưỡng 100-200 là tiêu chuẩn vàng để lọc nhiễu nhẹ, chỉ lấy nét chính
        edges = cv2.Canny(img, 100, 200)
        
        # Đếm tổng số điểm ảnh là cạnh (pixel màu trắng = 255)
        # np.count_nonzero đếm số phần tử khác 0
        score = np.count_nonzero(edges)
        
        return float(score)
    except:
        return 0.0

def generate_html_report(duplicate_log, quality_log, output_file):
    print("📝 Đang tạo báo cáo HTML (UI/UX Ultimate Version)...")

    # --- 1. XỬ LÝ DỮ LIỆU ---
    move_map = {entry['del_origin']: entry['kept_path'] for entry in duplicate_log}
    def find_ultimate_keeper(current_path):
        if current_path in move_map: return find_ultimate_keeper(move_map[current_path])
        return current_path

    grouped_data = {}
    for entry in duplicate_log:
        final_keeper = find_ultimate_keeper(entry['kept_path'])
        if final_keeper not in grouped_data:
            k_name = os.path.basename(final_keeper)
            k_score = entry['kept_score'] if final_keeper == entry['kept_path'] else calculate_sharpness(final_keeper)
            grouped_data[final_keeper] = {'kept_info': {'name': k_name, 'path': final_keeper, 'score': k_score}, 'deleted_list': []}
        grouped_data[final_keeper]['deleted_list'].append(entry)

    # Thống kê
    total_quality = len(quality_log)
    total_dups = sum(len(g['deleted_list']) for g in grouped_data.values())
    
    # --- 2. HTML TEMPLATE ---
    html_head = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dataset Cleaning Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --primary: #3B82F6; --primary-light: #DBEAFE;
                --success: #10B981; --success-light: #D1FAE5;
                --warning: #F59E0B; --warning-light: #FEF3C7;
                --danger: #EF4444; --danger-light: #FEE2E2;
                --dark: #111827; --gray: #6B7280; --bg: #F9FAFB; --card: #FFFFFF;
                --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
                --radius: 16px;
            }
            
            /* Dark Mode Variables */
            [data-theme="dark"] {
                --bg: #0F172A; --card: #1E293B; --text: #F8FAFC; --dark: #F3F4F6;
                --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
                color: #F8FAFC;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; transition: background 0.3s, color 0.3s; }
            body { font-family: 'Plus Jakarta Sans', sans-serif; background: var(--bg); color: var(--dark); padding-bottom: 100px; }
            
            /* Navbar */
            .navbar {
                position: fixed; top: 0; width: 100%; z-index: 1000;
                background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(12px);
                border-bottom: 1px solid rgba(0,0,0,0.05);
                [data-theme="dark"] & { background: rgba(30, 41, 59, 0.8); border-bottom: 1px solid rgba(255,255,255,0.05); }
            }
            .nav-content {
                max-width: 1400px; margin: 0 auto; height: 70px; padding: 0 24px;
                display: flex; justify-content: space-between; align-items: center;
            }
            .logo { font-weight: 800; font-size: 20px; display: flex; align-items: center; gap: 8px; background: linear-gradient(135deg, #3B82F6, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .nav-menu { display: flex; gap: 8px; background: rgba(0,0,0,0.03); padding: 4px; border-radius: 12px; }
            .nav-item { 
                padding: 8px 16px; border-radius: 8px; font-size: 14px; font-weight: 600; color: var(--gray); text-decoration: none; 
                transition: all 0.2s; display: flex; align-items: center; gap: 6px;
            }
            .nav-item:hover { color: var(--primary); background: rgba(255,255,255,0.5); }
            .nav-item.active { background: var(--card); color: var(--primary); box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
            
            /* Toggle Theme */
            .theme-toggle {
                width: 40px; height: 40px; border-radius: 50%; border: none; cursor: pointer;
                background: rgba(0,0,0,0.05); display: flex; align-items: center; justify-content: center; font-size: 18px;
            }

            /* Dashboard */
            .container { max-width: 1400px; margin: 0 auto; padding: 100px 24px 40px; }
            .dashboard { 
                display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 24px; margin-bottom: 40px; 
            }
            .stat-card {
                background: var(--card); padding: 24px; border-radius: var(--radius); box-shadow: var(--shadow);
                display: flex; flex-direction: column; gap: 8px; position: relative; overflow: hidden;
            }
            .stat-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; }
            .stat-icon { width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px; margin-bottom: 8px; }
            .stat-value { font-size: 32px; font-weight: 800; }
            .stat-label { font-size: 14px; color: var(--gray); font-weight: 500; }

            /* Section */
            .section-header { 
                display: flex; justify-content: space-between; align-items: end; margin-bottom: 24px; 
                border-bottom: 2px solid rgba(0,0,0,0.05); padding-bottom: 16px;
            }
            .title-group h2 { font-size: 24px; font-weight: 700; display: flex; align-items: center; gap: 12px; }
            .badge-count { background: var(--primary); color: white; padding: 4px 12px; border-radius: 20px; font-size: 14px; }
            
            /* Grid Layouts */
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 24px; }
            .card { 
                background: var(--card); border-radius: 16px; overflow: hidden; box-shadow: var(--shadow);
                transition: transform 0.2s; border: 1px solid rgba(0,0,0,0.03);
            }
            .card:hover { transform: translateY(-6px); box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.1); }
            
            .card-img-box { position: relative; padding-top: 75%; overflow: hidden; background: #f1f5f9; }
            .card-img { 
                position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; 
                transition: transform 0.5s; cursor: zoom-in;
            }
            .card:hover .card-img { transform: scale(1.05); }
            
            .card-body { padding: 16px; }
            .tag { 
                display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 6px; 
                font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
            }
            
            /* Comparison Group */
            .comp-group { 
                background: var(--card); border-radius: 24px; padding: 32px; margin-bottom: 40px; 
                box-shadow: var(--shadow); display: flex; gap: 40px; position: relative;
            }
            .comp-keeper { flex: 0 0 300px; text-align: center; border-right: 1px solid rgba(0,0,0,0.05); padding-right: 40px; position: sticky; top: 100px; height: fit-content; }
            .comp-deleted { flex: 1; }
            
            .keeper-preview { 
                width: 100%; aspect-ratio: 1/1; object-fit: contain; border-radius: 16px; 
                background: #F8FAFC; border: 1px solid rgba(0,0,0,0.05); margin: 16px 0;
                cursor: zoom-in;
            }
            
            /* Deleted Grid */
            .del-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 16px; }
            .del-card { position: relative; border-radius: 12px; overflow: hidden; background: #000; }
            .del-card img { width: 100%; height: 100px; object-fit: cover; opacity: 0.7; transition: 0.3s; }
            .del-card:hover img { opacity: 1; }
            .del-info { 
                position: absolute; bottom: 0; left: 0; width: 100%; padding: 8px;
                background: linear-gradient(to top, rgba(0,0,0,0.8), transparent);
                color: white; font-size: 10px; display: flex; justify-content: space-between;
            }
            .del-badge { 
                position: absolute; top: 6px; right: 6px; padding: 2px 6px; 
                border-radius: 4px; font-size: 10px; font-weight: 700; color: #fff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }

            /* Colors */
            .c-blur { background: var(--warning-light); color: var(--warning); }
            .c-dark { background: #E5E7EB; color: #374151; }
            .c-bright { background: #DBEAFE; color: #1E40AF; }
            
            .c-sha { background: #059669; }
            .c-vis { background: #0891B2; }
            .c-ai { background: #7C3AED; }

            /* Modal */
            .modal { display: none; position: fixed; z-index: 2000; inset: 0; background: rgba(0,0,0,0.95); backdrop-filter: blur(5px); cursor: zoom-out; }
            .modal-img { max-width: 90%; max-height: 90vh; margin: auto; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); border-radius: 8px; box-shadow: 0 20px 50px rgba(0,0,0,0.5); }
        </style>
    </head>
    <body>
        <nav class="navbar">
            <div class="nav-content">
                <div class="logo"><i class="fa-solid fa-wand-magic-sparkles"></i> Neural Cleaner</div>
                <div class="nav-menu">
                    <a href="#stats" class="nav-item active"><i class="fa-solid fa-chart-pie"></i> Tổng quan</a>
                    <a href="#quality" class="nav-item"><i class="fa-solid fa-triangle-exclamation"></i> Chất lượng</a>
                    <a href="#hashing" class="nav-item"><i class="fa-solid fa-fingerprint"></i> Hashing</a>
                    <a href="#ai" class="nav-item" style="color:var(--ai)"><i class="fa-solid fa-brain"></i> AI Deep Learning</a>
                </div>
                <button class="theme-toggle" onclick="toggleTheme()"><i class="fa-solid fa-moon"></i></button>
            </div>
        </nav>

        <div class="container" id="stats">
            <div class="dashboard">
                <div class="stat-card" style="border-top: 4px solid var(--warning);">
                    <div class="stat-icon c-blur"><i class="fa-solid fa-eye-slash"></i></div>
                    <span class="stat-value">{qty_bad}</span>
                    <span class="stat-label">Ảnh kém chất lượng</span>
                </div>
                <div class="stat-card" style="border-top: 4px solid var(--success);">
                    <div class="stat-icon" style="background:var(--success-light); color:var(--success)"><i class="fa-solid fa-clone"></i></div>
                    <span class="stat-value">{hash_dups}</span>
                    <span class="stat-label">Trùng lặp (Hashing)</span>
                </div>
                <div class="stat-card" style="border-top: 4px solid var(--ai);">
                    <div class="stat-icon" style="background:#F3E8FF; color:var(--ai)"><i class="fa-solid fa-robot"></i></div>
                    <span class="stat-value">{ai_dups}</span>
                    <span class="stat-label">Trùng lặp (AI Detected)</span>
                </div>
                <div class="stat-card" style="background:linear-gradient(135deg, var(--primary), #2563EB); color:white">
                    <div class="stat-icon" style="background:rgba(255,255,255,0.2); color:white"><i class="fa-solid fa-broom"></i></div>
                    <span class="stat-value">{total_cleaned}</span>
                    <span class="stat-label" style="color:rgba(255,255,255,0.8)">Tổng file đã lọc</span>
                </div>
            </div>
    """

    # --- 3. SECTION: QUALITY ---
    html_quality = f"""
        <div id="quality" class="section">
            <div class="section-header">
                <div class="title-group">
                    <h2><i class="fa-solid fa-filter" style="color:var(--warning)"></i> Ảnh Kém Chất Lượng <span class="badge-count">{len(quality_log)}</span></h2>
                </div>
            </div>
            <div class="grid">
    """
    for item in quality_log:
        reason = item['reason'].lower()
        icon = "fa-moon" if "dark" in reason else ("fa-sun" if "bright" in reason else "fa-blur")
        badge_cls = f"c-{reason}"
        
        html_quality += f"""
            <div class="card">
                <div class="card-img-box">
                    <img class="card-img" data-src="{item['path']}" loading="lazy" onclick="openModal(this)">
                </div>
                <div class="card-body">
                    <span class="tag {badge_cls}"><i class="fa-solid {icon}"></i> {item['reason']}</span>
                    <div style="font-weight:600; font-size:13px; margin-bottom:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{item['name']}</div>
                    <div style="font-size:11px; color:var(--gray)">Score: <b>{item['score']:.1f}</b></div>
                </div>
            </div>
        """
    html_quality += "</div></div>"

    # --- 4. SECTION: DUPLICATES ---
    # Tách dữ liệu
    sorted_groups = sorted(grouped_data.values(), key=lambda x: len(x['deleted_list']), reverse=True)
    
    html_hash = '<div id="hashing" class="section"><div class="section-header"><div class="title-group"><h2><i class="fa-solid fa-fingerprint" style="color:var(--success)"></i> Lọc Hashing</h2></div></div>'
    html_ai = '<div id="ai" class="section"><div class="section-header"><div class="title-group"><h2><i class="fa-solid fa-brain" style="color:var(--ai)"></i> Lọc AI Deep Learning</h2></div></div>'
    
    count_hash_del = 0
    count_ai_del = 0

    for group in sorted_groups:
        kept = group['kept_info']
        deleted = group['deleted_list']
        
        hash_dels = [d for d in deleted if "AI" not in d['reason']]
        ai_dels = [d for d in deleted if "AI" in d['reason']]
        
        count_hash_del += len(hash_dels)
        count_ai_del += len(ai_dels)

        def render_block(dels, type="hash"):
            if not dels: return ""
            cards = ""
            for d in dels:
                badge_cls = "c-ai" if type == "ai" else ("c-sha" if "SHA" in d['reason'] else "c-vis")
                cards += f"""
                <div class="del-card">
                    <span class="del-badge {badge_cls}">{d['reason']}</span>
                    <img data-src="{d['del_path']}" loading="lazy" onclick="openModal(this)">
                    <div class="del-info">
                        <span><i class="fa-solid fa-trash"></i></span>
                        <span>{d['del_score']:.0f}</span>
                    </div>
                </div>
                """
            
            theme_color = "var(--ai)" if type == "ai" else "var(--success)"
            return f"""
            <div class="comp-group">
                <div class="comp-keeper">
                    <span class="tag" style="background:var(--success-light); color:var(--success); font-size:12px;"><i class="fa-solid fa-check"></i> GIỮ LẠI (BEST)</span>
                    <img class="keeper-preview" src="{kept['path']}" onclick="openModal(this)">
                    <div style="font-weight:700;">{kept['name']}</div>
                    <div style="color:var(--gray); font-size:12px;">Độ nét: {kept['score']:.1f}</div>
                </div>
                <div class="comp-deleted">
                    <h4 style="margin-bottom:16px; color:{theme_color}; display:flex; align-items:center; gap:8px;">
                        <i class="fa-solid fa-trash-can"></i> Đã loại bỏ {len(dels)} bản sao
                    </h4>
                    <div class="del-grid">{cards}</div>
                </div>
            </div>
            """

        html_hash += render_block(hash_dels, "hash")
        html_ai += render_block(ai_dels, "ai")

    html_hash += "</div>"
    html_ai += "</div>"

    # --- 5. FOOTER & JS ---
    html_end = """
        </div> <div id="viewer" class="modal" onclick="this.style.display='none'">
            <img class="modal-img" id="modal-img">
        </div>
        
        <script>
            // Lazy Load Images
            document.addEventListener("DOMContentLoaded", function() {
                const observer = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        if(entry.isIntersecting) {
                            const img = entry.target;
                            img.src = img.dataset.src;
                            observer.unobserve(img);
                        }
                    });
                });
                document.querySelectorAll('img[data-src]').forEach(img => observer.observe(img));
            });

            function openModal(el) {
                document.getElementById('viewer').style.display = 'block';
                document.getElementById('modal-img').src = el.src || el.dataset.src;
            }

            function toggleTheme() {
                const body = document.body;
                body.setAttribute('data-theme', body.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
            }
            
            // Scroll Spy
            window.onscroll = () => {
                document.querySelectorAll('.section').forEach(sec => {
                    if(window.scrollY >= (sec.offsetTop - 100)) {
                        document.querySelectorAll('.nav-item').forEach(a => a.classList.remove('active'));
                        document.querySelector('.nav-item[href*=' + sec.id + ']').classList.add('active');
                    }
                });
            };
        </script>
    </body>
    </html>
    """
    
    # Replace Placeholders
    final_html = html_head.replace("{qty_bad}", str(total_quality)) \
                          .replace("{hash_dups}", str(count_hash_del)) \
                          .replace("{ai_dups}", str(count_ai_del)) \
                          .replace("{total_cleaned}", str(total_quality + count_hash_del + count_ai_del)) \
               + html_quality + html_hash + html_ai + html_end

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_html)
        print(f"✅ Đã tạo báo cáo HTML (V5 - Ultimate UI/UX) tại: {output_file}")
    except Exception as e:
        print(f"❌ Lỗi report: {e}")
# ================= HÀM MAIN (ĐIỀU PHỐI CHÍNH) =================
def main():
    start_time = time.time()

    # Xoá folder results cũ nếu tồn tại, tạo lại folder mới
    setup_folders()

    # Bước 0: Lấy danh sách ảnh
    all_images = get_image_paths()
    if all_images == []:
        return
    else:
        print(f"🔍 Tổng ảnh đầu vào: {len(all_images)}")

    # Bước 1: Lọc chất lượng của ảnh
    # clean_images: ảnh vượt qua vòng kiểm tra chất lượng
    # quality_log: Những ảnh không vượt qua kiểm tra, lưu lại mọi thông tin -> Báo cáo
    clean_images, quality_log = scan_and_filter_quality(all_images_path=all_images)
    print(f"📉 Sau lọc chất lượng còn: {len(clean_images)}")

    # B2: Lọc Hashing
    # deleted_hashing: Ảnh bị xoá
    # duplicate_log: Những ảnh không vượt qua kiểm tra, lưu lại mọi thông tin -> Báo cáo
    deleted_hashing, duplicate_log = find_duplicates_by_hashing(clean_images)
    # Cập nhật danh sách ảnh sạch (trừ ảnh đã xóa do hashing)
    clean_images = [img for img in clean_images if img not in deleted_hashing]
    print(f"📉 Sau lọc Hashing còn: {len(clean_images)}\n")

    # B3: Trích xuất đặc trưng (Deep Learning)
    # features: Vector của danh sách ảnh
    # paths: Đường dẫn của ảnh
    features, paths = extract_features(clean_images)

    if features is not None and len(paths) > 0:
        # B4: Lọc FAISS Clustering
        deleted_faiss_count = cluster_and_filter_faiss(features, paths, duplicate_log)
        print(f"📉 Đã lọc thêm {deleted_faiss_count} ảnh trùng bằng AI.\n")
    else:
        print("⚠️ Không có feature nào để chạy FAISS.")

    

    # B5: Tạo báo cáo (Tổng hợp tất cả log)
    # (Bạn cần copy lại hàm generate_html_report vào code này để chạy dòng dưới)
    generate_html_report(duplicate_log, quality_log, os.path.join(OUTPUT_BASE, REPORT_FILE))
    # Lưu log ra JSON để backup
    log_data = {
    "quality_log": quality_log,
    "duplicate_log": duplicate_log,
    "stats": {
        "total_input": len(all_images),
        "clean_after_quality": len(clean_images) if 'clean_images' in locals() else 0,
    }
}
    with open(os.path.join(OUTPUT_BASE, "cleaning_log.json"), "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
        print("✅ Đã lưu file log thô (JSON).")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'\n')
    print(f"🏁 Thời gian chạy: {elapsed_time} giây")




if __name__ == "__main__":
    main()