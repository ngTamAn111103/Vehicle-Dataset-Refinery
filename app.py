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
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import html

# ================= CẤU HÌNH (CONFIG) =================
TEST = False
SAMPLE_SIZE = 10000

# ___Ngưỡng lọc ảnh___
BLUR_THRESHOLD = 50.0      # Độ nét
DARK_THRESHOLD = 10.0       # Độ tối
BRIGHT_THRESHOLD = 220.0    # Độ sáng
THRESHOLD_FAISS = 0.7      # Ngưỡng giống nhau Deep Learning

# ___Tốc độ___
BATCH_SIZE = 128
# Sử dụng 50% sức mạnh CPU 
WORKERS = max(1, int(os.cpu_count() - 2))

# ___Đường dẫn (Nên để tuyệt đối)___
# INPUT_FOLDER = '/Volumes/MICRON/raw_dataset_v1.1'
INPUT_FOLDER = '/Volumes/MICRON/FriendNightClub'
OUTPUT_BASE = '/Users/nguyentaman/Downloads/ResNet-FAISS-Dedup/results2'
# ___Đường dẫn (Tương đối cũng được)___
WEIGHTS_PATH = "configs/vehicle_weights.pth"
CONFIG_FILE = "configs/vehicle_config.yaml"
REPORT_FILE = 'cleaning_report.html'
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp")
FOLDERS = ["blur", "dark", "bright", "duplicates", "similar", "output_features"]

# ___Thiết bị___
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ================= CÁC HÀM HỖ TRỢ (UTILS) =================
def setup_folders():
    """Kiểm tra và tạo lại cấu trúc thư mục đầu ra."""
    # Nếu tồn tại thư mục đầu ra -> Xoá
    if os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)

    # Tạo lại các thư mục đầu ra
    for folder in FOLDERS:
        os.makedirs(os.path.join(OUTPUT_BASE, folder), exist_ok=True)

def get_image_paths() -> List[str]:
    """
    Lấy danh sách đường dẫn tuyệt đối tất cả ảnh (đệ quy).

    Returns:
        List[str]: Danh sách đường dẫn tuyệt đối
    """
    # Danh sách đường dẫn tuyệt đối
    all_files = []
    # Đầu vào không tồn tại
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder không tồn tại: {INPUT_FOLDER}")
        return []
    
    # Đệ quy thư mục INPUT_FOLDER
    # root: Thư mục đang đứng
    # files: Tất cả các file
    for root, _, files in os.walk(INPUT_FOLDER):
        # Duyệt tất cả các file
        for file in files:
            # Nếu file có đuôi trong IMAGE_EXTENSIONS & không bắt đầu bằng '.'
            if file.lower().endswith(IMAGE_EXTENSIONS) and not file.startswith('.'):
                # Cho vào danh sách trả ra
                all_files.append(os.path.abspath(os.path.join(root, file)))

    # Nếu đang TEST & số lượng ảnh đủ/dư
    if TEST and len(all_files) > SAMPLE_SIZE:
        print(f"⚠️ Chế độ TEST: Lấy ngẫu nhiên {SAMPLE_SIZE} ảnh.")
        return random.sample(all_files, SAMPLE_SIZE)
    # Nếu chạy thật -> Sort từ đầu
    return sorted(all_files)

def calculate_file_hash(filepath: str, method: str = 'sha256') -> str:
    """Tính hash file binary."""
    hasher = hashlib.sha256() if method == 'sha256' else hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except:
        return None

def calculate_sharpness(image_path):
    """Tính độ nét (Laplacian Variance)."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            return 0.0
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except: 
        return 0.0

def calculate_detail_score(image_path: str) -> float:
    """Tính điểm chi tiết (Canny Edge)."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            return 0.0
        edges = cv2.Canny(img, 100, 200)
        return float(np.count_nonzero(edges))
    except:
        return 0.0

def process_duplicate_pair(path_a: str, path_b: str, duplicate_log: list, reason: str) -> str:
    """Xử lý cặp trùng lặp: Giữ ảnh nét hơn, xóa ảnh kia."""
    # Nếu đường dẫn tuyệt đối 1 trong 2 ảnh không tồn tại -> dừng
    if not os.path.exists(path_a) or not os.path.exists(path_b): 
        return None
    
    score_a = calculate_sharpness(path_a)
    score_b = calculate_sharpness(path_b)
    
    # Giữ ảnh điểm cao hơn, xoá ảnh điểm thấp hơn, lưu điểm của ảnh bị xoá (Ghi log)
    # Giữ điểm của ảnh điểm cao hơn (Cũng ghi log)
    if score_a >= score_b:
        keep, delete, score_del = path_a, path_b, score_b
        score_keep = score_a
    else:
        keep, delete, score_del = path_b, path_a, score_a
        score_keep = score_b
    
    # Trùng SHA-256 là duplicates, còn lại similar
    folder = 'duplicates' if reason == "SHA-256" else 'similar'
    target_path = os.path.join(OUTPUT_BASE, folder, os.path.basename(delete))
    
    try:
        shutil.move(delete, target_path)
        duplicate_log.append({
            'kept_path': keep, # Đường dẫn tuyệt đối file giữ lại
            'kept_name': os.path.basename(keep), # Tên file giữ lại
            'kept_score': score_keep, # Điểm file giữ lại
            'del_path': target_path, # Đường dẫn tuyệt đối của bị xoá
            'del_name': os.path.basename(delete), # Tên file bị xoá
            'del_score': score_del, # Điểm của file bị xoá
            'reason': reason, # Lý do xoá
            'del_origin': delete # Đường dẫn tuyệt đối của file trước khi bị xoá
        })
        return delete
    except Exception as e: 
        print(f"Lỗi khi di chuyển file {delete}: {e}")
        return None

# ================= BƯỚC 1: LỌC CHẤT LƯỢNG (QUALITY CHECK) =================
def check_image_quality(image_path: str = "") -> Tuple[str, str, float]:
    """Hàm worker kiểm tra chất lượng 1 ảnh."""
    try:
        # Lấy ảnh đen/trắng (Bỏ lớp thứ 3 của ảnh)
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Không đọc được ảnh
        if img_gray is None: 
            return image_path, 'error', 0.0

        # Tính độ nét bằng Laplacian
        blur_score = cv2.Laplacian(img_gray, cv2.CV_64F).var()

        if blur_score < BLUR_THRESHOLD: 
            return image_path, 'blur', blur_score
        
        # Độ sáng của ảnh được tính bằng trung bình cộng của giá trị từng pixel chạy từ 0-255 (đen trắng)
        mean_brightness = np.mean(img_gray)
        if mean_brightness < DARK_THRESHOLD: 
            return image_path, 'dark', mean_brightness
        if mean_brightness > BRIGHT_THRESHOLD: 
            return image_path, 'bright', mean_brightness

        return image_path, 'ok', blur_score
    except: 
        return image_path, 'error', 0.0

def scan_and_filter_quality(all_images_path: List[str] = None) -> Tuple[List[str], List[Dict]]:
    """Đa xử lý (Multiprocessing) để lọc chất lượng ảnh."""

    # Đường dẫn tuyệt đối ảnh đủ điều kiện -> Trả về
    clean_images = []
    # Thông tin của ảnh kém chất lượng -> Trả về
    quality_log = []
    
    print(f"\n🧹 [Bước 1] Kiểm tra chất lượng ảnh (Sử dụng {WORKERS} nhân CPU)...")
    
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        future_results = executor.map(check_image_quality, all_images_path, chunksize=BATCH_SIZE)
        
        # Duyệt qua lô BATCH_SIZE của mỗi WORKERS
        for filepath, status, score in tqdm(future_results, total=len(all_images_path), desc="Filtering"):
            if status == 'ok':
                clean_images.append(filepath)
            # Ảnh không đủ chất lượng nhưng không phải lỗi
            elif status != 'error':
                try:
                    # lấy 'status' làm đích -> thư mục sẽ di chuyển ảnh tới
                    target_folder = os.path.join(OUTPUT_BASE, status)
                    # Tên file
                    filename = os.path.basename(filepath)
                    # thư mục sẽ di chuyển ảnh tới + tên file ==> Đường dẫn tuyệt đối tương lai
                    target_path = os.path.join(target_folder, filename)
                    shutil.move(filepath, target_path)
                    # Ghi LOG lại để report
                    quality_log.append({
                        'name': filename, # Tên file
                        'path': target_path, # Đường dẫn tuyệt đối thực tế sau khi di chuyển
                        'reason': status.upper(), # Điều kiện bị loại
                        'score': score # Số điểm 
                    })
                except Exception as e:
                    print(f"Lỗi: {e}")
    # clean_images: Ảnh vượt qua bài kiểm tra
    # quality_log: Thông tin của ảnh kém chất lượng đã bị di chuyển
    return clean_images, quality_log

# ================= BƯỚC 2: HASHING DEDUPLICATION =================
def compute_all_hashes(filepath: str) -> Tuple[str, str, str, str]:
    """Hàm worker tính gộp 3 loại hash."""
    try:
        # SHA-256: Trừ khi coppy paste, không bao giờ trùng
        sha = calculate_file_hash(filepath)
        if sha is None: 
            return filepath, None, None, None

        img = Image.open(filepath)
        d_hash = str(imagehash.dhash(img))
        p_hash = str(imagehash.phash(img))
        
        return filepath, sha, d_hash, p_hash
    except Exception as e:
        print("Không tính Hashing của ảnh được - {e}")
        return filepath, None, None, None

def find_duplicates_by_hashing(image_paths: List[str]) -> Tuple[Set[str], List[Dict]]:
    """Lọc trùng bằng Hashing (Map-Reduce)."""
    # Lưu vân tay của các ảnh
    hashes_sha, hashes_d, hashes_p = {}, {}, {}
    deleted = set()
    dup_log = []
    
    print(f"\n⚡ [Bước 2] Quét trùng lặp Hashing (Sử dụng {WORKERS} nhân CPU)...")

    results_cache = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        future_results = executor.map(compute_all_hashes, image_paths, chunksize=BATCH_SIZE)
        
        for res in tqdm(future_results, total=len(image_paths), desc="Hashing Calculation"):
            # Nếu tính được SHA-256 -> Không lỗi -> Lưu cache
            if res[1] is not None:
                results_cache.append(res)

    # Duyệt cache
    for f, sha, dh, ph in tqdm(results_cache, desc="Filtering Logic"):
        # 1. SHA-256
        # Nếu SHA-256 đã tồn tại -> 99.99% Ảnh coppy paste
        if sha in hashes_sha:
            # Tính toán độ nét -> Trả ra đường dẫn ảnh thấp điểm hơn
            del_path = process_duplicate_pair(hashes_sha[sha], f, dup_log, "SHA-256")
            if del_path: 
                deleted.add(del_path)
                continue 
        else:
            hashes_sha[sha] = f

        # 2. dHash
        # Nếu dHash đã tồn tại
        if dh in hashes_d:
            # Lấy đường dẫn của ảnh có dHash đã tồn tại trước đó
            existing_path = hashes_d[dh]
            # Trường hợp SHA-256 move đi trước rồi
            if not os.path.exists(existing_path):
                # Cập nhật lại: Với dHash cũ, gắn đường dẫn ảnh mới
                hashes_d[dh] = f
            # Ảnh vẫn tồn tại
            else:
                # Lấy ảnh trước và ảnh hiện tại đi so độ nét -> trả ra đường dẫn của ảnh bị move
                del_path = process_duplicate_pair(existing_path, f, dup_log, "dHash")
                if del_path: 
                    deleted.add(del_path)
                    # Xui sao ảnh trước kém chất lượng hơn ảnh hiện tại
                    if del_path == existing_path: 
                        # Cập nhật lại: Với dHash cũ, gắn đường dẫn ảnh mới
                        hashes_d[dh] = f
                    continue
        else:
            hashes_d[dh] = f

        # 3. pHash
        # Nếu pHash đã tồn tại
        if ph in hashes_p:
            # Lấy đường dẫn của ảnh có pHash đã tồn tại trước đó
            existing_path = hashes_p[ph]
            # Trường hợp SHA-256/dHash move đi trước rồi
            if not os.path.exists(existing_path):
                # Cập nhật lại: Với pHash cũ, gắn đường dẫn ảnh mới
                hashes_p[ph] = f
            else:
                # Lấy ảnh trước và ảnh hiện tại đi so độ nét -> trả ra đường dẫn của ảnh bị move
                del_path = process_duplicate_pair(existing_path, f, dup_log, "pHash")
                if del_path: 
                    deleted.add(del_path)
                    # Xui sao ảnh trước kém chất lượng hơn ảnh hiện tại
                    if del_path == existing_path: 
                        # Cập nhật lại: Với pHash cũ, gắn đường dẫn ảnh mới
                        hashes_p[ph] = f
        else:
            hashes_p[ph] = f
    
    # deleted: Đường dẫn ảnh đã bị xoá
    # dup_log: LOG
    return deleted, dup_log

# ================= BƯỚC 3: DEEP LEARNING EMBEDDING =================
class VehicleDataset(Dataset):
    """
    Dataset tùy chỉnh để nạp và tiền xử lý ảnh xe cộ cho mô hình Deep Learning.

    Lớp này kế thừa từ `torch.utils.data.Dataset`, chịu trách nhiệm chuẩn bị dữ liệu 
    để đưa vào mô hình ResNet/FastReID.

    Attributes:
        image_paths (List[str]): Danh sách đường dẫn tuyệt đối của ảnh.
        transform (T.Compose): Chuỗi các bước biến đổi ảnh (Resize -> ToTensor -> Normalize).
    """

    def __init__(self, image_paths: List[str]):
        """
        Khởi tạo dataset với danh sách đường dẫn ảnh.

        Args:
            image_paths (List[str]): Danh sách các đường dẫn file ảnh đầu vào.
        """
        self.image_paths = image_paths
        self.transform = T.Compose([
            # Resize về kích thước chuẩn của model (thường là 256x256 hoặc 256x128 tùy config)
            T.Resize((256, 256)),
            # Chuyển đổi từ ảnh PIL sang Tensor và đưa về khoảng [0, 1]
            T.ToTensor(),
            # Chuẩn hóa theo thống kê của ImageNet (Mean & Std) giúp model hội tụ nhanh hơn
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        """Trả về tổng số lượng ảnh trong dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Đọc và xử lý một ảnh tại vị trí index cụ thể.

        Args:
            idx (int): Chỉ số của ảnh trong danh sách.

        Returns:
            Tuple[torch.Tensor, str]: 
                - Tensor ảnh đã qua xử lý (C, H, W).
                - Đường dẫn gốc của file ảnh (để truy vết sau này).
                - Trả về (None, path) nếu file ảnh bị lỗi không đọc được.
        """
        path = self.image_paths[idx]
        try:
            # .convert("RGB") để đảm bảo ảnh luôn có 3 kênh màu (xử lý ảnh xám hoặc PNG trong suốt)
            img = Image.open(path).convert("RGB")
            return self.transform(img), path
        except Exception as e:
            # Trả về None để collate_fn lọc bỏ
            return None, path

def collate_fn(batch: List) -> Tuple[torch.Tensor, List[str]]:
    """
    Hàm gom nhóm (Collate) tùy chỉnh để xử lý các ảnh bị lỗi khi nạp dữ liệu.

    Mặc định DataLoader sẽ lỗi nếu một trong các mẫu là None. Hàm này giúp lọc bỏ 
    các mẫu None đó trước khi đóng gói thành Batch.

    Args:
        batch (List): Danh sách các mẫu dữ liệu trả về từ `__getitem__`.

    Returns:
        Tuple[torch.Tensor, List[str]]: 
            - Batch Tensor (N, C, H, W).
            - List đường dẫn ảnh tương ứng.
            - Trả về (None, None) nếu toàn bộ batch bị lỗi.
    """
    # Lọc bỏ các phần tử mà img (x[0]) là None
    batch = list(filter(lambda x: x[0] is not None, batch))
    
    # Nếu lọc xong mà không còn gì (batch rỗng) -> Báo hiệu bỏ qua
    if not batch: return None, None
    
    # Sử dụng hàm collate mặc định của PyTorch cho các dữ liệu sạch
    return torch.utils.data.dataloader.default_collate(batch)

def setup_fastreid_model() -> torch.nn.Module:
    """
    Khởi tạo, cấu hình và nạp trọng số cho mô hình FastReID.

    Quy trình:
    1. Nạp cấu hình từ file YAML.
    2. Áp dụng workaround `DEVICE="cpu"` để vượt qua kiểm tra khởi tạo trên macOS.
    3. Xây dựng kiến trúc mạng (Backbone + Head).
    4. Nạp trọng số (Weights) đã train.
    5. Chuyển sang chế độ đánh giá (Eval) và đẩy sang thiết bị (MPS/CUDA).

    Returns:
        torch.nn.Module: Mô hình Deep Learning đã sẵn sàng hoạt động.
    """
    # Lấy cấu hình mặc đinh từ nhà sản xuất
    cfg = get_cfg()
    # Ghi đè cấu hình của mình vào
    cfg.merge_from_file(CONFIG_FILE)
    
    # WORKAROUND: Ép khởi tạo trên CPU để tránh lỗi backend CUDA trên máy Mac
    cfg.MODEL.DEVICE = "cpu" 
    # Build model từ cấu hình nhà sản xuất + của mình custom lại
    model = build_model(cfg)

    if os.path.exists(WEIGHTS_PATH):
        # Nạp kiến thức Weights vào
        Checkpointer(model).load(WEIGHTS_PATH)
    else:
        print(f"❌ LỖI: Không tìm thấy file weights tại {WEIGHTS_PATH}.")
        exit()
        
    model.eval() # Tắt Dropout, Batch Norm dynamic
    model.to(DEVICE) # Đẩy sang GPU/MPS thực tế
    return model

def extract_features(clean_images: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Trích xuất vector đặc trưng (Feature Embedding) từ danh sách ảnh.

    Hàm này thực hiện toàn bộ quy trình Inference:
    1. Tạo DataLoader với đa luồng (Workers) và bộ nhớ ghim (Pin Memory).
    2. Chạy mô hình để lấy vector thô.
    3. Chuẩn hóa vector L2 bằng NumPy (Thay thế FAISS để tránh xung đột bộ nhớ trên Mac).
    4. Lưu trữ kết quả `features.npy` và `paths.npy` xuống ổ cứng.

    Args:
        clean_images (List[str]): Danh sách đường dẫn ảnh đầu vào.

    Returns:
        Tuple[np.ndarray, List[str]]: 
            - features_matrix: Ma trận vector đặc trưng (N, 2048) kiểu float32.
            - all_paths: Danh sách đường dẫn tương ứng 1-1 với ma trận.
    """
    print(f'✨ [Bước 3] Trích xuất đặc trưng Deep Learning ({len(clean_images)} ảnh)...')
    
    # Sort để tất cả lần chạy đều giống nhau
    clean_images = sorted(list(set(clean_images)))
    # Số lượng ảnh để khai báo cấp phát bộ nhớ (Đầu vào lớn mà không cấp phát trước dễ bị crash)
    num_images = len(clean_images) 
    
    # Build model
    model = setup_fastreid_model()
    # Tạo 1 bộ dataset: phân lô batch_size, resize, ...
    dataset = VehicleDataset(clean_images)
    
    # Cấu hình worker load ảnh
    loader_workers = WORKERS
    print(f"   ⚙️  Cấu hình: {loader_workers} Workers | Device: {DEVICE}")

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # Duyệt tuần tự, không xáo trộn ảnh
        num_workers=loader_workers, 
        collate_fn=collate_fn, # Bộ lọc hàng lỗi
        # pin_memory=True: ra lệnh cho hệ điều hành: "Cấp cho tao vùng RAM này và GHIM CHẶT nó lại, cấm di chuyển!".
        pin_memory=True if torch.cuda.is_available() or torch.backends.mps.is_available() else False, 
        # prefetch_factor=2: Trong lúc GPU đang xử lý Lô 1, CPU bắt Workers đi lấy ảnh và đóng gói Lô 2 và 3 để chuẩn bị.
        prefetch_factor=2 if loader_workers > 0 else None,
        # persistent_workers=False: Làm xong 1 epoch, kill process hết luồng. Epoch sau khởi tạo lại -> Tốn thời gian khởi động.
        # persistent_workers=True: Giữ luồng đó sống, ngồi chờ lệnh tiếp theo. Đỡ tốn công khởi tạo lại tiến trình (Process).
        persistent_workers=True if loader_workers > 0 else False,
    )
    
    # Cấp phát trước bộ nhớ để tránh phân mảnh.
    # np.zeros(số ảnh, số chiều vector): Tạo ra 1 ma trận với kích thước đã tạo với các giá trị mặc định = 0 -> sau xử lý xong lô nào chỉ cần bỏ vào thôi
    # FAISS được build bằng C++, nó chỉ hiểu tới float32.
    # Fload 64 là lỗi -> nên phải én về f32
    features_matrix = np.zeros((num_images, 2048), dtype='float32')
    all_paths = []
    start_idx = 0 

    # Tắt chế độ train
    with torch.no_grad():
        # imgs: 1 lô ảnh
        # paths: 1 lô đường dẫn
        for imgs, paths in tqdm(dataloader, desc="Embedding"):
            if imgs is None: 
                continue
            
            # Non_blocking giúp CPU không phải chờ GPU copy xong dữ liệu (chả hiểu cái gì)
            # Giúp CPU và việc truyền dữ liệu diễn ra song song (overlap), che giấu độ trễ đường truyền. (Vẫn ko hiểu lắm)... Kệ đi
            imgs = imgs.to(DEVICE, non_blocking=True)
            # Trích xuất đặc trưng của lô ảnh
            feats = model(imgs)
            
            # Flatten: Nếu đầu ra là khối lập phương (Batch, 2048, 1, 1) -> ép dẹp thành tờ giấy (Batch, 2048)
            if len(feats.shape) > 2: 
                feats = feats.view(feats.size(0), -1)
            # Chuyển về CPU để lưu trữ (vì RAM rẻ hơn VRAM)
            batch_feats = feats.cpu().numpy()
            batch_size = batch_feats.shape[0]
            
            # Điền vào đúng vị trí phòng trong khách sạn đã xây sẵn (features_matrix = np.zeros((num_images, 2048), dtype='float32') ở trên)
            end_idx = start_idx + batch_size
            features_matrix[start_idx:end_idx, :] = batch_feats
            all_paths.extend(paths)
            start_idx = end_idx
            
    # Cắt bỏ phần thừa nếu có ảnh lỗi bị loại bỏ (Có nhưng rất hiếm)
    if len(all_paths) < num_images:
        features_matrix = features_matrix[:len(all_paths)]
    
    # Thường là bị lỗi mới dính đk này
    if len(all_paths) == 0: 
        return None, None
    
    # --- CHUẨN HÓA L2 (NumPy Implementation) ---
    # An toàn tuyệt đối cho macOS, thay thế cho faiss.normalize_L2
    print("   📐 Đang chuẩn hóa L2 (Numpy)...")
    # Tính độ dài vector
    # features_matrix: ma trận kích thước (ảnh, chiều vector) (ở trên)
    # axis=1: tính toán theo chiều ngang (từng dòng/từng ảnh).  ==> Thôi khúc này tra google đi (Nhức đầu quá)
    norm = np.linalg.norm(features_matrix, axis=1, keepdims=True)
    # Chia vector cho độ dài (+1e-10 để tránh chia cho 0)
    features_matrix = features_matrix / (norm + 1e-10)
    features_matrix = features_matrix.astype('float32')
    
    # Lưu kết quả
    out_dir = os.path.join(OUTPUT_BASE, "output_features")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "features.npy"), features_matrix)
    np.save(os.path.join(out_dir, "paths.npy"), all_paths)
    
    return features_matrix, all_paths

# ================= BƯỚC 5: FAISS CLUSTERING =================
def cluster_and_filter_faiss(features: np.ndarray, paths: List[str], duplicate_log: List[Dict]) -> int:
    """
    Phân cụm và lọc trùng lặp sử dụng FAISS kết hợp Lý thuyết đồ thị (Graph Theory).

    Chiến lược: "Detail Priority" (Ưu tiên chi tiết).
    1. **Tìm kiếm (FAISS):** Tìm tất cả các cặp ảnh có độ tương đồng >= Threshold.
    2. **Gom nhóm (NetworkX):** Xây dựng đồ thị và tìm các thành phần liên thông (nhóm ảnh trùng).
    3. **Chọn lọc (Keeper Selection):** Trong mỗi nhóm, chọn ảnh giữ lại dựa trên:
       - Ưu tiên 1: Điểm chi tiết cao nhất (Canny Edge) - Để giữ lại ảnh rõ biển số/góc cạnh.
       - Ưu tiên 2: Phải đạt độ nét tối thiểu (Blur Threshold).
    4. **Kiểm chứng (Re-check):** Tính lại Cosine Similarity giữa Keeper và Candidate trước khi xóa
       để tránh lỗi bắc cầu trong đồ thị.

    Args:
        features (np.ndarray): Ma trận đặc trưng đã chuẩn hóa (N, 2048).
        paths (List[str]): Danh sách đường dẫn ảnh.
        duplicate_log (List[Dict]): List để ghi lại nhật ký xóa.

    Returns:
        int: Số lượng ảnh đã bị loại bỏ.
    """
    print(f"\n✨ [Bước 5] Gom nhóm ảnh trùng bằng FAISS (Threshold={THRESHOLD_FAISS} - Detail Priority)...")
    
    # 1. Tạo Index và tìm kiếm
    # Lấy chiều vector của đặc trưng (ResNet50 == 2048)
    d = features.shape[1]
    # Khai báo 1 không gian lưu trữ
    # IndexFlat: Cấu trúc phẳng -> Lưu trữ nguyên bản, tìm kiếm vét cạn -> Không cắt gọt gì (Raw)
    # IP: tích vô hướng
    index = faiss.IndexFlatIP(d) 
    # Nạp các đặc trưng vào
    index.add(features)
    
    # Range Search: Tìm tất cả hàng xóm trong bán kính Threshold
    lims, D, I = index.range_search(features, THRESHOLD_FAISS)
    
    # 2. Xây dựng đồ thị kết nối
    G = nx.Graph()

    # Nếu có 100k ảnh => Tạo 100k node vào đồ thị
    G.add_nodes_from(range(len(paths)))
    
    # Tìm hàng xóm
    for i in tqdm(range(len(paths)), desc="Building Graph"):
        start, end = lims[i], lims[i+1]
        for j in range(start, end):
            if i != I[j]: # Không tự nối với chính mình
                G.add_edge(i, I[j])

    # Tìm các nhóm liên thông (Connected Components)
    components = list(nx.connected_components(G))
    # Lấy ra những cụm có 2 ảnh trở lên 
    duplicate_groups = [c for c in components if len(c) > 1]
    
    deleted_count = 0
    metrics_cache = {} 

    # Hàm helper để lấy chỉ số ảnh (có cache)
    def get_metrics(idx):
        if idx not in metrics_cache:
            p = paths[idx]
            metrics_cache[idx] = {
                'detail': calculate_detail_score(p),
                'sharpness': calculate_sharpness(p)
            }
        return metrics_cache[idx]

    # 3. Duyệt và lọc từng nhóm
    for component in tqdm(duplicate_groups, desc="AI Filtering"):
        comp_list = list(component)
        candidates = []
        
        # Lấy thông tin chi tiết của tất cả ảnh trong nhóm
        for idx in comp_list:
            m = get_metrics(idx)
            candidates.append({
                'idx': idx, 'detail': m['detail'], 'sharpness': m['sharpness']
            })
        
        # Sắp xếp giảm dần theo độ chi tiết
        candidates.sort(key=lambda x: x['detail'], reverse=True)
        
        # Chọn Keeper: Mặc định là ảnh chi tiết nhất, nhưng phải đủ nét
        keeper_candidate = candidates[0] 
        for cand in candidates:
            if cand['sharpness'] >= BLUR_THRESHOLD:
                keeper_candidate = cand
                break
        
        keeper_idx = keeper_candidate['idx']
        keeper_vec = features[keeper_idx]
        keeper_path = paths[keeper_idx]
        keeper_score_log = keeper_candidate['detail'] 
        
        # Danh sách các ảnh cần xem xét xóa (tất cả trừ Keeper)
        duplicates_idx = [x['idx'] for x in candidates if x['idx'] != keeper_idx]
        
        for del_idx in duplicates_idx:
            # 4. Kiểm chứng lần cuối (Direct Check)
            candidate_vec = features[del_idx]
            sim = np.dot(keeper_vec, candidate_vec)
            
            if sim >= THRESHOLD_FAISS:
                del_path = paths[del_idx]
                target_path = os.path.join(OUTPUT_BASE, "similar", os.path.basename(del_path))
                try:
                    shutil.move(del_path, target_path)
                    
                    # Ghi log chi tiết
                    duplicate_log.append({
                        'kept_path': keeper_path, 
                        'kept_name': os.path.basename(keeper_path), 
                        'kept_score': keeper_score_log,
                        'del_path': target_path, 
                        'del_name': os.path.basename(del_path), 
                        'del_score': get_metrics(del_idx)['detail'],
                        'reason': f"AI: {sim * 100:.2f}%", 
                        'del_origin': del_path
                    })
                    deleted_count += 1
                except: pass

    return deleted_count

# ================= REPORTING =================
def generate_html_report(duplicate_log: list, quality_log: list, output_file: str = "Wow_Report.html", total_input: int = 0):
    """
    Phiên bản V3.1 (Fixed):
    - Fix bug: SHA-256 bị nhận nhầm là AI.
    - Giữ nguyên các tính năng xịn xò của V3.
    """
    
    # --- 0. HELPER: PATH TRACING & CONSTANTS ---
    redirect_map = {}
    
    # Map đường dẫn bị thay đổi từ duplicate log
    for item in duplicate_log:
        if 'del_origin' in item and 'del_path' in item:
            redirect_map[item['del_origin']] = item['del_path']

    def resolve_final_path(path):
        """Đệ quy tìm đường dẫn cuối cùng của file."""
        if os.path.exists(path): return path
        current_check = path
        visited = set()
        while current_check in redirect_map:
            if current_check in visited: break
            visited.add(current_check)
            current_check = redirect_map[current_check]
            if os.path.exists(current_check): return current_check
        return path

    # --- 1. PHÂN LOẠI & THỐNG KÊ CHI TIẾT ---
    stats = {
        "blur": 0, "dark": 0, "bright": 0, 
        "ai_dup": 0, "hash_dup": 0, 
        "total_removed": 0
    }

    categories = {
        "blur": {"data": [], "id": "section-blur", "title": "Ảnh Mờ (Blur)"},
        "dark": {"data": [], "id": "section-dark", "title": "Ảnh Tối/Sáng"},
        "ai_dup": {"groups": {}, "id": "section-ai", "title": "AI Duplicates"},
        "hash_dup": {"groups": {}, "id": "section-hash", "title": "Hash Duplicates"}
    }

    # 1.1 Xử lý Quality Log
    for item in quality_log:
        reason = item.get('reason', '').upper()
        real_path = resolve_final_path(item.get('path', ''))
        file_exists = os.path.exists(real_path)
        item_data = {**item, 'path': real_path, 'file_exists': file_exists}

        if "BLUR" in reason:
            categories["blur"]["data"].append(item_data)
            stats["blur"] += 1
        elif "DARK" in reason:
            categories["dark"]["data"].append(item_data)
            stats["dark"] += 1
        elif "BRIGHT" in reason:
            categories["dark"]["data"].append(item_data)
            stats["bright"] += 1
        else:
            categories["blur"]["data"].append(item_data)
            stats["blur"] += 1

    # 1.2 Xử lý Duplicate Log (FIXED LOGIC HERE)
    for item in duplicate_log:
        reason = item.get('reason', '').upper()
        final_kept = resolve_final_path(item.get('kept_path', ''))
        final_del = resolve_final_path(item.get('del_path', ''))
        
        item_data = {
            **item, 'kept_path': final_kept, 'del_path': final_del,
            'kept_exists': os.path.exists(final_kept),
            'del_exists': os.path.exists(final_del)
        }

        # --- SỬA LỖI TẠI ĐÂY ---
        # Thêm điều kiện check "SHA" để bắt được SHA-256
        is_hash = ("HASH" in reason or "EXACT" in reason or "SHA" in reason)
        cat_key = "hash_dup" if is_hash else "ai_dup"
        # -----------------------
        
        # Gom nhóm
        group_key = final_kept
        if group_key not in categories[cat_key]["groups"]:
            categories[cat_key]["groups"][group_key] = {
                "kept_info": {
                    "path": final_kept,
                    "name": item.get('kept_name', os.path.basename(final_kept)),
                    "score": item.get('kept_score', 0),
                    "exists": os.path.exists(final_kept)
                },
                "deleted_items": []
            }
        categories[cat_key]["groups"][group_key]["deleted_items"].append(item_data)
        
        if is_hash: stats["hash_dup"] += 1
        else: stats["ai_dup"] += 1

    stats["total_removed"] = sum(stats.values())
    if total_input == 0: total_input = stats["total_removed"]
    survivors = max(0, total_input - stats["total_removed"])

    # --- 2. RENDER HELPERS ---
    
    def render_lazy_img(src, exists, css_class=""):
        if not exists: return f'<div class="missing-box {css_class}">🚫 Missing</div>'
        placeholder = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        return f'<img src="{placeholder}" data-src="{html.escape(src)}" class="lazy-load {css_class}" loading="lazy" alt="img">'

    def render_quality_card(item, type_badge):
        score = float(item.get('score', 0))
        reason = item.get('reason', 'UNK')
        
        if "BLUR" in reason: badge_color = "#fbbf24"; text_color = "#000"; 
        elif "DARK" in reason: badge_color = "#4b5563"; text_color = "#fff"; 
        elif "BRIGHT" in reason: badge_color = "#f3f4f6"; text_color = "#000"; 
        else: badge_color = "#ef4444"; text_color = "#fff"; 

        img_html = render_lazy_img(item['path'], item['file_exists'], "card-img")
        
        return f"""
        <div class="card fade-in">
            <div class="card-image-container" onclick="openLightbox('{html.escape(item['path'])}')">
                {img_html}
                <div class="stat-badge" style="background: {badge_color}; color: {text_color}">
                    {reason} <span style="opacity:0.8">|</span> {score:.1f}
                </div>
                <div class="card-name-overlay" title="{html.escape(item['name'])}">
                    {html.escape(item['name'])}
                </div>
            </div>
        </div>
        """

    def render_group_row(group_data):
        kept = group_data["kept_info"]
        deleted_list = group_data["deleted_items"]
        
        kept_html = f"""
        <div class="kept-column">
            <div class="status-label kept-label">GIỮ LẠI (KEPT)</div>
            <div class="img-wrapper main-img" onclick="openLightbox('{html.escape(kept['path'])}')">
                {render_lazy_img(kept['path'], kept['exists'])}
            </div>
            <div class="meta-info">
                <div class="filename" title="{kept['name']}">{kept['name']}</div>
                <div class="score-bar">Score: <strong style="color: #00ff88">{float(kept['score']):.1f}</strong></div>
            </div>
        </div>
        """

        del_items_html = ""
        for item in deleted_list:
            d_score = float(item.get('del_score', 0))
            diff = float(kept['score']) - d_score
            del_items_html += f"""
            <div class="del-item-card" onclick="openLightbox('{html.escape(item['del_path'])}')">
                <div class="del-img-box">
                    {render_lazy_img(item['del_path'], item['del_exists'])}
                    <div class="overlay-reason">{item['reason']}</div>
                </div>
                <div class="del-meta">
                    <div class="score-mini">{d_score:.1f} <span class="diff">(-{diff:.1f})</span></div>
                </div>
            </div>
            """

        return f"""
        <div class="group-row fade-in">
            {kept_html}
            <div class="arrow-container">
                <div class="arrow-icon">➔</div>
                <div class="clean-count">Cleaned: {len(deleted_list)}</div>
            </div>
            <div class="deleted-column">
                <div class="status-label del-label">ĐÃ XÓA ({len(deleted_list)})</div>
                <div class="del-grid">{del_items_html}</div>
            </div>
        </div>
        """

    # --- 3. HTML TEMPLATE ---
    html_content = f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>⚡ Cleaning Report V3.1</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #0f172a; --sidebar: #1e293b; --card-bg: #1e293b;
                --text-main: #f8fafc; --text-sub: #94a3b8;
                --primary: #3b82f6; --success: #10b981; --danger: #ef4444; --warning: #f59e0b;
                --bright: #e2e8f0;
            }}
            * {{ box-sizing: border-box; }}
            body {{ margin: 0; font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }}
            
            .sidebar {{ width: 260px; background: var(--sidebar); padding: 20px; display: flex; flex-direction: column; border-right: 1px solid rgba(255,255,255,0.05); z-index: 10; }}
            .logo {{ font-size: 1.5rem; font-weight: 800; background: linear-gradient(45deg, var(--primary), var(--success)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 30px; }}
            .nav-item {{ padding: 12px; border-radius: 8px; cursor: pointer; color: var(--text-sub); display: flex; justify-content: space-between; margin-bottom: 5px; transition: 0.2s; }}
            .nav-item:hover, .nav-item.active {{ background: rgba(255,255,255,0.05); color: #fff; }}
            .nav-item.active {{ border-left: 3px solid var(--primary); background: linear-gradient(90deg, rgba(59,130,246,0.1), transparent); }}
            .badge {{ background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; }}

            .main {{ flex: 1; overflow-y: auto; padding: 0; position: relative; scroll-behavior: smooth; }}
            .section {{ display: none; padding: 40px; }}
            .section.active {{ display: block; animation: fadeIn 0.3s ease; }}

            .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-top: 20px; }}
            .stat-box {{ background: var(--card-bg); padding: 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); text-align: center; }}
            .stat-box.big {{ grid-column: span 2; background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(16,185,129,0.1)); border: 1px solid rgba(59,130,246,0.2); }}
            .stat-num {{ font-size: 2.5rem; font-weight: 800; margin-bottom: 5px; color: var(--text-main); }}
            .stat-label {{ color: var(--text-sub); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }}

            .grid-container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }}
            .card {{ background: rgba(255,255,255,0.03); border-radius: 12px; overflow: hidden; position: relative; }}
            .card-image-container {{ height: 220px; position: relative; background: #000; cursor: zoom-in; }}
            
            .stat-badge {{ 
                position: absolute; top: 10px; left: 10px; z-index: 5;
                padding: 4px 10px; border-radius: 6px; 
                font-size: 0.75rem; font-weight: 800; 
                box-shadow: 0 4px 10px rgba(0,0,0,0.5);
                font-family: 'JetBrains Mono', monospace;
            }}
            .card-name-overlay {{
                position: absolute; bottom: 0; left: 0; width: 100%;
                background: linear-gradient(to top, rgba(0,0,0,0.9), transparent);
                color: white; padding: 30px 10px 10px 10px;
                font-size: 0.8rem; font-weight: 600;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }}

            .group-row {{ display: flex; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 20px; margin-bottom: 30px; gap: 20px; align-items: stretch; }}
            .kept-column {{ width: 250px; display: flex; flex-direction: column; }}
            .deleted-column {{ flex: 1; display: flex; flex-direction: column; background: rgba(0,0,0,0.2); border-radius: 12px; padding: 15px; }}
            .arrow-container {{ display: flex; flex-direction: column; justify-content: center; align-items: center; width: 50px; color: var(--text-sub); }}
            
            .img-wrapper.main-img {{ height: 200px; border: 2px solid var(--success); border-radius: 12px; overflow: hidden; margin-bottom: 10px; cursor: zoom-in; }}
            
            .del-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; }}
            .del-item-card {{ background: rgba(255,255,255,0.05); border-radius: 8px; overflow: hidden; cursor: zoom-in; transition: transform 0.2s; }}
            .del-item-card:hover {{ transform: translateY(-3px); }}
            .del-img-box {{ height: 80px; position: relative; }}
            .overlay-reason {{ position: absolute; bottom: 0; width: 100%; background: rgba(239, 68, 68, 0.9); color: white; font-size: 0.6rem; text-align: center; }}
            .del-meta {{ padding: 5px; text-align: center; }}
            .score-mini {{ font-size: 0.65rem; color: var(--text-sub); }}
            .diff {{ color: var(--danger); font-weight: bold; }}

            img.lazy-load {{ opacity: 0; transition: opacity 0.5s; width: 100%; height: 100%; object-fit: cover; }}
            img.lazy-load.loaded {{ opacity: 1; }}
            .status-label {{ font-size: 0.7rem; font-weight: 800; padding: 4px 8px; border-radius: 4px; margin-bottom: 10px; display: inline-block; }}
            .kept-label {{ background: rgba(16, 185, 129, 0.2); color: var(--success); }}
            .del-label {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); }}
            .meta-info .filename {{ font-size: 0.8rem; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
            
            #lightbox {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.95); z-index: 9999; display: none; justify-content: center; align-items: center; }}
            #lightbox img {{ max-width: 95%; max-height: 95%; box-shadow: 0 0 30px rgba(0,0,0,0.5); }}
            .close-lb {{ position: absolute; top: 20px; right: 30px; font-size: 3rem; color: white; cursor: pointer; }}

            @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        </style>
    </head>
    <body>

        <div class="sidebar">
            <div class="logo">🚀 CLEANER V3.1</div>
            <div class="nav-item active" onclick="switchTab('section-summary', this)">
                <span>Overview</span>
            </div>
            <div style="height:1px; background:rgba(255,255,255,0.1); margin:15px 0;"></div>
            <div class="nav-item" onclick="switchTab('section-blur', this)">
                <span>Blurry</span> <span class="badge" style="color:var(--warning)">{len(categories['blur']['data'])}</span>
            </div>
            <div class="nav-item" onclick="switchTab('section-dark', this)">
                <span>Dark/Bright</span> <span class="badge" style="color:var(--bright)">{len(categories['dark']['data'])}</span>
            </div>
            <div style="height:1px; background:rgba(255,255,255,0.1); margin:15px 0;"></div>
            <div class="nav-item" onclick="switchTab('section-ai', this)">
                <span>AI Duplicates</span> <span class="badge" style="color:var(--primary)">{stats['ai_dup']}</span>
            </div>
            <div class="nav-item" onclick="switchTab('section-hash', this)">
                <span>Hash Duplicates</span> <span class="badge" style="color:var(--success)">{stats['hash_dup']}</span>
            </div>
        </div>

        <div class="main">
            <div id="section-summary" class="section active">
                <h1 style="color:var(--primary)">Processing Statistics</h1>
                
                <div class="stats-grid">
                    <div class="stat-box big">
                        <div class="stat-num" style="color: var(--primary)">{total_input}</div>
                        <div class="stat-label">Total Files Input</div>
                    </div>
                    <div class="stat-box big">
                        <div class="stat-num" style="color: var(--success)">{survivors}</div>
                        <div class="stat-label">Clean Files Remaining</div>
                    </div>
                    
                    <div class="stat-box">
                        <div class="stat-num" style="color: var(--warning)">{stats['blur']}</div>
                        <div class="stat-label">Blurry Removed</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num" style="color: var(--danger)">{stats['dark']}</div>
                        <div class="stat-label">Too Dark</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num" style="color: var(--bright)">{stats['bright']}</div>
                        <div class="stat-label">Too Bright</div>
                    </div>
                    <div class="stat-box" style="background:rgba(59,130,246,0.05)">
                         <div class="stat-num" style="font-size: 1.5rem; margin-top:10px">{stats['total_removed']}</div>
                         <div class="stat-label">Total Removed</div>
                    </div>

                    <div class="stat-box big" style="background:rgba(255,255,255,0.02)">
                        <div class="stat-num" style="color: var(--text-sub)">{stats['hash_dup']}</div>
                        <div class="stat-label">Exact Hash Duplicates</div>
                    </div>
                    <div class="stat-box big" style="background:rgba(255,255,255,0.02)">
                        <div class="stat-num" style="color: var(--text-sub)">{stats['ai_dup']}</div>
                        <div class="stat-label">AI Semantic Duplicates</div>
                    </div>
                </div>
            </div>

            <div id="section-blur" class="section">
                <h2>Blurry Images</h2>
                <div class="grid-container">
                    {"".join([render_quality_card(i, "BLUR") for i in categories['blur']['data']])}
                </div>
            </div>

            <div id="section-dark" class="section">
                <h2>Dark / Bright Images</h2>
                <div class="grid-container">
                    {"".join([render_quality_card(i, "DARK") for i in categories['dark']['data']])}
                </div>
            </div>

            <div id="section-ai" class="section">
                <h2>AI Semantic Duplicates</h2>
                <p style="color:var(--text-sub)">AI-detected similar images. The best version is kept.</p>
                {"".join([render_group_row(g) for g in categories['ai_dup']['groups'].values()])}
            </div>

            <div id="section-hash" class="section">
                <h2>Hash Exact Duplicates</h2>
                <p style="color:var(--text-sub)">Bit-by-bit exact copies.</p>
                {"".join([render_group_row(g) for g in categories['hash_dup']['groups'].values()])}
            </div>
        </div>

        <div id="lightbox" onclick="this.style.display='none'">
            <span class="close-lb">&times;</span>
            <img id="lb-img" src="">
        </div>

        <script>
            function switchTab(id, el) {{
                document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
                document.getElementById(id).classList.add('active');
                document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
                el.classList.add('active');
                setTimeout(observeImages, 100);
            }}

            function openLightbox(src) {{
                document.getElementById('lb-img').src = src;
                document.getElementById('lightbox').style.display = 'flex';
            }}

            function observeImages() {{
                const images = document.querySelectorAll('img.lazy-load');
                const observer = new IntersectionObserver((entries, obs) => {{
                    entries.forEach(entry => {{
                        if (entry.isIntersecting) {{
                            const img = entry.target;
                            img.src = img.dataset.src;
                            img.classList.add('loaded');
                            obs.unobserve(img);
                        }}
                    }});
                }}, {{ rootMargin: "200px" }});
                images.forEach(img => observer.observe(img));
            }}
            document.addEventListener('DOMContentLoaded', observeImages);
        </script>
    </body>
    </html>
    """
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ Báo cáo V3.1 (Fixed SHA Logic) đã xong: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"❌ Lỗi ghi file: {e}")
# ================= MAIN =================
def main():
    start_time = time.time()
    setup_folders()

    # Bước 0: Load ảnh từ INPUT
    all_images = get_image_paths()
    if not all_images: 
        print("❌ Không đủ ảnh - dừng chương trình")
        return
    print(f"🔍 Tổng ảnh đầu vào: {len(all_images)}")

    # Bước 1: Lọc chất lượng ảnh
    clean_images, quality_log = scan_and_filter_quality(all_images_path=all_images)
    print(f"📉 Đã lọc được: {len(all_images) - len(clean_images)} ảnh kém chất lượng")

    # Bước 2: Hashing
    deleted_hashing, duplicate_log = find_duplicates_by_hashing(clean_images)
    clean_images = [img for img in clean_images if img not in deleted_hashing]
    print(f"📉 Đã lọc được: {len(clean_images) - len(deleted_hashing)} bằng Hashings\n")

    # Bước 3: Deep Learning
    # features: Các đặc trưng của ảnh
    # paths: Đường dẫn ảnh trùng với đặc trưng
    features, paths = extract_features(clean_images)

    # Bước 4 & 5: FAISS Clustering
    if features is not None and len(paths) > 0:
        deleted_faiss_count = cluster_and_filter_faiss(features, paths, duplicate_log)
        print(f"📉 Đã lọc thêm {deleted_faiss_count} ảnh trùng bằng AI.\n")
    else:
        print("⚠️ Không có feature nào để chạy FAISS.")

    # Bước 6: Report
    generate_html_report(
    duplicate_log, 
    quality_log, 
    os.path.join(OUTPUT_BASE, REPORT_FILE), 
    total_input=len(all_images)  # <--- Thêm tham số này vào
)
    
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
        
    print(f'\n🏁 Thời gian chạy: {time.time() - start_time:.2f} giây')

if __name__ == "__main__":
    # --- THIẾT LẬP QUAN TRỌNG CHO MACOS/LINUX ---
    try:
        # Tránh lỗi malloc error.
        # Thay vì dùng fork mặc định -> chuyển qua dùng spawn
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Không thể chuyển đổi qua spawn")
        pass
    
    # Dành cho Chip M seri:
    # Nếu gặp phép tính nào mà GPU (MPS) không làm được, không báo lỗi. Hãy chuyển phép tính đó về CPU để xử lý, rồi sau đó lại dùng GPU tiếp.
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    main()
