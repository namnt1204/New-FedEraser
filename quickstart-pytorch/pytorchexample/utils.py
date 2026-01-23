import os
import torch
import numpy as np
from typing import List, OrderedDict
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters

def ndarrays_to_vector(ndarrays: List[np.ndarray]) -> torch.Tensor:
    """
    Chuyển đổi danh sách các mảng NumPy (weights của các layer) thành 
    một Tensor 1 chiều duy nhất (Flatten).
    
    Mục đích: Để tính Norm (độ dài) của toàn bộ model/update dễ dàng.
    """
    # 1. Chuyển từng mảng numpy thành tensor và duỗi phẳng (flatten)
    tensors = [torch.from_numpy(arr).view(-1) for arr in ndarrays]
    
    # 2. Nối tất cả lại thành 1 vector dài
    vector = torch.cat(tensors)
    return vector

def vector_to_ndarrays(vector: torch.Tensor, template_ndarrays: List[np.ndarray]) -> List[np.ndarray]:
    """
    Chuyển đổi ngược từ Tensor 1 chiều về danh sách mảng NumPy 
    theo đúng kích thước của model gốc.
    
    Cần 'template_ndarrays' để biết kích thước (shape) của từng layer.
    """
    new_ndarrays = []
    offset = 0
    
    for arr in template_ndarrays:
        # Lấy số lượng phần tử của layer hiện tại
        num_elements = arr.size
        
        # Cắt một đoạn từ vector tương ứng với layer này
        chunk = vector[offset : offset + num_elements]
        
        # Reshape lại cho đúng kích thước layer và chuyển về numpy
        new_arr = chunk.reshape(arr.shape).numpy()
        new_ndarrays.append(new_arr)
        
        # Cập nhật vị trí cắt cho layer tiếp theo
        offset += num_elements
        
    return new_ndarrays

def save_client_updates(save_dir: str, round_num: int, client_id: str, parameters: Parameters):
    """
    Lưu trọng số (Parameters) của Client xuống đĩa cứng dưới dạng file .pt.
    
    Args:
        save_dir: Thư mục gốc để lưu log (ví dụ: 'logs/').
        round_num: Số vòng hiện tại.
        client_id: ID của client.
        parameters: Đối tượng Parameters của Flower gửi về.
    """
    # 1. Tạo đường dẫn thư mục theo vòng: logs/round_1/
    round_dir = os.path.join(save_dir, f"round_{round_num}")
    os.makedirs(round_dir, exist_ok=True)
    
    # 2. Chuyển đổi Flower Parameters -> List[np.ndarray] -> List[Tensor] để lưu bằng torch
    ndarrays = parameters_to_ndarrays(parameters)
    
    # Chúng ta lưu dưới dạng List[Tensor] để sau này load lên dễ tính toán
    tensors = [torch.from_numpy(arr) for arr in ndarrays]
    
    # 3. Định nghĩa tên file và lưu
    filename = f"client_{client_id}.pt"
    filepath = os.path.join(round_dir, filename)
    
    torch.save(tensors, filepath)
    # print(f"Saved update to {filepath}") # Uncomment nếu muốn debug

def load_client_updates(save_dir: str, round_num: int, client_id: str) -> List[np.ndarray]:
    """
    Hàm bổ trợ: Đọc file .pt từ đĩa cứng và trả về List[np.ndarray] 
    để dùng trong Strategy.
    """
    filepath = os.path.join(save_dir, f"round_{round_num}", f"client_{client_id}.pt")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find update file: {filepath}")
        
    # Load list các tensor
    tensors = torch.load(filepath)
    
    # Chuyển về numpy để Flower Strategy sử dụng được
    ndarrays = [t.numpy() for t in tensors]
    return ndarrays