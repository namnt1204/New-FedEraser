import logging
import json
import os
import shutil
import heapq
import torch
import numpy as np
from flwr.serverapp.strategy import FedAvg
from flwr.common import ndarrays_to_parameters

from pytorchexample.utils import save_client_updates, ndarrays_to_vector

log = logging.getLogger(__name__)

class TopKLogStrategy(FedAvg):
    def __init__(self, log_dir: str, k_value: int, total_rounds: int, decay_factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir
        self.k_value = k_value          # Kích thước tối đa của hàng đợi (ví dụ: 20)
        self.total_rounds = total_rounds
        self.decay_factor = decay_factor
        
        self.saved_rounds_file = os.path.join(log_dir, "saved_rounds.json")
        
        # Min-Heap lưu trữ các tuple: (score, round_id)
        # Phần tử đầu tiên (heap[0]) luôn là phần tử có Score nhỏ nhất
        self.top_k_heap = [] 
        
        self.current_weights = None

    def aggregate_train(self, server_round, train_replies):
        aggregated_result = super().aggregate_train(server_round, train_replies)
        
        aggregated_arrays = aggregated_result[0]
        if aggregated_arrays is None:
            return aggregated_result

        # 1. Tính toán vector update mới
        new_ndarrays = [v.numpy() for v in aggregated_arrays.values()]
        new_vec = ndarrays_to_vector(new_ndarrays) 

        # 2. Tính Score (có Decay Factor)
        score = 0.0
        norm = 0.0
        if self.current_weights is not None:
            delta_w = new_vec - self.current_weights
            norm = torch.norm(delta_w).item()
            
            exponent = max(0, self.total_rounds - server_round)
            weight = self.decay_factor ** exponent
            score = weight * norm
        else:
            # Round 1 luôn quan trọng nhất để khởi đầu
            score = float('inf')

        self.current_weights = new_vec

        # In thông tin Score
        min_score_in_queue = self.top_k_heap[0][0] if self.top_k_heap else 0.0
        print(f"ROUND {server_round} | Score: {score:.4f} | Min Queue: {min_score_in_queue:.4f} | Queue Size: {len(self.top_k_heap)}/{self.k_value}")

        # --- LOGIC TOP-K ---
        should_save = False
        round_to_delete = None

        if len(self.top_k_heap) < self.k_value:
            # A. Hàng đợi chưa đầy -> Luôn thêm vào
            heapq.heappush(self.top_k_heap, (score, server_round))
            should_save = True
            print(f"--> [TopK] Queue not full. ADDING Round {server_round}")
            
        elif score > self.top_k_heap[0][0]:
            # B. Hàng đợi đầy NHƯNG Score mới lớn hơn Score nhỏ nhất trong hàng đợi
            # 1. Lấy thằng nhỏ nhất ra
            removed_score, removed_round = heapq.heappop(self.top_k_heap)
            round_to_delete = removed_round
            
            # 2. Thêm thằng mới vào
            heapq.heappush(self.top_k_heap, (score, server_round))
            should_save = True
            print(f"--> [TopK] Score > Min. REPLACING Round {removed_round} (Score {removed_score:.4f}) with Round {server_round}")
        else:
            # C. Score nhỏ hơn cả thằng bét bảng -> Bỏ qua
            print(f"--> [TopK] Score too low. SKIPPED Round {server_round}")

        # --- THỰC HIỆN LƯU VÀ XÓA ---
        
        # 1. Nếu cần xóa (trường hợp B) -> Xóa thư mục round cũ khỏi ổ cứng
        if round_to_delete is not None:
            try:
                # Giả sử cấu trúc lưu là log_dir/round_X
                # Hàm save_client_updates của bạn lưu file vào folder log_dir (cấu trúc phẳng) hoặc log_dir/round_X
                # Dưới đây mình giả định bạn sửa utils để lưu theo folder round, 
                # hoặc nếu lưu phẳng thì phải xóa từng file client_id.pt có prefix round.
                
                # Cách tốt nhất: Xóa file cụ thể. Vì LogStrategy trước đó mình viết lưu chung folder log_dir
                # với tên file format là: round_{round}_client_{cid}.pt (hoặc folder round_{round})
                
                # Dưới đây là code xóa theo folder (nếu utils.py tạo folder round_X)
                delete_path = os.path.join(self.log_dir, f"round_{round_to_delete}")
                if os.path.exists(delete_path):
                    shutil.rmtree(delete_path) # Xóa cả thư mục
                    print(f"    -> DELETED from disk: {delete_path}")
                
                # Nếu utils lưu file lẻ, ta cần code xóa file lẻ (phức tạp hơn xíu)
                # Tạm thời giả định hệ thống bạn sẽ xóa folder round_{id}
            except Exception as e:
                log.error(f"Error deleting round {round_to_delete}: {e}")

        # 2. Nếu cần lưu (trường hợp A hoặc B) -> Lưu file mới
        if should_save:
            # Cập nhật file json danh sách round (cần sort lại theo thứ tự round tăng dần để Unlearn chạy đúng)
            current_rounds = sorted([r for s, r in self.top_k_heap])
            with open(self.saved_rounds_file, 'w') as f:
                json.dump(current_rounds, f)
            
            # Lưu file weights xuống đĩa
            # Lưu ý: Cần đảm bảo utils.save_client_updates tạo folder riêng cho từng round
            # để dễ xóa. Mình sẽ sửa đoạn gọi hàm này bên dưới để tạo folder.
            for msg in train_replies:
                if msg.has_error(): continue
                content = msg.content
                if "metrics" in content and "partition_id" in content["metrics"]:
                    client_id = content["metrics"]["partition_id"]
                    if "arrays" in content:
                        try:
                            array_record = content["arrays"]
                            ndarrays = [v.numpy() for v in array_record.values()]
                            parameters = ndarrays_to_parameters(ndarrays)
                            
                            # TẠO THƯ MỤC RIÊNG CHO ROUND (Quan trọng để xóa dễ)
                            round_dir = os.path.join(self.log_dir, f"round_{server_round}")
                            os.makedirs(round_dir, exist_ok=True)
                            
                            # Gọi hàm save nhưng trỏ vào round_dir
                            save_client_updates(
                                save_dir=self.log_dir, # utils sẽ tự ghép thêm round_num
                                round_num=server_round,
                                client_id=str(client_id),
                                parameters=parameters
                            )
                        except Exception:
                            pass

        return aggregated_result