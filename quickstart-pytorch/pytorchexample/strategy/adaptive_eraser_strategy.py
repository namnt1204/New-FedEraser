import logging
import torch
from flwr.serverapp.strategy import FedAvg
from flwr.common import ArrayRecord

from pytorchexample.utils import load_client_updates, ndarrays_to_vector, vector_to_ndarrays

log = logging.getLogger(__name__)

class AdaptiveEraserStrategy(FedAvg):
    def __init__(self, log_dir: str, unlearn_cid: str, saved_rounds_list: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir
        self.unlearn_cid = str(unlearn_cid)
        self.saved_rounds_list = saved_rounds_list # Danh sách [1, 5, 8...]

    def aggregate_train(self, server_round, train_replies):
        # --- MAPPING ROUND ---
        # Server chạy từ 1, 2, 3... ta cần ánh xạ sang round lịch sử thực tế
        if server_round - 1 < len(self.saved_rounds_list):
            actual_history_round = self.saved_rounds_list[server_round - 1]
            log.info(f"--> [Adaptive] Unlearn Step {server_round} maps to History Round {actual_history_round}")
        else:
            # Trường hợp phòng hờ, thường không xảy ra nếu config đúng num_rounds
            actual_history_round = server_round 

        calibrated_replies = []

        for msg in train_replies:
            if msg.has_error():
                calibrated_replies.append(msg)
                continue
            
            # Kiểm tra Partition ID
            if "metrics" not in msg.content or "partition_id" not in msg.content["metrics"]:
                calibrated_replies.append(msg)
                continue

            client_id = str(msg.content["metrics"]["partition_id"])
            
            # Bỏ qua client cần xóa
            if client_id == str(self.unlearn_cid):
                log.info(f"--> [Adaptive] Skipping Target Client {client_id}")
                continue

            if "arrays" in msg.content:
                try:
                    # 1. Update mới
                    array_record = msg.content["arrays"]
                    new_ndarrays = [v.numpy() for v in array_record.values()]
                    new_vec = ndarrays_to_vector(new_ndarrays)
                    
                    # 2. LOAD UPDATE CŨ (Dùng actual_history_round)
                    old_ndarrays = load_client_updates(self.log_dir, actual_history_round, client_id)
                    old_vec = ndarrays_to_vector(old_ndarrays)
                    
                    # 3. Calibration (FedEraser Math)
                    norm_old = torch.norm(old_vec)
                    norm_new = torch.norm(new_vec)
                    
                    if norm_new == 0:
                        calibrated_vec = new_vec
                    else:
                        calibrated_vec = norm_old * (new_vec / norm_new)
                    
                    # 4. Save back (Fix lỗi Tensor/Numpy cho ArrayRecord)
                    calibrated_ndarrays = vector_to_ndarrays(calibrated_vec, new_ndarrays)
                    updated_dict = {}
                    keys = list(array_record.keys())
                    for i, key in enumerate(keys):
                        updated_dict[key] = torch.from_numpy(calibrated_ndarrays[i])
                    
                    msg.content["arrays"] = ArrayRecord(updated_dict)
                    
                except FileNotFoundError:
                     log.warning(f"⚠️ Missing history for Client {client_id} at round {actual_history_round}. Using raw update.")
                except Exception as e:
                     log.error(f"Error calibrating client {client_id}: {e}")

            calibrated_replies.append(msg)

        return super().aggregate_train(server_round, calibrated_replies)