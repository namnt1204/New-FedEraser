import logging
import torch
import os
import time
import json
from flwr.serverapp.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import ArrayRecord

from pytorchexample.utils import load_client_updates, ndarrays_to_vector, vector_to_ndarrays

log = logging.getLogger(__name__)

class EraserStrategy(FedAvg):
    def __init__(self, log_dir: str, unlearn_cid: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir
        self.unlearn_cid = str(unlearn_cid)
        self.global_start_time = time.time()

    def aggregate_train(self, server_round, train_replies):
        """
        Ghi đè aggregate_train để thực hiện FedEraser Calibration.
        """
        log.info(f"--> [FedEraser] Round {server_round}: Calibrating updates...")
        
        calibrated_replies = []

        for msg in train_replies:
            if msg.has_error():
                calibrated_replies.append(msg)
                continue
            
            if "metrics" not in msg.content or "partition_id" not in msg.content["metrics"]:
                calibrated_replies.append(msg)
                continue

            client_id = str(msg.content["metrics"]["partition_id"])
            
            if client_id == str(self.unlearn_cid):
                log.info(f"--> [FedEraser] Skipping Target Client (Partition {client_id})")
                continue

            if "arrays" in msg.content:
                try:
                    # --- BƯỚC 1: LẤY UPDATE MỚI ---
                    array_record = msg.content["arrays"]
                    # Chuyển sang List[Numpy] -> Tensor Vector
                    new_ndarrays = [v.numpy() for v in array_record.values()]
                    new_vec = ndarrays_to_vector(new_ndarrays)
                    
                    # --- BƯỚC 2: LOAD UPDATE CŨ ---
                    # Load từ file log lịch sử
                    old_ndarrays = load_client_updates(self.log_dir, server_round, client_id)
                    old_vec = ndarrays_to_vector(old_ndarrays)
                    
                    # --- BƯỚC 3: TÍNH TOÁN HIỆU CHỈNH (Calibration) ---
                    norm_old = torch.norm(old_vec)
                    norm_new = torch.norm(new_vec)
                    
                    if norm_new == 0:
                        calibrated_vec = new_vec
                    else:
                        # Công thức: Direction(New) * Magnitude(Old)
                        calibrated_vec = norm_old * (new_vec / norm_new)
                    
                    # --- BƯỚC 4: GHI ĐÈ LẠI VÀO MESSAGE ---
                    # Chuyển ngược lại thành List[Numpy]
                    calibrated_ndarrays = vector_to_ndarrays(calibrated_vec, new_ndarrays)
                    
                    # Cập nhật lại nội dung ArrayRecord trong tin nhắn
                    updated_dict = {}
                    keys = list(array_record.keys()) # Lấy danh sách tên layer
                    
                    for i, key in enumerate(keys):
                        # SỬA LỖI Ở ĐÂY: Chuyển Numpy -> Tensor
                        # ArrayRecord chấp nhận dict nếu value là Tensor (giống state_dict)
                        updated_dict[key] = torch.from_numpy(calibrated_ndarrays[i])
                    
                    
                    msg.content["arrays"] = ArrayRecord(updated_dict)
                    
                except FileNotFoundError:
                     log.warning(f"⚠️ Missing history for Client {client_id}. Using raw update.")
                except Exception as e:
                     log.error(f"Error calibrating client {client_id}: {e}")

            calibrated_replies.append(msg)

        aggregated_result = super().aggregate_train(server_round, calibrated_replies)
        
        total_e2e_time = time.time() - self.global_start_time
        log.info(f"🕒 Total E2E Time up to round {server_round}: {total_e2e_time:.4f}s")

        time_file = "unlearn_times.json"
        try:
            times = {}
            if os.path.exists(time_file):
                with open(time_file, "r") as f:
                    times = json.load(f)
            times["FedEraser"] = total_e2e_time
            with open(time_file, "w") as f:
                json.dump(times, f, indent=4)
        except Exception as e:
            log.error(f"Error: {e}")

        return aggregated_result