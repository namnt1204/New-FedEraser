import logging
import torch
import time
import json
import os
from flwr.serverapp.strategy import FedAvg
from flwr.common import ArrayRecord

from pytorchexample.utils import load_client_updates, ndarrays_to_vector, vector_to_ndarrays

log = logging.getLogger(__name__)

class TopKEraserStrategy(FedAvg):
    def __init__(self, log_dir: str, unlearn_cid: str, saved_rounds_list: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir
        self.unlearn_cid = str(unlearn_cid)
        self.saved_rounds_list = saved_rounds_list # List round id đã sort [2, 5, 49, 50...]
        self.global_start_time = time.time()

    def aggregate_train(self, server_round, train_replies):
        # Mapping: Unlearn Round 1 -> Saved Round[0] (ví dụ Round 2)
        if server_round - 1 < len(self.saved_rounds_list):
            actual_history_round = self.saved_rounds_list[server_round - 1]
            print(f"--> [TopK Unlearn] Step {server_round} maps to History Round {actual_history_round}")
        else:
            actual_history_round = server_round 

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
                continue

            if "arrays" in msg.content:
                try:
                    # 1. Update mới
                    array_record = msg.content["arrays"]
                    new_ndarrays = [v.numpy() for v in array_record.values()]
                    new_vec = ndarrays_to_vector(new_ndarrays)
                    
                    # 2. LOAD UPDATE CŨ
                    # Lưu ý: Load từ đúng thư mục TopK
                    old_ndarrays = load_client_updates(self.log_dir, actual_history_round, client_id)
                    old_vec = ndarrays_to_vector(old_ndarrays)
                    
                    # 3. Calibration
                    norm_old = torch.norm(old_vec)
                    norm_new = torch.norm(new_vec)
                    
                    if norm_new == 0:
                        calibrated_vec = new_vec
                    else:
                        calibrated_vec = norm_old * (new_vec / norm_new)
                    
                    # 4. Save back
                    calibrated_ndarrays = vector_to_ndarrays(calibrated_vec, new_ndarrays)
                    updated_dict = {}
                    keys = list(array_record.keys())
                    for i, key in enumerate(keys):
                        updated_dict[key] = torch.from_numpy(calibrated_ndarrays[i])
                    
                    msg.content["arrays"] = ArrayRecord(updated_dict)
                    
                except FileNotFoundError:
                     log.warning(f"⚠️ Missing history for Client {client_id} (Round {actual_history_round}).")
                except Exception as e:
                     log.error(f"Error calibrating client {client_id}: {e}")

            calibrated_replies.append(msg)

        aggregated_result = super().aggregate_train(server_round, calibrated_replies)
        
        total_e2e_time = time.time() - self.global_start_time
        log.info(f"🕒 [TopK] Total E2E Time up to round {server_round}: {total_e2e_time:.4f}s")

        time_file = "unlearn_times.json"
        try:
            times = {}
            if os.path.exists(time_file):
                with open(time_file, "r") as f:
                    times = json.load(f)
            times["Top-K"] = total_e2e_time
            with open(time_file, "w") as f:
                json.dump(times, f, indent=4)
        except Exception as e:
            log.error(f"Error: {e}")

        return aggregated_result