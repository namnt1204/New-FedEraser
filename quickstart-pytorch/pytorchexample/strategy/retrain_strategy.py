import logging
import time
import json
import os
from flwr.serverapp.strategy import FedAvg

log = logging.getLogger(__name__)

class RetrainStrategy(FedAvg):
    def __init__(self, unlearn_cid: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unlearn_cid = str(unlearn_cid)
        self.global_start_time = time.time()

    def aggregate_train(self, server_round, train_replies):
        """
        Ghi đè aggregate_train để thực hiện FedRetrain: 
        Loại bỏ hoàn toàn đóng góp của client bị xóa, sau đó mới tính trung bình (FedAvg).
        """
        log.info(f"--> [FedRetrain] Round {server_round}: Filtering out target client...")
        
        filtered_replies = []

        for msg in train_replies:
            if msg.has_error():
                filtered_replies.append(msg)
                continue
            
            if "metrics" in msg.content and "partition_id" in msg.content["metrics"]:
                client_id = str(msg.content["metrics"]["partition_id"])
                
                # NẾU LÀ CLIENT MỤC TIÊU -> BỎ QUA HOÀN TOÀN KHÔNG CHO VÀO LIST
                if client_id == self.unlearn_cid:
                    log.info(f"    -> 🚫 [FedRetrain] Dropped update from Target Client (Partition {client_id})")
                    continue
            
            # Nếu là client bình thường -> Thêm vào danh sách để đem đi Average
            filtered_replies.append(msg)

        # --- GỌI HÀM AGGREGATE GỐC CỦA FEDAVG VÀ CHỐT THỜI GIAN ---
        # Hàm gốc sẽ tự động lấy trung bình trọng số của các client CÒN LẠI
        aggregated_result = super().aggregate_train(server_round, filtered_replies)
        
        total_e2e_time = time.time() - self.global_start_time
        log.info(f"🕒 [FedRetrain] Total E2E Time up to round {server_round}: {total_e2e_time:.4f}s")

        time_file = "unlearn_times.json"
        try:
            times = {}
            if os.path.exists(time_file):
                with open(time_file, "r") as f:
                    times = json.load(f)
            times["FedRetrain"] = total_e2e_time
            with open(time_file, "w") as f:
                json.dump(times, f, indent=4)
        except Exception as e:
            log.error(f"Error: {e}")

        return aggregated_result