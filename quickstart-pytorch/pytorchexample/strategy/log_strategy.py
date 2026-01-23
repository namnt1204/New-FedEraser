import logging
from flwr.serverapp.strategy import FedAvg
from flwr.common import ndarrays_to_parameters

from pytorchexample.utils import save_client_updates

log = logging.getLogger(__name__)

class LogStrategy(FedAvg):
    def __init__(self, log_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir

    def aggregate_train(self, server_round, train_replies):
        """
        Ghi đè hàm aggregate_train (Thay vì aggregate_fit).
        Hàm này nhận trực tiếp danh sách tin nhắn (train_replies) từ Client.
        """
        
        # --- LOGIC LƯU FILE ---
        log.info(f"--> [LogStrategy] Round {server_round}: Saving updates from {len(train_replies)} clients...")

        for msg in train_replies:
            if msg.has_error():
                continue
            
            # 1. Lấy ID và Dữ liệu
            client_id = msg.metadata.src_node_id
            content = msg.content
            
            # 2. Kiểm tra xem có chứa 'arrays' (trọng số) không
            if "arrays" in content:
                array_record = content["arrays"]
                
                # 3. Chuyển đổi ArrayRecord -> List[Numpy] -> Parameters
                try:
                    # Lấy các mảng numpy từ ArrayRecord
                    # Lưu ý: ArrayRecord lưu dưới dạng Dict, ta lấy values() 
                    # để có danh sách các layer weights.
                    ndarrays = [v.numpy() for v in array_record.values()]
                    
                    # Đóng gói thành Parameters
                    parameters = ndarrays_to_parameters(ndarrays)
                    
                    # 4. Lưu xuống đĩa
                    save_client_updates(
                        save_dir=self.log_dir,
                        round_num=server_round,
                        client_id=str(client_id),
                        parameters=parameters
                    )
                except Exception as e:
                    log.error(f"Failed to save update from client {client_id}: {e}")

        # --- GỌI LOGIC GỐC ---
        # Để FedAvg tính toán trung bình cộng như bình thường
        return super().aggregate_train(server_round, train_replies)