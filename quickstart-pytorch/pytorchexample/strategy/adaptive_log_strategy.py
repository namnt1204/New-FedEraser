import logging
import json
import os
import torch
import numpy as np
from flwr.serverapp.strategy import FedAvg
from flwr.common import ndarrays_to_parameters

from pytorchexample.utils import save_client_updates, ndarrays_to_vector

log = logging.getLogger(__name__)

class AdaptiveLogStrategy(FedAvg):
    def __init__(self, log_dir: str, threshold: float, total_rounds: int, decay_factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir
        self.threshold = threshold
        self.total_rounds = total_rounds
        self.decay_factor = decay_factor
        
        self.saved_rounds_file = os.path.join(log_dir, "saved_rounds.json")
        self.saved_rounds = []
        
        self.current_weights = None

    def aggregate_train(self, server_round, train_replies):
        aggregated_result = super().aggregate_train(server_round, train_replies)
        
        aggregated_arrays = aggregated_result[0]
        if aggregated_arrays is None:
            return aggregated_result

        # 1. Lấy list các mảng numpy
        new_ndarrays = [v.numpy() for v in aggregated_arrays.values()]
        
        # Truyền trực tiếp list numpy vào
        new_vec = ndarrays_to_vector(new_ndarrays) 

        # --- TÍNH SCORE ---
        score = 0.0
        norm = 0.0
        
        if self.current_weights is not None:
            delta_w = new_vec - self.current_weights
            norm = torch.norm(delta_w).item()
            
            # Áp dụng công thức: Score = lambda^(T-t) * ||Delta W||
            # decay_factor là lambda
            # total_rounds là T = global round = 50
            # server_round là t = round hiện tại
            exponent = max(0, self.total_rounds - server_round)
            weight = self.decay_factor ** exponent
            
            score = weight * norm
        else:
            score = float('inf')

        self.current_weights = new_vec

        print(f"ROUND {server_round} | Norm: {norm:.4f} | Weight: {self.decay_factor}^{self.total_rounds - server_round} | SCORE: {score:.4f} (Threshold: {self.threshold})")

        # --- QUYẾT ĐỊNH LƯU HAY BỎ ---
        if score > self.threshold:
            print(f"--> [Adaptive] Round {server_round} SAVED")
            
            self.saved_rounds.append(server_round)
            with open(self.saved_rounds_file, 'w') as f:
                json.dump(self.saved_rounds, f)

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
                            
                            save_client_updates(
                                save_dir=self.log_dir,
                                round_num=server_round,
                                client_id=str(client_id),
                                parameters=parameters
                            )
                        except Exception as e:
                            log.error(f"Failed to save partition {client_id}: {e}")
        else:
            print(f"--> [Adaptive] Round {server_round} SKIPPED")

        return aggregated_result