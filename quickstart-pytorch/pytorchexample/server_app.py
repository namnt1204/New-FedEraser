"""pytorchexample: A Flower / PyTorch app."""

import os
import torch
import json
import time
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.server.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, test

# Import Baseline
from pytorchexample.strategy.log_strategy import LogStrategy
from pytorchexample.strategy.eraser_strategy import EraserStrategy
from pytorchexample.strategy.retrain_strategy import RetrainStrategy

# Import Adaptive (Threshold)
from pytorchexample.strategy.adaptive_log_strategy import AdaptiveLogStrategy
from pytorchexample.strategy.adaptive_eraser_strategy import AdaptiveEraserStrategy

# Import Top-K (New)
from pytorchexample.strategy.topk_log_strategy import TopKLogStrategy
from pytorchexample.strategy.topk_eraser_strategy import TopKEraserStrategy

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    
    # Modes: train, unlearn, adaptive_train, adaptive_unlearn, topk_train, topk_unlearn
    mode: str = context.run_config.get("mode", "train")
    unlearn_cid: str = context.run_config.get("unlearn-cid", "0")

    print(f"STARTING SERVER IN MODE: {mode.upper()}")

    # --- ĐỊNH NGHĨA THƯ MỤC LOG RIÊNG BIỆT ---
    if "topk" in mode:
        log_dir = "topk_logs"       # Thư mục cho phương pháp Top-K
    elif "adaptive" in mode:
        log_dir = "adaptive_logs"   # Thư mục cho phương pháp Threshold
    else:
        log_dir = "history_logs"    # Thư mục cho phương pháp Gốc
    
    os.makedirs(log_dir, exist_ok=True)
    saved_rounds_list = []

    # Xử lý load history cho các mode Unlearn đặc biệt
    if mode == "adaptive_unlearn" or mode == "topk_unlearn":
        history_file = os.path.join(log_dir, "saved_rounds.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                saved_rounds_list = json.load(f)
            num_rounds = len(saved_rounds_list)
            print(f"--> [{mode}] Detected {num_rounds} saved rounds.")
        else:
            print(f"--> [{mode}] ERROR: No saved_rounds.json found in {log_dir}!")
            return 

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # --- CHỌN CHIẾN THUẬT ---
    if mode == "train":
        strategy = LogStrategy(log_dir=log_dir, fraction_evaluate=fraction_evaluate)
    elif mode == "retrain":
        strategy = RetrainStrategy(unlearn_cid=unlearn_cid, fraction_evaluate=fraction_evaluate)
    elif mode == "unlearn":
        strategy = EraserStrategy(log_dir=log_dir, unlearn_cid=unlearn_cid, fraction_evaluate=fraction_evaluate)
        
    elif mode == "adaptive_train":
        strategy = AdaptiveLogStrategy(
            log_dir=log_dir, threshold=0.01, total_rounds=num_rounds, decay_factor=0.95, fraction_evaluate=fraction_evaluate
        )
    elif mode == "adaptive_unlearn":
        strategy = AdaptiveEraserStrategy(
            log_dir=log_dir, unlearn_cid=unlearn_cid, saved_rounds_list=saved_rounds_list, fraction_evaluate=fraction_evaluate
        )
        
    # --- TOP-K STRATEGIES ---
    elif mode == "topk_train":
        strategy = TopKLogStrategy(
            log_dir=log_dir,
            k_value=30,             # <--- Giữ lại đúng 20 round quan trọng nhất
            total_rounds=num_rounds,
            decay_factor=0.95,       # Vẫn dùng decay để ưu tiên round cuối
            fraction_evaluate=fraction_evaluate
        )
    elif mode == "topk_unlearn":
        strategy = TopKEraserStrategy(
            log_dir=log_dir,
            unlearn_cid=unlearn_cid,
            saved_rounds_list=saved_rounds_list,
            fraction_evaluate=fraction_evaluate
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    client_mode = "unlearn" if "unlearn" in mode else "train"

    print(f"⏱️  STARTED TIMING FOR MODE: {mode}...")
    start_time = time.time()

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({
            "lr": lr, "mode": client_mode, "local_epochs": context.run_config["local-epochs"]
        }),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n🚀 EXECUTION TIME ({mode.upper()}): {execution_time:.2f} seconds\n")

    save_name = f"final_model_{mode}.pt"
    print(f"\nSaving final model to {save_name}...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, save_name)

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})