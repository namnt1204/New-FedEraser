"""pytorchexample: A Flower / PyTorch app."""

import os
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.server.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, test

# Import 2 chiến thuật tùy chỉnh chúng ta vừa viết
from pytorchexample.strategy.log_strategy import LogStrategy
from pytorchexample.strategy.eraser_strategy import EraserStrategy

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # 1. Đọc cấu hình từ pyproject.toml
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    
    # Đọc cấu hình chế độ chạy (Mặc định là 'train' nếu không tìm thấy)
    mode: str = context.run_config.get("mode", "train")
    unlearn_cid: str = context.run_config.get("unlearn-cid", "0")

    print(f"STARTING SERVER IN MODE: {mode.upper()}")
    if mode == "unlearn":
        print(f"TARGET CLIENT TO REMOVE: {unlearn_cid}")

    # 2. Định nghĩa thư mục lưu log lịch sử
    # Lưu tại thư mục gốc dự án/history_logs
    log_dir = "history_logs"
    os.makedirs(log_dir, exist_ok=True)

    # 3. Load model khởi tạo (Global Model)
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # 4. CHỌN STRATEGY) DỰA TRÊN MODE
    if mode == "train":
        # Chế độ Train: Sử dụng LogStrategy để vừa train vừa lưu file updates
        strategy = LogStrategy(
            log_dir=log_dir,
            fraction_evaluate=fraction_evaluate
        )
    elif mode == "unlearn":
        # Chế độ Unlearn: Sử dụng EraserStrategy để hiệu chỉnh vector
        strategy = EraserStrategy(
            log_dir=log_dir,
            unlearn_cid=unlearn_cid,
            fraction_evaluate=fraction_evaluate
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'unlearn'.")

    # 5. Bắt đầu chạy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    save_name = f"final_model_{mode}.pt"
    print(f"\nSaving final model to {save_name}...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, save_name)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})