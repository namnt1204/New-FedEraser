import torch
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from pytorchexample.task import Net, load_centralized_dataset, load_data

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TARGET_CLIENT_ID = 1
NUM_PARTITIONS = 20

# Đã thêm mô hình Retrain (Baseline) để chuẩn form bài báo
MODELS_TO_COMPARE = {
    "Original": "final_model_train.pt",
    "FedRetrain": "final_model_retrain.pt",
    "FedEraser": "final_model_unlearn.pt",
    "Adaptive": "final_model_adaptive_unlearn.pt",
    "Top-K": "final_model_topk_unlearn.pt"
}

# Khai báo Dictionary để lưu kết quả xuất ra file
results_utility = {}
results_forgetting = {}
results_deviation = {}

# ==========================================
# 1. CÁC HÀM HỖ TRỢ (HELPER FUNCTIONS)
# ==========================================
def get_model(path):
    if not os.path.exists(path):
        return None
    model = Net()
    try:
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None

def extract_batch(batch):
    """Trích xuất images và labels từ nhiều định dạng batch khác nhau."""
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]
    elif isinstance(batch, dict):
        images = batch.get("image", batch.get("pixel_values", list(batch.values())[0]))
        labels = batch.get("label", batch.get("labels", list(batch.values())[1]))
        return images, labels
    return batch

# ==========================================
# 2. CÁC HÀM ĐÁNH GIÁ (EVALUATION METRICS)
# ==========================================
def evaluate_utility(model, dataloader):
    """Tính Accuracy và Loss tiêu chuẩn."""
    criterion = CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels = extract_batch(batch)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / total, correct / total

def calculate_mia_metrics(model, member_loader, non_member_loader):
    """Tính MIA Precision và Recall."""
    criterion = CrossEntropyLoss(reduction="none")
    
    def get_losses(loader):
        losses_arr = []
        with torch.no_grad():
            for batch in loader:
                images, labels = extract_batch(batch)
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                losses = criterion(outputs, labels)
                losses_arr.extend(losses.cpu().numpy())
        return np.array(losses_arr)

    member_losses = get_losses(member_loader)
    non_member_losses = get_losses(non_member_loader)
    
    threshold = np.mean(non_member_losses) 
    
    y_pred_member = member_losses < threshold
    y_pred_non_member = non_member_losses < threshold
    
    true_positive = np.sum(y_pred_member)
    false_positive = np.sum(y_pred_non_member)
    false_negative = len(member_losses) - true_positive
    
    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)
    
    return float(precision), float(recall)

def calculate_p_diff(orig_model, unlearned_model, target_loader):
    """Tính toán Prediction Difference (P_diff) theo công thức bài báo."""
    if orig_model is None or unlearned_model is None:
        return 0.0
        
    p_diff_sum = 0.0
    total = 0
    with torch.no_grad():
        for batch in target_loader:
            images, _ = extract_batch(batch)
            images = images.to(DEVICE)
            
            out_orig = F.softmax(orig_model(images), dim=1)
            out_unlearned = F.softmax(unlearned_model(images), dim=1)
            
            diff = out_orig - out_unlearned
            l2_norm = torch.norm(diff, p=2, dim=1)
            p_diff_sum += l2_norm.sum().item()
            total += images.size(0)
            
    return float(p_diff_sum / total)

def calculate_parameter_deviation(baseline_model, unlearned_model):
    """Tính góc lệch theta (độ) của layer cuối cùng."""
    if baseline_model is None or unlearned_model is None:
        return 0.0
        
    w_base = list(baseline_model.modules())[-1].weight.data.flatten()
    w_unlearn = list(unlearned_model.modules())[-1].weight.data.flatten()
    
    cos_sim = F.cosine_similarity(w_base.unsqueeze(0), w_unlearn.unsqueeze(0)).item()
    cos_sim = max(min(cos_sim, 1.0), -1.0)
    angle_deg = np.degrees(np.arccos(cos_sim))
    
    return float(angle_deg)

def generate_charts(eval_utility, eval_forgetting, eval_param_deviation, unlearn_times):
    methods = [m for m in MODELS_TO_COMPARE.keys() if m in eval_utility]
    methods_unlearn = [m for m in methods if m != "Original"]

    if not methods:
        return

    plt.rcParams.update({'font.size': 12})
    x = np.arange(len(methods))
    width = 0.35

    test_acc = [eval_utility[m].get("test_accuracy", 0) for m in methods]
    target_acc = [eval_forgetting.get(m, {}).get("target_accuracy", 0) for m in methods]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width/2, test_acc, width, label='Testing Data (Utility)', color='#1f77b4', edgecolor='black', hatch='//')
    ax.bar(x + width/2, target_acc, width, label='Target Data (Forgetting)', color='#d62728', edgecolor='black', hatch='\\\\')

    ax.set_ylabel('Prediction Accuracy')
    ax.set_title('Fig 1: Prediction Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='lower left')
    
    min_val = min([val for val in test_acc + target_acc if val > 0] + [0.8])
    ax.set_ylim(max(0, min_val - 0.1), 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("Fig1_Prediction_Accuracy.png", bbox_inches='tight')
    plt.close()

    mia_prec = [eval_forgetting.get(m, {}).get("mia_precision", 0) for m in methods]
    mia_rec = [eval_forgetting.get(m, {}).get("mia_recall", 0) for m in methods]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width/2, mia_prec, width, label='Attack Precision', color='#2ca02c', edgecolor='black', hatch='xx')
    ax.bar(x + width/2, mia_rec, width, label='Attack Recall', color='#ff7f0e', edgecolor='black', hatch='++')

    ax.set_ylabel('MIA Metric Score')
    ax.set_title('Fig 2: Performance of Membership Inference Attacks (MIA)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='center right')
    ax.set_ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("Fig2_MIA_Performance.png", bbox_inches='tight')
    plt.close()

    if methods_unlearn:
        theta = [eval_param_deviation.get(m, {}).get("theta_degrees", 0) for m in methods_unlearn]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.bar(methods_unlearn, theta, color='#9467bd', edgecolor='black', width=0.5, hatch='..')
        ax.set_ylabel('Angle Deviation (\u03b8 degrees)')
        ax.set_title('Fig 3: Parameter Deviation from Retrained Model')
        for i, v in enumerate(theta):
            ax.text(i, v + 2, f"{v:.1f}°", ha='center', fontweight='bold')
        ax.set_ylim(0, max(theta + [100]) * 1.15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig("Fig3_Parameter_Deviation.png", bbox_inches='tight')
        plt.close()

    if methods_unlearn:
        times = [unlearn_times.get(m, 0.0) for m in methods_unlearn]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.bar(methods_unlearn, times, color='#8c564b', edgecolor='black', width=0.5, hatch='*')
        ax.set_ylabel('Time Consumption (Seconds)')
        ax.set_title('Fig 4: Unlearning Time Consumption (End-to-End)')
        max_time = max(times + [10])
        for i, v in enumerate(times):
            ax.text(i, v + (max_time * 0.02), f"{v:.1f}s", ha='center', fontweight='bold')
        ax.set_ylim(0, max_time * 1.15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig("Fig4_Time_Consumption.png", bbox_inches='tight')
        plt.close()


# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH (MAIN PROCESS)
# ==========================================
def main():
    print(f"🚀 STARTING COMPARISON EVALUATION on {DEVICE}")
    
    time_file = "unlearn_times.json"
    execution_times = {}
    if os.path.exists(time_file):
        try:
            with open(time_file, "r") as f:
                execution_times = json.load(f)
        except Exception as e:
            pass

    # --- Load Data ---
    raw_dataset = load_centralized_dataset()
    if isinstance(raw_dataset, dict):
        global_testloader = raw_dataset.get("test", raw_dataset.get("val", list(raw_dataset.values())[0]))
    elif isinstance(raw_dataset, (list, tuple)):
        global_testloader = raw_dataset[1] if len(raw_dataset) > 1 else raw_dataset[0]
    else:
        global_testloader = raw_dataset

    # Lấy dữ liệu của Client Target (Tập Train của client đó chính là Target Data để test quên)
    res = load_data(partition_id=TARGET_CLIENT_ID, num_partitions=NUM_PARTITIONS, batch_size=32)
    forget_loader = res[0] if isinstance(res, (list, tuple)) else res

    # --- Load Original Model làm tham chiếu ---
    orig_model = get_model(MODELS_TO_COMPARE.get("Original", ""))
    
    # Nếu bạn có model FedRetrain, hãy dùng nó làm baseline cho Parameter Deviation. 
    # Nếu không, tạm dùng Original model.
    baseline_model = get_model(MODELS_TO_COMPARE.get("FedRetrain", ""))
    if baseline_model is None:
        baseline_model = orig_model

    for name, path in MODELS_TO_COMPARE.items():
        print(f"\n🔍 Evaluating: {name}...")
        model = get_model(path)
        if model is None:
            continue

        # 1. Tính Utility
        u_loss, u_acc = evaluate_utility(model, global_testloader)
        
        # 2. Tính Forgetting
        f_loss, f_acc = evaluate_utility(model, forget_loader)
        mia_precision, mia_recall = calculate_mia_metrics(model, forget_loader, global_testloader)
        p_diff = calculate_p_diff(orig_model, model, forget_loader)
        
        # 3. Tính Parameter Deviation
        theta = calculate_parameter_deviation(baseline_model, model) if name != "Original" else 0.0

        print(f"   ► Global Utility: Acc = {u_acc*100:.2f}%, Loss = {u_loss:.4f}")
        print(f"   ► Target Data:    Acc = {f_acc*100:.2f}%, Loss = {f_loss:.4f}, P_diff = {p_diff:.4f}")
        print(f"   ► MIA Attack:     Prec = {mia_precision:.4f}, Recall = {mia_recall:.4f}")
        if name != "Original":
            print(f"   ► Param Deviation: Theta = {theta:.2f}°")

        # Lưu vào dict để xuất file
        results_utility[name] = {
            "test_accuracy": u_acc,
            "test_loss": u_loss,
            "execution_time_seconds": execution_times.get(name, 0.0) # TODO: Bạn nhập thủ công thời gian từ log của server_app.py vào file JSON sau nhé
        }
        
        results_forgetting[name] = {
            "target_accuracy": f_acc,
            "target_loss": f_loss,
            "p_diff": p_diff,
            "mia_precision": mia_precision,
            "mia_recall": mia_recall
        }
        
        if name != "Original":
            results_deviation[name] = {
                "theta_degrees": theta
            }

    # --- Lưu dữ liệu ra files JSON ---
    with open("eval_utility.json", "w") as f:
        json.dump(results_utility, f, indent=4)
    with open("eval_forgetting.json", "w") as f:
        json.dump(results_forgetting, f, indent=4)
    with open("eval_param_deviation.json", "w") as f:
        json.dump(results_deviation, f, indent=4)
        
    generate_charts(results_utility, results_forgetting, results_deviation, execution_times)

if __name__ == "__main__":
    main()