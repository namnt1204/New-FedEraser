import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import CrossEntropyLoss
from pytorchexample.task import Net, load_centralized_dataset, load_data

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TARGET_CLIENT_ID = 1
NUM_PARTITIONS = 20

MODELS_TO_COMPARE = {
    "Original (Full Train)": "final_model_train.pt",
    "FedEraser (Base)":      "final_model_unlearn.pt",
    "Adaptive (Threshold)":  "final_model_adaptive_unlearn.pt",
    "Top-K (Priority)":      "final_model_topk_unlearn.pt"
}

def get_model(path):
    if not os.path.exists(path):
        print(f"⚠️ Warning: File {path} not found. Skipping.")
        return None
    
    model = Net()
    try:
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None

def get_losses(model, dataloader):
    criterion = CrossEntropyLoss(reduction="none")
    all_losses = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            elif isinstance(batch, dict):
                if "image" in batch: images = batch["image"]
                elif "pixel_values" in batch: images = batch["pixel_values"]
                else: images = list(batch.values())[0]

                if "label" in batch: labels = batch["label"]
                elif "labels" in batch: labels = batch["labels"]
                else: labels = list(batch.values())[1]
            else:
                images, labels = batch

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            losses = criterion(outputs, labels)
            all_losses.extend(losses.cpu().numpy())
            
    return np.array(all_losses)

def calculate_mia_metrics(model, member_loader, non_member_loader):
    member_losses = get_losses(model, member_loader)
    non_member_losses = get_losses(model, non_member_loader)
    
    threshold = np.mean(non_member_losses) 
    
    y_pred_member = member_losses < threshold
    y_pred_non_member = non_member_losses < threshold
    
    true_positive = np.sum(y_pred_member)
    false_positive = np.sum(y_pred_non_member)
    false_negative = len(member_losses) - true_positive
    
    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)
    
    return precision, recall

def evaluate(model, dataloader, description):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            elif isinstance(batch, dict):
                if "image" in batch: images = batch["image"]
                elif "pixel_values" in batch: images = batch["pixel_values"]
                else: images = list(batch.values())[0]

                if "label" in batch: labels = batch["label"]
                elif "labels" in batch: labels = batch["labels"]
                else: labels = list(batch.values())[1]
            else:
                images, labels = batch

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    print(f"🚀 STARTING COMPARISON EVALUATION on {DEVICE}")
    print("="*60)

    print("📥 Loading Datasets...")
    
    raw_dataset = load_centralized_dataset()
    
    if isinstance(raw_dataset, dict):
        print("   -> Detected Dictionary dataset. Extracting 'test'...")
        if "test" in raw_dataset:
            global_testloader = raw_dataset["test"]
        elif "val" in raw_dataset:
            global_testloader = raw_dataset["val"]
        else:
            global_testloader = list(raw_dataset.values())[0]
            
    elif isinstance(raw_dataset, (list, tuple)):
        print("   -> Detected Tuple dataset. Taking the second element (assuming test)...")
        global_testloader = raw_dataset[1] if len(raw_dataset) > 1 else raw_dataset[0]
    else:
        global_testloader = raw_dataset

    res = load_data(partition_id=TARGET_CLIENT_ID, num_partitions=NUM_PARTITIONS, batch_size=32)
    if isinstance(res, (list, tuple)):
        forget_loader = res[0]
    else:
        forget_loader = res

    results = []

    for name, path in MODELS_TO_COMPARE.items():
        print(f"\n🔍 Evaluating: {name}...")
        model = get_model(path)
        
        if model is None:
            results.append((name, 0, 0, 0, 0, 0, 0, False)) 
            continue

        u_loss, u_acc = evaluate(model, global_testloader, "Utility")
        
        f_loss, f_acc = evaluate(model, forget_loader, "Forget")
        
        print("   ... Calculating MIA Metrics ...")
        mia_precision, mia_recall = calculate_mia_metrics(model, forget_loader, global_testloader)

        print(f"   ► Global Utility: Acc = {u_acc*100:.2f}%, Loss = {u_loss:.4f}")
        print(f"   ► Forget Check:   Acc = {f_acc*100:.2f}%, Loss = {f_loss:.4f}")
        print(f"   ► MIA Attack:     Prec = {mia_precision:.4f}, Recall = {mia_recall:.4f}")
        
        results.append((name, u_acc, u_loss, f_acc, f_loss, mia_precision, mia_recall, True))

    print("\n" + "="*110)
    print(f"{'METHOD':<25} | {'GLOBAL ACC':<12} | {'TARGET LOSS':<15} | {'MIA PRECISION':<15} | {'MIA RECALL':<15}")
    print("-" * 110)
    
    orig_acc = 0
    orig_forget_loss = 0
    orig_mia_prec = 0
    
    for r in results:
        if "Original" in r[0] and r[7]:
            orig_acc = r[1]
            orig_forget_loss = r[4]
            orig_mia_prec = r[5]
            break

    for name, u_acc, u_loss, f_acc, f_loss, m_prec, m_rec, exists in results:
        if not exists:
            print(f"{name:<25} | {'N/A':<12} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")
            continue
            
        utility_str = f"{u_acc*100:.2f}%"
        forget_str = f"{f_loss:.4f}"
        mia_str = f"{m_prec:.4f}"
        
        if "Original" not in name:
            diff_acc = u_acc - orig_acc
            utility_str += f" ({diff_acc*100:+.1f}%)"
            
            diff_loss = f_loss - orig_forget_loss
            if diff_loss > 0:
                forget_str += f" (+{diff_loss:.2f}✅)" 
            else:
                forget_str += f" ({diff_loss:.2f})"
                
            diff_mia = m_prec - orig_mia_prec
            if m_prec < orig_mia_prec:
                mia_str += f" (-{abs(diff_mia):.2f}✅)"
            else:
                mia_str += f" (+{diff_mia:.2f})"

        print(f"{name:<25} | {utility_str:<12} | {forget_str:<15} | {mia_str:<15} | {m_rec:.4f}")
    
    print("="*110)
    print("💡 NOTE:")
    print("1. TARGET LOSS:  Càng CAO càng tốt (so với Original).")
    print("2. MIA METRICS:  Precision & Recall càng THẤP càng tốt (An toàn).")
    print("="*110)

if __name__ == "__main__":
    main()