import os
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ===== CONFIG =====
csv_path = "test_labels.csv"
image_root = "dataset_test"
normalize_mean_std = False  # True nếu dùng chuẩn hóa ImageNet

# ===== Load CSV =====
df = pd.read_csv(csv_path)
df['style'] = df['style'].astype(str)

# Mapping label <-> id
label_set = sorted(df['style'].unique())
label2id = {name: i for i, name in enumerate(label_set)}
id2label = {i: name for name, i in label2id.items()}
df['label_id'] = df['style'].map(label2id)

# ===== Load ONNX model =====
session = ort.InferenceSession("custom_cnn.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g., [1, 3, 384, 384]

# Xác định kích thước ảnh model yêu cầu
_, c, h, w = [dim if isinstance(dim, int) else 1 for dim in input_shape]
img_size = (w, h)
print(f"✅ Model expects input shape: {input_shape} -> Resize to: {img_size}")

# ===== Load ảnh =====
X_test = []
valid_indices = []
for idx, row in df.iterrows():
    # Tìm file ở tất cả subfolder
    found = False
    for root, dirs, files in os.walk(image_root):
        if row['filename'] in files:
            img_path = os.path.join(root, row['filename'])
            found = True
            break
    if not found:
        print(f"[❌] Không tìm thấy ảnh: {row['filename']}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[⚠️] Không đọc được ảnh: {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    if normalize_mean_std:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

    X_test.append(img.transpose(2, 0, 1))  # CHW
    valid_indices.append(idx)

X_test = np.array(X_test, dtype=np.float32)
print(f"✅ Loaded {len(X_test)} images with shape: {X_test.shape}")

# ===== Inference =====
# Xử lý theo batch nhỏ hoặc từng ảnh nếu model yêu cầu batch size = 1
try:
    y_probs = session.run([output_name], {input_name: X_test})[0]
except Exception as e:
    print(f"⚠️ Batch inference failed: {e}")
    print("👉 Chuyển sang inference từng ảnh một...")
    y_probs = []
    for img in X_test:
        inp = np.expand_dims(img, axis=0)  # [1, C, H, W]
        out = session.run([output_name], {input_name: inp})[0]
        y_probs.append(out[0])
    y_probs = np.array(y_probs)

# Dự đoán nhãn
if y_probs.shape[1] == 1:
    y_scores = y_probs.squeeze()
    y_pred_ids = (y_scores > 0.5).astype(int)
else:
    y_pred_ids = np.argmax(y_probs, axis=1)
    y_scores = y_probs[:, 1] if y_probs.shape[1] == 2 else None

y_preds = [id2label[i] for i in y_pred_ids]

# ===== Gắn dự đoán vào DataFrame =====
df_result = df.iloc[valid_indices].copy()
df_result['predicted'] = y_preds

# ===== Đánh giá =====
y_true = df_result['label_id'].values
y_pred_ids_final = [label2id[label] for label in df_result['predicted']]

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_ids_final, target_names=label_set))
print("\n📈 Summary Metrics:")
print(f"✅ Accuracy:           {accuracy_score(y_true, y_pred_ids_final):.4f}")
print(f"✅ Precision (macro):  {precision_score(y_true, y_pred_ids_final, average='macro'):.4f}")
print(f"✅ Recall (macro):     {recall_score(y_true, y_pred_ids_final, average='macro'):.4f}")
print(f"✅ F1 Score (weighted):{f1_score(y_true, y_pred_ids_final, average='weighted'):.4f}")

# ===== Confusion Matrix =====
cm = confusion_matrix(y_true, y_pred_ids_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_set,
            yticklabels=label_set)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ===== ROC (nếu binary classification) =====
if len(label_set) == 2 and y_scores is not None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===== Lưu kết quả =====
df_result.to_csv("test_labels_with_predictions_custom.csv", index=False)
print("📁 Đã lưu kết quả vào: test_labels_with_predictions_custom.csv")
