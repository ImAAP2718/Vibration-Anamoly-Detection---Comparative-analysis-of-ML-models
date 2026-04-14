
# Induction Motor Imbalance Fault Detection
# Train on all except 30g, test also on held-out 30g
# With metrics + misclassification analysis

import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob

# Progress bars
from tqdm import tqdm
from tqdm.keras import TqdmCallback

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Signal processing
from scipy import signal

# Scikit‑learn: preprocessing, splitting, resampling, metrics, models
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Deep learning (TensorFlow / Keras)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. PATHS
base = r"C:\Users\Lenovo\Desktop\ML_code"

paths = {
    "normal": base + r"\normal",
    "6g":     base + r"\imbalance\6g",
    "10g":    base + r"\imbalance\10g",
    "15g":    base + r"\imbalance\15g",
    "20g":    base + r"\imbalance\20g",
    "25g":    base + r"\imbalance\25g",
    "30g":    base + r"\imbalance\30g",   # external test ONLY
    "35g":    base + r"\imbalance\35g",
}

# 2. HELPER FUNCTIONS

def downSampler(df, start, step):
    """Downsample large DataFrame by averaging blocks of 'step' rows."""
    data = df.values
    n = len(data) // step
    data = data[:n * step]
    data = data.reshape(n, step, data.shape[1]).mean(axis=1)
    return pd.DataFrame(data)

def FFT(df):
    """FFT-based autocorrelation along rows."""
    arr = df.values
    ac = signal.fftconvolve(arr, arr[::-1], mode="full", axes=0)
    return pd.DataFrame(ac)

def read_all_csv(folder, label):
    files = glob(folder + r"\*.csv")
    print(f"\n[INFO] ({label}) Reading {len(files)} files from {folder}")
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    df = pd.DataFrame()
    for f in tqdm(files, desc=f"Read {label}", unit="file"):
        low = pd.read_csv(f, header=None)
        df = pd.concat([df, low], ignore_index=True)
    return df

def print_metrics(name, y_true, y_pred):
    """Print classification report and return precision, recall, F1 for class 1."""
    print(f"\n{name} classification report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Imbalance"]))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None
    )
    print(f"{name} (class 1 = Imbalance) -> "
          f"Precision: {prec[1]:.4f}, Recall: {rec[1]:.4f}, F1: {f1[1]:.4f}")
    return prec[1], rec[1], f1[1]

def analyze_misclassifications(name, y_true, y_pred, X_test=None, test_type="internal"):
    """Analyze and print misclassified data points."""
    misclassified = y_true != y_pred
    misclassified_indices = np.where(misclassified)[0]
    
    print(f"\n{test_type.upper()} - {name}")
    print(f"Total misclassifications: {len(misclassified_indices)} out of {len(y_true)}")
    print(f"Misclassification rate: {len(misclassified_indices)/len(y_true)*100:.2f}%")
    
    if len(misclassified_indices) > 0:
        print("First 5 misclassified samples:")
        for i, idx in enumerate(misclassified_indices[:5]):
            true_class = "Normal" if y_true[idx] == 0 else "Imbalance"
            pred_class = "Normal" if y_pred[idx] == 0 else "Imbalance"
            print(f"  Index {idx}: True={true_class}, Pred={pred_class}")
            if X_test is not None:
                print(f"    First 3 features: [{X_test[idx][:3]}]")
        if len(misclassified_indices) > 5:
            print(f"  ... and {len(misclassified_indices) - 5} more")
    else:
        print("No misclassifications! Perfect performance.")
    
    return len(misclassified_indices)

def plot_30g_confusion_with_numbers(model_name, y_true_30g, y_pred_30g):
    n_correct = np.sum((y_true_30g == 1) & (y_pred_30g == 1))
    n_wrong = np.sum((y_true_30g == 1) & (y_pred_30g == 0))

    print(f"\n{model_name} - 30g external test:")
    print(f"  Imbalance correctly predicted as Imbalance: {n_correct}")
    print(f"  Imbalance misclassified as Normal: {n_wrong}")
    print(f"  Total 30g samples: {len(y_true_30g)}")

    cm_30g = np.array([[n_correct, n_wrong]])

    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Plot WITHOUT automatic numbers
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_30g,
        display_labels=["Imbalance"]
    )
    disp.plot(cmap="viridis", values_format=None, ax=ax)  # ← KEY: values_format=None

    # Manually add clean white numbers
    for j in range(cm_30g.shape[1]):
        ax.text(j, 0, f"{cm_30g[0, j]}", ha="center", va="center", color="white", fontsize=14)

    ax.set_xlabel("Predicted: [Imbalance, Normal]")
    ax.set_ylabel("True: Imbalance only")
    ax.set_title(f"{model_name} - 30g Confusion\nCorrect Imbalance: {n_correct}, As Normal: {n_wrong}")
    plt.tight_layout()
    plt.show()


# 3. LOAD RAW DATA
print("=== Stage 1: Loading raw data ===")
data_n   = read_all_csv(paths["normal"], "normal")
data_6g  = read_all_csv(paths["6g"],    "6g")
data_10g = read_all_csv(paths["10g"],   "10g")
data_15g = read_all_csv(paths["15g"],   "15g")
data_20g = read_all_csv(paths["20g"],   "20g")
data_25g = read_all_csv(paths["25g"],   "25g")
data_35g = read_all_csv(paths["35g"],   "35g")

print("\n=== Stage 1b: Loading 30g external data ===")
data_30g = read_all_csv(paths["30g"], "30g_external")  # external test only

print("\nRaw shapes:")
print("normal:", data_n.shape)
print("6g    :", data_6g.shape)
print("10g   :", data_10g.shape)
print("15g   :", data_15g.shape)
print("20g   :", data_20g.shape)
print("25g   :", data_25g.shape)
print("35g   :", data_35g.shape)
print("30g   :", data_30g.shape)

# 4. DOWNSAMPLING
print("\n=== Stage 2: Downsampling each class ===")
step = 5000
classes = [
    ("normal", "data_n"),
    ("6g", "data_6g"),
    ("10g", "data_10g"),
    ("15g", "data_15g"),
    ("20g", "data_20g"),
    ("25g", "data_25g"),
    ("35g", "data_35g"),
    ("30g_external", "data_30g"),
]
for name, var in tqdm(classes, desc="Downsampling classes"):
    locals()[var] = downSampler(locals()[var], 0, step)

print("Downsampled shapes:")
for name, var in classes:
    print(f"{name}: {locals()[var].shape}")

# 5. FFT AUTOCORR
print("\n=== Stage 3: FFT autocorrelation for each class ===")
for name, var in tqdm(classes, desc="FFT classes"):
    locals()[var] = FFT(locals()[var])

# Keep first 15 columns
for name, var in classes:
    locals()[var] = locals()[var].iloc[:, :15]

# 6. BUILD TRAIN DATA (EXCLUDE 30g)
print("\n=== Stage 4: Building labels and concatenating (excluding 30g) ===")
# 0 = normal, 1 = imbalance
y_normal  = pd.DataFrame(np.zeros(len(data_n), dtype=int))
y_6g      = pd.DataFrame(np.ones(len(data_6g), dtype=int))
y_10g     = pd.DataFrame(np.ones(len(data_10g), dtype=int))
y_15g     = pd.DataFrame(np.ones(len(data_15g), dtype=int))
y_20g     = pd.DataFrame(np.ones(len(data_20g), dtype=int))
y_25g     = pd.DataFrame(np.ones(len(data_25g), dtype=int))
y_35g     = pd.DataFrame(np.ones(len(data_35g), dtype=int))

y_all = pd.concat(
    [y_normal, y_6g, y_10g, y_15g, y_20g, y_25g, y_35g],
    ignore_index=True
)
data_all = pd.concat(
    [data_n, data_6g, data_10g, data_15g, data_20g, data_25g, data_35g],
    ignore_index=True
)

y_all.columns = ["y"]
data_all["y"] = y_all["y"]

print("\nClass counts before balancing:")
print(data_all["y"].value_counts())

# 7. BALANCE CLASSES
print("\n=== Stage 5: Balancing classes (excluding 30g) ===")
class_0 = data_all[data_all["y"] == 0]
class_1 = data_all[data_all["y"] == 1]

class_1_down = resample(
    class_1,
    replace=False,
    n_samples=len(class_0),
    random_state=42
)

balanced_data = pd.concat([class_0, class_1_down])
print("Balanced counts:\n", balanced_data["y"].value_counts())

plt.figure(figsize=(6, 4))
sns.barplot(x=["Normal", "Imbalance"], y=balanced_data["y"].value_counts().values)
plt.title("Balanced Class Distribution (excl. 30g)")
plt.show()

# 8. TRAIN–TEST SPLIT + SCALING
print("\n=== Stage 6: Train/test split and scaling ===")
X = balanced_data.iloc[:, :-1].values
y = balanced_data["y"].values

# 75% train, 25% internal test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=True, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("X_train:", X_train.shape, "X_test:", X_test.shape)

# External 30g test set (all imbalance = class 1)
X_30g = scaler.transform(data_30g.values)
y_30g = np.ones(len(data_30g), dtype=int)

# 9. SVM (grid search) + metrics
print("\n=== Stage 7: Training SVM (grid search) ===")
C_values = [0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, 1]

param_grid = {
    "C": C_values,
    "gamma": gamma_values,
    "kernel": ["rbf"],
}

grid = GridSearchCV(
    SVC(random_state=1),
    param_grid,
    cv=3,
    n_jobs=-1,
    scoring="accuracy",
)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

best_svm = grid.best_estimator_
best_svm_acc_int = best_svm.score(X_test, y_test)
best_svm_acc_30g = best_svm.score(X_30g, y_30g)
print("Best SVM Test Acc (internal):", best_svm_acc_int)
print("Best SVM Test Acc (external 30g):", best_svm_acc_30g)

svm_preds_int = best_svm.predict(X_test)
svm_preds_30g = best_svm.predict(X_30g)

svm_prec_int, svm_rec_int, svm_f1_int = print_metrics("SVM (internal)", y_test, svm_preds_int)
svm_prec_30g, svm_rec_30g, svm_f1_30g = print_metrics("SVM (external 30g)", y_30g, svm_preds_30g)

analyze_misclassifications("SVM", y_test, svm_preds_int, X_test, "Internal")
analyze_misclassifications("SVM", y_30g, svm_preds_30g, X_30g, "External 30g")

# Internal confusion matrix (normal 2x2)
plt.figure(figsize=(5, 4))
cm_int = confusion_matrix(y_test, svm_preds_int, labels=[0, 1])
ConfusionMatrixDisplay(cm_int, display_labels=["Normal", "Imbalance"]).plot(
    cmap="viridis", values_format="d", ax=plt.gca()
)
plt.title("Best SVM Confusion Matrix (Internal Test)")
plt.tight_layout()
plt.show()

# External 30g confusion (correct vs as-normal)
plot_30g_confusion_with_numbers("SVM", y_30g, svm_preds_30g)

# 10. kNN
print("\n=== Stage 8: Training kNN for different k ===")
k_values = list(range(2, 8))
knn_accs_int = []
knn_accs_30g = []

for k in k_values:
    print(f"\n--- kNN (k={k}) ---")
    knn_tmp = KNeighborsClassifier(n_neighbors=k)
    knn_tmp.fit(X_train, y_train)
    acc_int = knn_tmp.score(X_test, y_test)
    acc_30g = knn_tmp.score(X_30g, y_30g)
    knn_accs_int.append(acc_int)
    knn_accs_30g.append(acc_30g)
    print("Test Acc (internal):", acc_int)
    print("Test Acc (external 30g):", acc_30g)

plt.figure(figsize=(6, 4))
plt.plot(k_values, knn_accs_int, marker="o", label="Internal test")
plt.plot(k_values, knn_accs_30g, marker="x", label="External 30g")
plt.xlabel("k (n_neighbors)")
plt.ylabel("Test accuracy")
plt.title("kNN Test Accuracy vs k")
plt.xticks(k_values)
plt.ylim(0.8, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

best_k = k_values[int(np.argmax(knn_accs_int))]
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
best_knn_acc_int = best_knn.score(X_test, y_test)
best_knn_acc_30g = best_knn.score(X_30g, y_30g)
print(f"\nBest kNN (k={best_k}) Test Acc (internal): {best_knn_acc_int:.4f}")
print(f"Best kNN (k={best_k}) Test Acc (external 30g): {best_knn_acc_30g:.4f}")

knn_preds_int = best_knn.predict(X_test)
knn_preds_30g = best_knn.predict(X_30g)

knn_prec_int, knn_rec_int, knn_f1_int = print_metrics(f"kNN k={best_k} (internal)", y_test, knn_preds_int)
knn_prec_30g, knn_rec_30g, knn_f1_30g = print_metrics(f"kNN k={best_k} (external 30g)", y_30g, knn_preds_30g)

analyze_misclassifications(f"kNN k={best_k}", y_test, knn_preds_int, X_test, "Internal")
analyze_misclassifications(f"kNN k={best_k}", y_30g, knn_preds_30g, X_30g, "External 30g")

# Internal confusion
plt.figure(figsize=(5, 4))
cm_int = confusion_matrix(y_test, knn_preds_int, labels=[0, 1])
ConfusionMatrixDisplay(cm_int, display_labels=["Normal", "Imbalance"]).plot(
    cmap="viridis", values_format="d", ax=plt.gca()
)
plt.title(f"Best kNN (k={best_k}) Confusion Matrix (Internal)")
plt.tight_layout()
plt.show()

# External 30g confusion
plot_30g_confusion_with_numbers(f"kNN (k={best_k})", y_30g, knn_preds_30g)

# 11. DNN
print("\n=== Stage 9: Training DNN ===")
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

dnn = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid"),
])

dnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002),
            loss="binary_crossentropy",
            metrics=["accuracy"])

history = dnn.fit(
    X_train, y_train,
    epochs=30,
    validation_split=0.2,
    callbacks=[early_stop, TqdmCallback(verbose=0)],
    verbose=0,
)

test_loss_int, test_acc_int = dnn.evaluate(X_test, y_test, verbose=0)
test_loss_30g, test_acc_30g = dnn.evaluate(X_30g, y_30g, verbose=0)
print("DNN Test Accuracy (internal):", test_acc_int)
print("DNN Test Accuracy (external 30g):", test_acc_30g)

plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("DNN Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("DNN Loss")
plt.legend()
plt.tight_layout()
plt.show()

dnn_preds_int = (dnn.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
dnn_preds_30g = (dnn.predict(X_30g, verbose=0) > 0.5).astype(int).ravel()

dnn_prec_int, dnn_rec_int, dnn_f1_int = print_metrics("DNN (internal)", y_test, dnn_preds_int)
dnn_prec_30g, dnn_rec_30g, dnn_f1_30g = print_metrics("DNN (external 30g)", y_30g, dnn_preds_30g)

analyze_misclassifications("DNN", y_test, dnn_preds_int, X_test, "Internal")
analyze_misclassifications("DNN", y_30g, dnn_preds_30g, X_30g, "External 30g")

plt.figure(figsize=(5, 4))
cm_int = confusion_matrix(y_test, dnn_preds_int, labels=[0, 1])
ConfusionMatrixDisplay(cm_int, display_labels=["Normal", "Imbalance"]).plot(
    cmap="viridis", values_format="d", ax=plt.gca()
)
plt.title("DNN Confusion Matrix (Internal)")
plt.tight_layout()
plt.show()

plot_30g_confusion_with_numbers("DNN", y_30g, dnn_preds_30g)

# 12. GaussianNB
print("\n=== Stage 10: Gaussian Naive Bayes ===")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_acc_int = gnb.score(X_test, y_test)
gnb_acc_30g = gnb.score(X_30g, y_30g)
print("GNB Test Accuracy (internal):", gnb_acc_int)
print("GNB Test Accuracy (external 30g):", gnb_acc_30g)

gnb_preds_int = gnb.predict(X_test)
gnb_preds_30g = gnb.predict(X_30g)

gnb_prec_int, gnb_rec_int, gnb_f1_int = print_metrics("GaussianNB (internal)", y_test, gnb_preds_int)
gnb_prec_30g, gnb_rec_30g, gnb_f1_30g = print_metrics("GaussianNB (external 30g)", y_30g, gnb_preds_30g)

analyze_misclassifications("GaussianNB", y_test, gnb_preds_int, X_test, "Internal")
analyze_misclassifications("GaussianNB", y_30g, gnb_preds_30g, X_30g, "External 30g")

plt.figure(figsize=(5, 4))
cm_int = confusion_matrix(y_test, gnb_preds_int, labels=[0, 1])
ConfusionMatrixDisplay(cm_int, display_labels=["Normal", "Imbalance"]).plot(
    cmap="viridis", values_format="d", ax=plt.gca()
)
plt.title("GaussianNB Confusion Matrix (Internal)")
plt.tight_layout()
plt.show()

plot_30g_confusion_with_numbers("GaussianNB", y_30g, gnb_preds_30g)

print("\n=== Stage 11: Model Comparison (Accuracy, Precision, Recall, F1) ===")

# Define model names consistently
model_names = [
    f"SVM (C={grid.best_params_['C']}, γ={grid.best_params_['gamma']})",
    f"kNN (k={best_k})",
    "DNN",
    "GaussianNB"
]

# Accuracy dictionaries
model_accuracies_internal = {
    model_names[0]: best_svm_acc_int,
    model_names[1]: best_knn_acc_int,
    model_names[2]: test_acc_int,
    model_names[3]: gnb_acc_int,
}
model_accuracies_external = {
    model_names[0]: best_svm_acc_30g,
    model_names[1]: best_knn_acc_30g,
    model_names[2]: test_acc_30g,
    model_names[3]: gnb_acc_30g,
}

# Precision, Recall, F1 dictionaries (Imbalance class = 1)
precision_internal = {
    model_names[0]: svm_prec_int,
    model_names[1]: knn_prec_int,
    model_names[2]: dnn_prec_int,
    model_names[3]: gnb_prec_int,
}
precision_external = {
    model_names[0]: svm_prec_30g,
    model_names[1]: knn_prec_30g,
    model_names[2]: dnn_prec_30g,
    model_names[3]: gnb_prec_30g,
}

recall_internal = {
    model_names[0]: svm_rec_int,
    model_names[1]: knn_rec_int,
    model_names[2]: dnn_rec_int,
    model_names[3]: gnb_rec_int,
}
recall_external = {
    model_names[0]: svm_rec_30g,
    model_names[1]: knn_rec_30g,
    model_names[2]: dnn_rec_30g,
    model_names[3]: gnb_rec_30g,
}

f1_scores_internal = {
    model_names[0]: svm_f1_int,
    model_names[1]: knn_f1_int,
    model_names[2]: dnn_f1_int,
    model_names[3]: gnb_f1_int,
}
f1_scores_external = {
    model_names[0]: svm_f1_30g,
    model_names[1]: knn_f1_30g,
    model_names[2]: dnn_f1_30g,
    model_names[3]: gnb_f1_30g,
}

# Define consistent plotting function
def plot_metric_comparison(metric_internal, metric_external, metric_name, ylim=(0.8, 1.0)):
    x = np.arange(len(metric_internal))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, list(metric_internal.values()), width, label="Internal test", alpha=0.8)
    plt.bar(x + width/2, list(metric_external.values()), width, label="External 30g", alpha=0.8)
    plt.xticks(x, list(metric_internal.keys()), rotation=15)
    plt.ylim(*ylim)
    plt.ylabel(metric_name)
    plt.title(f"Model {metric_name} Comparison (Internal vs External 30g)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Generate all 4 comparison plots
plot_metric_comparison(model_accuracies_internal, model_accuracies_external, "Test Accuracy")
plot_metric_comparison(precision_internal, precision_external, "Precision (Imbalance Class)")
plot_metric_comparison(recall_internal, recall_external, "Recall (Imbalance Class)")
plot_metric_comparison(f1_scores_internal, f1_scores_external, "F1 Score (Imbalance Class)")


