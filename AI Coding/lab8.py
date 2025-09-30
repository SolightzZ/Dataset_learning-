import csv
import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ------------------------------
# 1. โหลด CSV (อ่านแบบ pure python)
# ------------------------------
filename = "Dataset path learning floor matrix task.csv"
data = []
with open(filename, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        data.append(row)

# ------------------------------
# 2. เตรียมข้อมูล
# ------------------------------
data = np.array(data)

# ลบคอลัมน์ที่ไม่ใช้
use_cols = [i for i, col in enumerate(header) if col not in ["ID", "Counterbalancing floor matrix task"]]
data = data[:, use_cols]
header = [header[i] for i in use_cols]

# แยก features และ target
X_raw = data[:, 1:]   # ไม่เอา Group
y_raw = data[:, 0]    # Group

# แก้ missing values
for j in range(X_raw.shape[1]):
    col = X_raw[:, j]
    try:
        col.astype(float)
        mask = col == ""
        median = np.nanmedian([float(v) for v in col if v != ""])
        col[mask] = str(median)
        X_raw[:, j] = col
    except:
        cnt = Counter([v for v in col if v != ""])
        mode = cnt.most_common(1)[0][0]
        col[col == ""] = mode
        X_raw[:, j] = col

# ------------------------------
# 3. Label Encoding
# ------------------------------
def label_encode(arr):
    uniq = list(sorted(set(arr)))
    mapping = {u: i for i, u in enumerate(uniq)}
    return np.array([mapping[v] for v in arr]), mapping

categorical_idx = []
for j in range(X_raw.shape[1]):
    try:
        X_raw[:, j].astype(float)
    except:
        categorical_idx.append(j)

for j in categorical_idx:
    X_raw[:, j], _ = label_encode(X_raw[:, j])

y, y_map = label_encode(y_raw)
num_classes = len(y_map)

# ------------------------------
# 4. Normalize
# ------------------------------
X = X_raw.astype(float)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# ------------------------------
# 5. Train/Test Split
# ------------------------------
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)
train_size = int(0.7 * len(X))
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ------------------------------
# 6. Helper: Metrics
# ------------------------------
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"===== {name} =====")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}\n")
    return [acc, prec, rec, f1]

results = {}
results["Naive Bayes"] = evaluate_model("Naive Bayes", GaussianNB(), X_train, y_train, X_test, y_test)
results["Decision Tree"] = evaluate_model("Decision Tree", DecisionTreeClassifier(random_state=42), X_train, y_train, X_test, y_test)
results["KNN"] = evaluate_model("KNN", KNeighborsClassifier(n_neighbors=5), X_train, y_train, X_test, y_test)
results["SVM"] = evaluate_model("SVM", SVC(kernel="linear", probability=True), X_train, y_train, X_test, y_test)

# ------------------------------
# 7. ANN (Manual - ของคุณ)
# ------------------------------
class ANN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_true):
        m = len(X)
        dz2 = self.a2 - y_true
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=200):
        for _ in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# one-hot encode
def one_hot(y, num_classes):
    oh = np.zeros((len(y), num_classes))
    for i, v in enumerate(y):
        oh[i, v] = 1
    return oh

y_train_oh = one_hot(y_train, num_classes)
y_test_oh = one_hot(y_test, num_classes)

model = ANN(input_dim=X_train.shape[1], hidden_dim=16, output_dim=num_classes, lr=0.05)
model.train(X_train, y_train_oh, epochs=500)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("===== ANN =====")
print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1 Score: {f1}\n")

results["ANN"] = [acc, prec, rec, f1]

# ------------------------------
# 8. สรุปผล
# ------------------------------
print("=== สรุปผลลัพธ์ ===")
print(f"{'Model':<15}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1-score':>10}")
for name, vals in results.items():
    print(f"{name:<15}{vals[0]:>10.6f}{vals[1]:>12.6f}{vals[2]:>10.6f}{vals[3]:>10.6f}")
