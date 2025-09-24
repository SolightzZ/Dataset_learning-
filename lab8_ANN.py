import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# โหลดข้อมูล
df = pd.read_csv("Dataset path learning floor matrix task.csv")

# แปลง target: TD = 1, Down = 0
df["target"] = df["Group"].apply(lambda x: 1 if x == "TD" else 0)

# ลบคอลัมน์ที่ไม่จำเป็น
drop_columns = ["ID", "Group", "Gender", "Counterbalancing floor matrix task"]
X = df.drop(columns=drop_columns + ["target"])
y = df["target"]

# จัดการ missing values
X = X.fillna(X.mean())

# มาตรฐานข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encoding สำหรับ y
y_categorical = to_categorical(y, num_classes=2)

# แบ่งข้อมูล Train 70%, Test 30%
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.3, random_state=42, stratify=y
)

# สร้างโมเดล Keras ANN
model = Sequential()
model.add(Dense(10, input_dim=X.shape[1], activation='relu'))  # Hidden layer
model.add(Dense(2, activation='softmax'))  # Output layer (2 class)

# คอมไพล์โมเดล
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ฝึกโมเดล
model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)

# ทำนาย
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# คำนวณ metric
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# แสดงผลลัพธ์
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
