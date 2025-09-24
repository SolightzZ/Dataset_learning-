import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf

# 1. โหลดข้อมูล
df = pd.read_csv('Dataset path learning floor matrix task.csv')

# 2. ดูข้อมูลเบื้องต้น
print(df.head())

# 3. ลบคอลัมน์ที่ไม่ใช้ (ID, Counterbalancing floor matrix task)
df = df.drop(['ID', 'Counterbalancing floor matrix task'], axis=1)

# 4. จัดการ missing values (เติมด้วยค่ากลาง เช่น median หรือ mode)
for col in df.columns:
    if df[col].dtype == 'O':  # categorical
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# 5. แปลง categorical เป็นตัวเลข (Label Encoding)
label_encoders = {}
for col in ['Group', 'Gender', 'Floor Matrix Map', 'Floor Matrix Obs']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 6. กำหนด X, y
X = df.drop('Group', axis=1).values
y = df['Group'].values

# 7. แบ่งข้อมูล Train/Test 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 8. Normalize feature
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. ANN Model Preparation
# One-hot encoding สำหรับ y_train และ y_test
num_classes = len(np.unique(y))
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes)

# สร้างโมเดล ANN
ann_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

ann_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ann_model.fit(X_train, y_train_oh, epochs=100, batch_size=10, verbose=0)

# ทำนาย ANN
y_pred_ann_prob = ann_model.predict(X_test)
y_pred_ann = np.argmax(y_pred_ann_prob, axis=1)

# ฟังก์ชันช่วยคำนวณ metrics
def print_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f'{model_name}:')
    print(f'  Accuracy = {acc:.4f}')
    print(f'  Precision = {prec:.4f}')
    print(f'  Recall = {rec:.4f}')
    print(f'  F1-Score = {f1:.4f}')
    print('---------------------------')

print_metrics(y_test, y_pred_ann, 'ANN')

# 10. โมเดลอื่น ๆ

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print_metrics(y_test, y_pred_lr, 'Logistic Regression')

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print_metrics(y_test, y_pred_dt, 'Decision Tree')

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print_metrics(y_test, y_pred_rf, 'Random Forest')

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print_metrics(y_test, y_pred_svm, 'SVM')
