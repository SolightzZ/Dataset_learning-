import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# โหลดข้อมูลจากไฟล์ CSV
file_path = r"Dataset path learning floor matrix task.csv"
data = pd.read_csv(file_path)

# แสดงจำนวนแถวในข้อมูล และชื่อคอลัมน์ พร้อมข้อมูลทั้งหมด
print(f"จำนวนแถวทั้งหมดในข้อมูล: {len(data)}")
print("ชื่อคอลัมน์:", data.columns)
print("\nข้อมูลทั้งหมด:")
print(data)  # แสดงข้อมูลทั้งหมดใน DataFrame

# ตรวจสอบขนาดชุดข้อมูล (batch size คือจำนวนข้อมูลทั้งหมด)
batch_size = len(data)
print(f"\nขนาดชุดข้อมูล (batch size): {batch_size}")

# เรียงข้อมูลตามคอลัมน์ 'ID' เพื่อให้ง่ายต่อการดูและจัดการ
data = data.sort_values(by='ID').reset_index(drop=True)

# กำหนดชื่อคอลัมน์ที่ใช้เป็นคุณลักษณะ (features)

# Group — กลุ่มตัวอย่าง (เช่น TD, Down)
# Gender — เพศ (M = ชาย, F = หญิง)
# Age_months — อายุเป็นเดือน
# Peabody — คะแนนจากแบบทดสอบ Peabody (วัดทักษะทางภาษา)
# Raven — คะแนนจากแบบทดสอบ Raven (วัดความสามารถเชิงตรรกะ)
# SAQ — คะแนน SAQ (ข้อมูลเชิงสมรรถภาพ)
# PMA-SR-K1 — คะแนนจากแบบทดสอบ PMA-SR-K1 (สมรรถภาพเชิงสติปัญญา)
# GPT_total — คะแนนรวม GPT (General Processing Task)
# WM_matr_sequential — คะแนนหน่วยความจำทำงาน (Working Memory) แบบ Sequential
# WM_matr_simultaneous — คะแนนหน่วยความจำทำงานแบบ Simultaneous
# Floor Matrix Map — คะแนนจาก Floor Matrix Map (แผนที่)
# Floor Matrix Obs — คะแนนจาก Floor Matrix Obs (สังเกตการณ์)

feature_cols = [
    "Group", "Gender", "Age_months", "Peabody", "Raven", "SAQ", "PMA-SR-K1",
    "GPT_total", "WM_matr_sequential", "WM_matr_simultaneous",
    "Floor Matrix Map", "Floor Matrix Obs"
]

X = data[feature_cols]  # ตัวแปรคุณลักษณะ (features)
y = data["Counterbalancing floor matrix task"]  # ตัวแปรเป้าหมาย (target)

# จัดการค่าที่ขาดหาย (missing values) ในข้อมูลตัวเลข ด้วยการแทนที่ด้วยค่าเฉลี่ยของแต่ละคอลัมน์
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
num_imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

# เติมค่า missing ในคอลัมน์ประเภทข้อความ ด้วยคำว่า 'missing'
categorical_cols = X.select_dtypes(include=['object']).columns
X[categorical_cols] = X[categorical_cols].fillna('missing')

# เข้ารหัสค่าข้อความในตัวแปรคุณลักษณะให้เป็นตัวเลข (Label Encoding)
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# เข้ารหัสค่าข้อความในตัวแปรเป้าหมาย (target) ให้เป็นตัวเลขเช่นกัน
y = y.fillna('missing')
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# สร้างโมเดลต้นไม้ตัดสินใจ (Decision Tree) โดยจำกัดความลึกของต้นไม้ไม่เกิน 3 ชั้น
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X, y)  # ฝึกโมเดลด้วยข้อมูลคุณลักษณะและเป้าหมาย

# วาดภาพต้นไม้ตัดสินใจให้เห็นภาพเข้าใจง่าย
plt.figure(figsize=(15, 10))
plot_tree(
    model,
    feature_names=feature_cols,          # แสดงชื่อคุณลักษณะในกราฟ
    class_names=target_encoder.classes_, # แสดงชื่อคลาสเป้าหมายในกราฟ
    filled=True,                        # เติมสีตามคลาส
    rounded=True,                      # ขอบกราฟเป็นมุมโค้ง
    fontsize=12                       # ขนาดตัวอักษรในกราฟ
)
plt.title("Decision Tree - Floor Matrix Task Classification")  # ชื่อกราฟ
plt.show()
