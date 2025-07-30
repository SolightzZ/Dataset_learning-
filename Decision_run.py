import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# โหลดข้อมูลจากไฟล์ CSV
file_path = r"Dataset path learning floor matrix task.csv"
data = pd.read_csv(file_path)

# แสดงข้อมูลสั้น ๆ
print(f"จำนวนแถวทั้งหมดในข้อมูล: {len(data)}")
print("ชื่อคอลัมน์:", data.columns)

# เรียงข้อมูลตาม ID
data = data.sort_values(by='ID').reset_index(drop=True)

# กำหนด feature columns
feature_cols = [
    "Group", "Gender", "Age_months", "Peabody", "Raven", "SAQ", "PMA-SR-K1",
    "GPT_total", "WM_matr_sequential", "WM_matr_simultaneous",
    "Floor Matrix Map", "Floor Matrix Obs"
]

X = data[feature_cols]
y = data["Counterbalancing floor matrix task"]

# จัดการ missing values ตัวเลข
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
num_imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

# เติม missing ในคอลัมน์ object ด้วย 'missing'
categorical_cols = X.select_dtypes(include=['object']).columns
X[categorical_cols] = X[categorical_cols].fillna('missing')

# เข้ารหัสข้อความใน features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# เข้ารหัส target
y = y.fillna('missing')
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# สร้างโมเดล Decision Tree
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X, y)

# รับ input จากผู้ใช้ (ข้อความภาษาไทย)
print("\n========================================== กรอกข้อมูลสำหรับทำนาย =============================================")

input_gender = input("เพศของคุณ (พิมพ์ M ถ้าเป็นชาย, F ถ้าเป็นหญิง): ").strip().upper()
input_age = float(input("อายุของคุณ (หน่วย: เดือน เช่น 129): "))
input_peabody = float(input("คะแนน Peabody (วัดทักษะทางภาษา) เช่น 90: "))
input_raven = float(input("คะแนน Raven (วัดตรรกะ) เช่น 7: "))
input_saq = float(input("คะแนน SAQ (วัดทักษะทางสังคม) เช่น 40: "))
input_pma_sr_k1 = float(input("คะแนน PMA-SR-K1 (คะแนนจากแบบทดสอบเฉพาะ) เช่น 50: "))
input_gpt_total = float(input("คะแนน GPT_total (คะแนนรวมแบบทดสอบ GPT) เช่น 60: "))
input_wm_seq = float(input("คะแนน WM_matr_sequential (หน่วยความจำแบบเรียงลำดับ) เช่น 90: "))
input_wm_sim = float(input("คะแนน WM_matr_simultaneous (หน่วยความจำแบบพร้อมกัน) เช่น 95: "))
input_floor_map = float(input("คะแนน Floor Matrix Map (คะแนนจากแบบทดสอบ) เช่น 70: "))
input_floor_obs = float(input("คะแนน Floor Matrix Obs (คะแนนจากการสังเกต) เช่น 65: "))

# สร้าง DataFrame สำหรับ input ผู้ใช้
input_df = pd.DataFrame({
    "Group": ['missing'],  # ใส่ค่า placeholder แทน ไม่รับจากผู้ใช้
    "Gender": [input_gender],
    "Age_months": [input_age],
    "Peabody": [input_peabody],
    "Raven": [input_raven],
    "SAQ": [input_saq],
    "PMA-SR-K1": [input_pma_sr_k1],
    "GPT_total": [input_gpt_total],
    "WM_matr_sequential": [input_wm_seq],
    "WM_matr_simultaneous": [input_wm_sim],
    "Floor Matrix Map": [input_floor_map],
    "Floor Matrix Obs": [input_floor_obs]
})

# แปลงข้อมูล object ด้วย label encoder ที่ฝึกไว้ (ตรวจสอบค่าก่อนแปลง)
for col in categorical_cols:
    if col in input_df.columns:
        le = label_encoders[col]
        val = input_df.at[0, col]
        if val in le.classes_:
            input_df[col] = le.transform([val])
        else:
            print(f"Warning: ค่า '{val}' ในคอลัมน์ '{col}' ไม่อยู่ในชุดข้อมูลที่โมเดลฝึกมา แปลงเป็นค่า default 0 แทน")
            input_df[col] = [0]

# ทำนายผล
prediction_encoded = model.predict(input_df)[0]
prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]

print(f"\nผลการทำนาย (Counterbalancing floor matrix task): {prediction_label}")

# หาแถวข้อมูลจริงที่ใกล้เคียงกับ input (คำนวณระยะห่าง)
input_gender_encoded = label_encoders['Gender'].transform([input_gender])[0]

def calculate_distance(row):
    dist_num = np.sqrt(
        (row['Age_months'] - input_age)**2 +
        (row['Peabody'] - input_peabody)**2 +
        (row['Raven'] - input_raven)**2
    )
    dist_gender = 0 if row['Gender'] == input_gender_encoded else 1
    return dist_num + (dist_gender * 10)  # เพิ่มน้ำหนัก 10 ถ้าเพศไม่ตรง

data['distance'] = data.apply(calculate_distance, axis=1)
closest_row = data.loc[data['distance'].idxmin()]

print("\n==========================================  ข้อมูลจริงที่ใกล้เคียงที่สุดจาก dataset ========================================== ")
print(f"Group: {closest_row['Group']}")
print(f"Gender (encoded): {closest_row['Gender']} (input encoded: {input_gender_encoded})")
print(f"Age_months: {closest_row['Age_months']}")
print(f"Peabody: {closest_row['Peabody']}")
print(f"Raven: {closest_row['Raven']}")
print(f"Counterbalancing floor matrix task จริง: {closest_row['Counterbalancing floor matrix task']}")

# แสดงผลสรุป Down Syndrome หรือไม่
if closest_row['Group'].lower() == 'down':
    print("ข้อมูลจริงที่ใกล้เคียงที่สุด: เป็น Down Syndrome")
else:
    print("ข้อมูลจริงที่ใกล้เคียงที่สุด: ไม่เป็น Down Syndrome")

# วาดต้นไม้ตัดสินใจ
plt.figure(figsize=(15, 10))
plot_tree(
    model,
    feature_names=feature_cols,
    class_names=target_encoder.classes_,
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Decision Tree - Floor Matrix Task Classification")
plt.show()


# Map-Observ คือ กลุ่มที่ทำคะแนนด้านการวาดแผนที่ (Mapping) ได้ดีกว่าการสังเกต (Observation)
# Observ-Map คือ กลุ่มที่ทำคะแนนด้านการสังเกต (Observation) ได้ดีกว่าการวาดแผนที่ (Mapping)
#  ระหว่าง “การวาดแผนที่” กับ “การสังเกต” 