import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# โหลดข้อมูลจากไฟล์ CSV
file_path = r"Dataset path learning floor matrix task.csv"  # ใส่ path ของไฟล์ CSV ที่นี่
data = pd.read_csv(file_path)

# แสดงข้อมูลสั้น ๆ
print(f"จำนวนแถวทั้งหมดในข้อมูล: {len(data)}")
print("ชื่อคอลัมน์:", data.columns)

# ตรวจสอบค่าที่ขาดหายไป (NaN)
print("\nข้อมูลที่ขาดหายไปในแต่ละคอลัมน์:")
print(data.isna().sum())

# เลือกเฉพาะคอลัมน์ที่เป็นตัวเลขเพื่อคำนวณค่าเฉลี่ย
numeric_columns = data.select_dtypes(include=[np.number]).columns

# แทนค่าที่ขาดหายไปในคอลัมน์ตัวเลขด้วยค่าเฉลี่ย
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# สำหรับคอลัมน์ที่เป็น categorical (เช่น 'Group', 'Gender', 'Counterbalancing floor matrix task')
# เราสามารถแทนค่าที่ขาดหายไปด้วยค่าที่พบมากที่สุด (mode)
categorical_columns = data.select_dtypes(include=[object]).columns
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# แสดงข้อมูลหลังการเติมค่าที่ขาดหายไป
print("\nข้อมูลหลังจากเติมค่าที่ขาดหายไป:")
print(data.head())

# เลือกฟีเจอร์ที่เป็นตัวเลข (X) และ label (y)
X = data[['Peabody', 'Raven', 'SAQ', 'PMA-SR-K1', 'GPT_total',
          'WM_matr_sequential', 'WM_matr_simultaneous', 'Floor Matrix Map', 'Floor Matrix Obs']].values

# เปลี่ยน 'Group' เป็นตัวเลข 1 สำหรับ Down Syndrome และ 2 สำหรับ TD (ปกติ)
y = data['Group'].map({'Down': 1, 'TD': 2}).values

# แบ่งข้อมูลเป็นชุดฝึก (training) และชุดทดสอบ (test) โดยใช้ train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Naive Bayes
nb_model = GaussianNB()

# ฝึกโมเดล
nb_model.fit(X_train, y_train)

# ทำนายผลลัพธ์จากชุดทดสอบ
y_pred = nb_model.predict(X_test)

# คำนวณความแม่นยำ (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nความแม่นยำของโมเดล: {accuracy * 100:.2f}%")

# รับค่าจากผู้ใช้
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

# แสดงข้อมูลที่ได้รับจากผู้ใช้
print(f"\nข้อมูลที่กรอกเข้ามามีดังนี้:")
print(f"เพศ: {input_gender}")
print(f"อายุ: {input_age} เดือน")
print(f"คะแนน Peabody: {input_peabody}")
print(f"คะแนน Raven: {input_raven}")
print(f"คะแนน SAQ: {input_saq}")
print(f"คะแนน PMA-SR-K1: {input_pma_sr_k1}")
print(f"คะแนน GPT_total: {input_gpt_total}")
print(f"คะแนน WM_matr_sequential: {input_wm_seq}")
print(f"คะแนน WM_matr_simultaneous: {input_wm_sim}")
print(f"คะแนน Floor Matrix Map: {input_floor_map}")
print(f"คะแนน Floor Matrix Obs: {input_floor_obs}")

# สร้าง array ของฟีเจอร์ที่รับจากผู้ใช้ (เลือกฟีเจอร์ทั้งหมด 9 ตัว)
input_features = np.array([[  
    input_peabody, input_raven, input_saq, input_pma_sr_k1, input_gpt_total,
    input_wm_seq, input_wm_sim, input_floor_map, input_floor_obs
]])

# ให้โมเดลทำนายค่า labels ของข้อมูลที่ผู้ใช้กรอก
predicted_label = nb_model.predict(input_features)

# แสดงผลลัพธ์ที่ทำนาย
print(f"\nผลลัพธ์ที่ทำนายจากข้อมูลที่กรอก: {predicted_label[0]}")

# เพิ่มข้อความอธิบายว่า label 1 คือ Down Syndrome
if predicted_label[0] == 1:
    print("ผลลัพธ์: เป็น Down Syndrome")
elif predicted_label[0] == 2:
    print("ผลลัพธ์: เป็นปกติ")
else:
    print("ผลลัพธ์: อาจจะเป็นกรณีอื่น ๆ")


# Map-Observ คือ กลุ่มที่ทำคะแนนด้านการวาดแผนที่ (Mapping) ได้ดีกว่าการสังเกต (Observation)
# Observ-Map คือ กลุ่มที่ทำคะแนนด้านการสังเกต (Observation) ได้ดีกว่าการวาดแผนที่ (Mapping)
#  ระหว่าง “การวาดแผนที่” กับ “การสังเกต” 