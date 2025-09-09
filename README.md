# Lab 4 – KNN Classification

## 📌 ภาพรวม (Overview)

แลปนี้เป็นตัวอย่างการจำแนกกลุ่มผู้เข้าร่วม (Down / TD) โดยใช้ **K-Nearest Neighbors (KNN)**

- ใช้คะแนนจากแบบทดสอบ: Peabody, Raven, SAQ
- รับข้อมูลทดสอบจากผู้ใช้
- คำนวณ **Euclidean Distance** เพื่อหาความใกล้เคียงกับข้อมูลฝึก
- ตัดสินใจจาก K เพื่อนบ้านที่ใกล้ที่สุด

---

## ⚙️ ความต้องการ (Requirements)

````bash
pip install numpy pandas
## 🧩 โครงสร้างโค้ด (Code Structure)

1. **โหลดและเตรียมข้อมูล (Load & Prepare Dataset)**
   - โหลดไฟล์ CSV ด้วย `pandas.read_csv`
   - เลือก features (`Peabody`, `Raven`, `SAQ`)
   - แปลง label `Group` → Down = 0, TD = 1
   - รวม features และ label เป็น numpy array

2. **รับข้อมูลทดสอบ (User Input for Test Data)**
   - รับค่า x1, x2, x3 จากผู้ใช้

3. **คำนวณ Euclidean Distance**
   - คำนวณระยะห่างระหว่าง test data กับ train data แต่ละตัว

4. **เรียงลำดับระยะห่าง (Sort Distances)**
   - หาค่า index ของเพื่อนบ้านที่ใกล้ที่สุด

5. **เลือก K เพื่อนบ้าน (Select K Nearest Neighbors)**
   - K = 3 (สามารถปรับได้)
   - ดึงคลาสของ K เพื่อนบ้านที่ใกล้ที่สุด

6. **นับจำนวนคลาสและทำนาย (Count Classes & Predict)**
   - นับจำนวนแต่ละคลาสใน K เพื่อนบ้าน
   - เลือกคลาสที่มีจำนวนมากที่สุดเป็นผลลัพธ์

7. **แสดงผลลัพธ์ (Display Result)**
   - แสดงผล test data และผลลัพธ์การทำนาย
## 📂 ตัวอย่างโค้ด (Code Example)

```python
import numpy as np
import pandas as pd

# โหลดและเตรียมข้อมูล
file_path = r"Dataset path learning floor matrix task.csv"
data = pd.read_csv(file_path)
X = data[['Peabody', 'Raven', 'SAQ']].values
y = data['Group'].map({'Down': 0, 'TD': 1}).values
train_data = np.column_stack((X, y))

# รับข้อมูลทดสอบจากผู้ใช้
x1_test = float(input("กรอกค่า x1 (Peabody): "))
x2_test = float(input("กรอกค่า x2 (Raven): "))
x3_test = float(input("กรอกค่า x3 (SAQ): "))
test_data = np.array([x1_test, x2_test, x3_test])

# คำนวณ Euclidean Distance
distances = np.sqrt(np.sum((train_data[:, :3] - test_data) ** 2, axis=1))
indices = np.argsort(distances)

# เลือก K เพื่อนบ้าน
K = 3
nearest_classes = train_data[indices[:K], 3]

# นับจำนวนคลาสและทำนาย
class_counts = np.bincount(nearest_classes.astype(int))
predicted_class = np.argmax(class_counts)

# แสดงผล
print("\nTest Data:", test_data)
if predicted_class == 0:
    print("ผลลัพธ์การทำนาย: Down Syndrome")
else:
    print("ผลลัพธ์การทำนาย: TD (ปกติ)")

---

### 📝 หมายเหตุ (Notes)

```markdown
## 📝 หมายเหตุ (Notes)

- K สามารถปรับเปลี่ยนได้ตามต้องการ
- ค่า Euclidean Distance ใช้สำหรับวัดความใกล้เคียงระหว่างตัวอย่าง
- อย่าลืมแทนที่ `"Dataset path learning floor matrix task.csv"` ด้วย path จริงของชุดข้อมูลคุณ
- การทำนายใช้ majority vote ของ K เพื่อนบ้านใกล้ที่สุด



# Lab 5 – การประเมินโมเดลการเรียนรู้ของเครื่อง

## 📂 ชุดข้อมูล (Dataset)

ไฟล์นำเข้า: `Dataset path learning floor matrix task.csv`

**Features (คุณลักษณะ):**

- Peabody
- Raven
- SAQ

**Target (เป้าหมาย):**

- Group → แปลงเป็น Down = 0, TD = 1

**การจัดการค่าที่หายไป (Missing Values):**

- ใช้ mean imputation

## 🧩 โครงสร้างโค้ด (Code Structure)

1. **โหลดชุดข้อมูล (Load Dataset)**

   - ใช้ `pandas.read_csv` เพื่อนำเข้าข้อมูล
   - จัดการค่าที่หายไปด้วย `SimpleImputer`

2. **ฟังก์ชัน Cross Validation**

   - ทำการ K-Fold Cross Validation (n=5)
   - รองรับหลายโมเดล:
     - Naive Bayes
     - Decision Tree
     - KNN

3. **ตัวชี้วัดการประเมิน (Evaluation Metrics)**

   - สำหรับแต่ละ fold จะคำนวณ:
     - Confusion Matrix
     - Accuracy
     - Recall
     - Precision
     - F1-Measure
   - คำนวณค่าเฉลี่ยเมื่อจบการทดลอง

4. **สรุปผลลัพธ์ของทุกโมเดล (Summary of All Models)**
   - แสดงผลรวมของแต่ละโมเดล
   - คำนวณค่าเฉลี่ยสุดท้าย

## 📊 ตัวอย่างผลลัพธ์ (Example Output)

```yaml
================ Fold 1 (NaiveBayes) ================
Confusion Matrix:
               Pred Down   Pred TD
Actual Down         5          2
Actual TD           1          7

Accuracy : 85.71%
Recall   : 87.50%
Precision: 77.78%
F-Measure: 82.35%
...

=========== Summary NaiveBayes (Average from 5 folds) ===========
Mean Accuracy : 82.50%
Mean Recall   : 83.00%
Mean Precision: 80.50%
Mean F-Measure: 81.70%

## 📝 หมายเหตุ (Notes)

- สามารถปรับค่า k ใน KNN ได้ โดยค่าดีฟอลต์คือ k=3
- ค่า random seed ถูกตั้งไว้ที่ 42 เพื่อให้ผลลัพธ์ทำซ้ำได้ (reproducibility)
- อย่าลืมแทนที่ `"Dataset path learning floor matrix task.csv"` ด้วย path จริงของชุดข้อมูลคุณ
````
