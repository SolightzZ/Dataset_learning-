# Lab 5 – การประเมินโมเดลการเรียนรู้ของเครื่อง

## 📌 ภาพรวม (Overview)

แลปนี้สาธิตการประเมินโมเดลการจำแนก (classification) โดยใช้:

- Naive Bayes
- Decision Tree
- K-Nearest Neighbors (KNN)

เราใช้ **5-Fold Cross Validation** และคำนวณตัวชี้วัดหลักดังนี้:

- Accuracy
- Recall
- Precision
- F1-Score

ชุดข้อมูลประกอบด้วยผลการทดสอบทางจิตวิทยา (Peabody, Raven, SAQ) เพื่อจำแนกผู้เข้าร่วมออกเป็นกลุ่ม (Down, TD)

---

## 📂 ชุดข้อมูล (Dataset)

ไฟล์นำเข้า: `Dataset path learning floor matrix task.csv`

**Features (คุณลักษณะ):**

- Peabody
- Raven
- SAQ

**Target (เป้าหมาย):**

- Group → แปลงเป็น `Down = 0`, `TD = 1`

**การจัดการค่าที่หายไป:**

- ใช้ mean imputation สำหรับค่า NaN

---

## 🚀 วิธีการรัน (How to Run)

```bash
python lab5.py
```

🧩 โครงสร้างโค้ด (Code Structure)

โหลดชุดข้อมูล (Load Dataset)

ใช้ pandas.read_csv เพื่อนำเข้าข้อมูล

จัดการค่าที่หายไปด้วย SimpleImputer

ฟังก์ชัน Cross Validation

ทำการ K-Fold Cross Validation (n=5)

รองรับหลายโมเดล:

NaiveBayes

DecisionTree

KNN

ตัวชี้วัดการประเมิน (Evaluation Metrics)

สำหรับแต่ละ fold จะคำนวณ:

Confusion Matrix

Accuracy

Recall

Precision

F1-Measure

เมื่อจบการทดลอง → คำนวณค่าเฉลี่ยของทุก fold

สรุปผลลัพธ์ของทุกโมเดล (Summary of All Models)

แสดงผลการทำงานรวมของแต่ละโมเดล

คำนวณค่าเฉลี่ยสุดท้ายของแต่ละโมเดล

📊 ตัวอย่างผลลัพธ์ (Example Output)
================ Fold 1 (NaiveBayes) ================
Confusion Matrix:
Pred Down Pred TD
Actual Down 5 2
Actual TD 1 7

Accuracy : 85.71%
Recall : 87.50%
Precision: 77.78%
F-Measure: 82.35%
...

=========== Summary NaiveBayes (Average from 5 folds) ===========
Mean Accuracy : 82.50%
Mean Recall : 83.00%
Mean Precision: 80.50%
Mean F-Measure: 81.70%

✅ การเปรียบเทียบผลลัพธ์ (Results Comparison)
| Metric | Naive Bayes | Decision Tree | KNN |
| ---------- | ----------- | ------------- | ------- |
| Accuracy | xx.xx % | xx.xx % | xx.xx % |
| Recall | xx.xx % | xx.xx % | xx.xx % |
| Precision | xx.xx % | xx.xx % | xx.xx % |
| F1-Measure | xx.xx % | xx.xx % | xx.xx % |

📝 หมายเหตุ (Notes)

สามารถปรับค่า k ใน KNN ได้ โดยค่าดีฟอลต์คือ k=3

ค่า random seed ถูกตั้งไว้ที่ 42 เพื่อให้ผลลัพธ์ทำซ้ำได้ (reproducibility)

อย่าลืมแทนที่ "Dataset path learning floor matrix task.csv" ด้วย path จริงของชุดข้อมูลคุณ

ถ้าต้องการ ผมสามารถเพิ่ม **ตารางโค้ดตัวอย่างการประเมินโมเดล Naive Bayes, Decision Tree, และ KNN** ให้ดูชัดเจนเป็น Markdown Table ด้วย จะทำให้ README ดูสมบูรณ์แบบมากขึ้น

คุณอยากให้ผมเพิ่มตรงนั้นไหม?
