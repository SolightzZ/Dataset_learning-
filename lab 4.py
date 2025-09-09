

# lab 4
import numpy as np
import pandas as pd

# -----------------------------
# 1) โหลดและเตรียมข้อมูล train
# -----------------------------
file_path = r"Dataset path learning floor matrix task.csv"  # เปลี่ยนเป็น path ของคุณ
data = pd.read_csv(file_path)  # โหลดไฟล์ CSV เข้ามาเป็น DataFrame

# เลือกเฉพาะฟีเจอร์ที่ต้องการใช้ (Peabody, Raven, SAQ)
# และแปลงค่าในคอลัมน์ 'Group' เป็นตัวเลข: 'Down' = 0, 'TD' = 1
X = data[['Peabody', 'Raven', 'SAQ']].values  # ดึงข้อมูล features เป็น numpy array
y = data['Group'].map({'Down': 0, 'TD': 1}).values  # แปลง label เป็นตัวเลข

# รวมข้อมูล features และ labels เข้าด้วยกันในรูปแบบ numpy array 
# โดยมีคอลัมน์ 4 คอลัมน์คือ x1, x2, x3, และ class (label)
train_data = np.column_stack((X, y))


# -----------------------------
# 2) รับข้อมูล test จากผู้ใช้
# -----------------------------
# รับค่าคะแนนจากผู้ใช้เพื่อใช้เป็นข้อมูลทดสอบ (test sample)
x1_test = float(input("กรอกค่า x1 (Peabody) - คะแนนวัด ทักษะด้านภาษา / คำศัพท์ (ช่วง 40–120): "))
x2_test = float(input("กรอกค่า x2 (Raven): - คะแนนวัด การคิดเชิงตรรกะ / เหตุผลเชิงนามธรรม (ช่วง 7-25):"))
x3_test = float(input("กรอกค่า x3 (SAQ): - คะแนนวัด ทักษะทางสังคม (ช่วง 6-18):"))
test_data = np.array([x1_test, x2_test, x3_test])  # สร้าง numpy array ของข้อมูลทดสอบ


# -----------------------------
# 3) คำนวณ Euclidean Distance
# -----------------------------
# คำนวณระยะห่างแบบ Euclidean ระหว่างข้อมูลทดสอบกับข้อมูลฝึกแต่ละตัว
# โดยใช้สูตร sqrt( (x1 - x1_test)^2 + (x2 - x2_test)^2 + (x3 - x3_test)^2 )
distances = np.sqrt(np.sum((train_data[:, :3] - test_data) ** 2, axis=1))


# -----------------------------
# 4) เรียงลำดับระยะห่าง
# -----------------------------
# หา index ของข้อมูลฝึกที่มีระยะห่างจากข้อมูลทดสอบใกล้ที่สุดเป็นอันดับต้น ๆ
indices = np.argsort(distances)  # คืนลำดับ index จากระยะใกล้ไปไกล


# -----------------------------
# 5) เลือก K เพื่อนบ้านที่ใกล้ที่สุด
# -----------------------------
K = 3  # จำนวนเพื่อนบ้านที่ใช้ในการตัดสินใจ สามารถเปลี่ยนค่าได้
nearest_classes = train_data[indices[:K], 3]  # ดึงคลาสของ K เพื่อนบ้านใกล้สุด


# -----------------------------
# 6) นับจำนวนคลาส และหาคลาสที่มากที่สุด
# -----------------------------
# นับจำนวนของแต่ละคลาสใน K เพื่อนบ้าน
class_counts = np.bincount(nearest_classes.astype(int))  # แปลงเป็น int ก่อนนับ
predicted_class = np.argmax(class_counts)  # เลือกคลาสที่มีจำนวนมากที่สุด


# -----------------------------
# 7) แสดงผลลัพธ์
# -----------------------------
print("\nTest Data:", test_data)
if predicted_class == 0:
    print("ผลลัพธ์การทำนาย: Down Syndrome")  # ถ้าค่าที่ได้คือ 0 แปลว่า Down Syndrome
else:
    print("ผลลัพธ์การทำนาย: TD (ปกติ)")  # ถ้าไม่ใช่ 0 คือ TD (ปกติ)