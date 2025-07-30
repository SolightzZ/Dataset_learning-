import numpy as np
import pandas as pd

path = r"diabetes.csv"
df = pd.read_csv(path)

print("\n[❤️] ข้อมูลจากไฟล์ CSV (แสดงทีละ 50 แถว):")
batch_size = 50  

for start in range(0, len(df), batch_size):
    print(df.iloc[start:start+batch_size])
numeric_data = df.select_dtypes(include=[np.number])

min_vals = numeric_data.min(axis=0)
max_vals = numeric_data.max(axis=0)

normalized_data = (numeric_data - min_vals) / (max_vals - min_vals)

print("\n[👌] ข้อมูลที่ผ่านการ Normalization (แสดงทีละ 50 แถว):")
for start in range(0, len(normalized_data), batch_size):
    print(normalized_data.iloc[start:start+batch_size])
