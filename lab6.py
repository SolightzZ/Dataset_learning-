import numpy as np
import pandas as pd

# โหลด dataset
df = pd.read_csv("Dataset path learning floor matrix task.csv")

# สร้าง target = 0 (Down), 1 (TD)
df["target"] = df["Group"].apply(lambda x: 1 if x == "TD" else 0)

# แยกคลาส
class0 = df[df['target'] == 0]   # Down
class1 = df[df['target'] == 1]   # TD

# คำนวณ mean และ variance
mean0 = class0.mean(numeric_only=True)
mean1 = class1.mean(numeric_only=True)
var0 = class0.var(numeric_only=True)
var1 = class1.var(numeric_only=True)

results = {}

# loop ทุก feature ยกเว้น target
for feature in df.select_dtypes(include=[np.number]).drop(columns=['target']).columns:
    mu_i, mu_j = mean0[feature], mean1[feature]
    var_i, var_j = var0[feature], var1[feature]

    # ใช้สูตร d_ij
    d = 0.5 * ((var_j / var_i) + (var_i / var_j) - 2) \
        + 0.5 * ((mu_i - mu_j) ** 2) * ((1 / var_i) + (1 / var_j))

    results[feature] = d

# แปลงเป็น DataFrame เรียงลำดับ
df_result = pd.DataFrame(list(results.items()), columns=['Feature', 'Distance'])
df_result = df_result.sort_values(by="Distance", ascending=False).reset_index(drop=True)

# ปัดทศนิยม
df_result["Distance"] = df_result["Distance"].round(4)

# ✅ สร้างตาราง Markdown
print("\n|-----------------------------------|")
print("| Rank     | Feature   | Distance   |")
print("|----------|-----------|------------|\n")
for i, row in df_result.iterrows():
    print(f"| {i+1}     | {row['Feature']}  | {row['Distance']} ")
print("--------------------------------\n")
