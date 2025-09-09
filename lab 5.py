#Lab 5
import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer   # สำหรับจัดการ NaN


# numpy, pandas → จัดการข้อมูล
# KFold → แบ่งข้อมูลสำหรับ cross-validation
# confusion_matrix, accuracy_score,... → คำนวณ metrics การประเมินผล
# GaussianNB → โมเดล Naive Bayes
# DecisionTreeClassifier → โมเดล Decision Tree
# KNeighborsClassifier → โมเดล KNN (k=3)
# SimpleImputer → จัดการค่าที่หาย (NaN)

# -----------------------------
# 1) โหลด Dataset
# -----------------------------
file_path = r"Dataset path learning floor matrix task.csv"  # เปลี่ยนเป็น path ของไฟล์ dataset ของคุณ
data = pd.read_csv(file_path)

# เตรียม features และ labels
X = data[['Peabody', 'Raven', 'SAQ']].values
y = data['Group'].map({'Down': 0, 'TD': 1}).values

# จัดการ NaN (แทนที่ด้วยค่าเฉลี่ยของแต่ละ column)
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# -----------------------------
# 2) ฟังก์ชัน Cross Validation
# -----------------------------
def run_cross_validation(X, y, model_name="NaiveBayes", n_splits=5, k=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_list, rec_list, prec_list, f1_list = [], [], [], []
    fold = 1  
    
    for train_index, test_index in kf.split(X):
        print(f"\n================ Fold {fold} ({model_name}) ================")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # ---------- เลือกโมเดล ----------
        if model_name == "NaiveBayes":
            model = GaussianNB()
        elif model_name == "DecisionTree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=k)
        else:
            raise ValueError("Model name not supported")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ---------- Evaluate ----------
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        cm_df = pd.DataFrame(cm, index=["Actual Down", "Actual TD"],
                                columns=["Pred Down", "Pred TD"])
        print("Confusion Matrix:")
        print(cm_df)

        acc = accuracy_score(y_test, y_pred)  * 100
        rec = recall_score(y_test, y_pred, pos_label=1)  * 100
        prec = precision_score(y_test, y_pred, pos_label=1)  * 100
        f1 = f1_score(y_test, y_pred, pos_label=1)  * 100

        print(f"Accuracy : {acc:.2f}%")
        print(f"Recall   : {rec:.2f}%")
        print(f"Precision: {prec:.2f}%")
        print(f"F-Measure: {f1:.2f}%")

        acc_list.append(acc)
        rec_list.append(rec)
        prec_list.append(prec)
        f1_list.append(f1)
        
        fold += 1  

    # -----------------------------
    # Summary ของโมเดลนี้
    # -----------------------------
    print(f"\n=========== Summary {model_name} (Average from {n_splits} folds) ===========")
    print(f"Mean Accuracy : {np.mean(acc_list):.2f}%")
    print(f"Mean Recall   : {np.mean(rec_list):.2f}%")
    print(f"Mean Precision: {np.mean(prec_list):.2f}%")
    print(f"Mean F-Measure: {np.mean(f1_list):.2f}%")

    return np.mean(acc_list), np.mean(rec_list), np.mean(prec_list), np.mean(f1_list)

# -----------------------------
# 3) Run Cross Validation ทั้ง 3 โมเดล
# -----------------------------
results = {}
results["NaiveBayes"] = run_cross_validation(X, y, model_name="NaiveBayes", n_splits=5)
results["DecisionTree"] = run_cross_validation(X, y, model_name="DecisionTree", n_splits=5)
results["KNN"] = run_cross_validation(X, y, model_name="KNN", n_splits=5, k=3)

# -----------------------------
# 4) สรุปผลรวมของทั้ง 3 โมเดล
# -----------------------------
print("\n================ Summary of All Models ================")
df_results = pd.DataFrame(results, index=["Accuracy", "Recall", "Precision", "F-Measure"])
print(df_results)

print("\n=========== Final Average across models ===========")
print(df_results.mean(axis=1))