import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- โหลดข้อมูลจากไฟล์ CSV ---
df = pd.read_csv("Dataset path learning floor matrix task.csv")  # 🔁 เปลี่ยน path ให้ถูกต้อง

# --- เลือกเฉพาะฟีเจอร์ตัวเลขที่ต้องการใช้ ---
features = ['Age_months', 'Peabody', 'Raven', 'SAQ',
            'PMA-SR-K1', 'GPT_total', 'WM_matr_sequential', 'WM_matr_simultaneous']
data = df[features].dropna().reset_index(drop=True)
data_np = data.values

# --------------------------
# 1. Sequential Clustering
# --------------------------
def sequential_clustering(data, threshold=50):
    clusters = []
    centroids = []
    labels = []

    for point in data:
        if not clusters:
            clusters.append([point])
            centroids.append(point)
            labels.append(0)
        else:
            distances = [np.linalg.norm(point - np.mean(cluster, axis=0)) for cluster in clusters]
            min_dist = min(distances)
            idx = distances.index(min_dist)
            if min_dist < threshold:
                clusters[idx].append(point)
                centroids[idx] = np.mean(clusters[idx], axis=0)
                labels.append(idx)
            else:
                clusters.append([point])
                centroids.append(point)
                labels.append(len(clusters)-1)
    return np.array(labels), np.array(centroids)

seq_labels, seq_centroids = sequential_clustering(data_np, threshold=50)

# --------------------------
# 2. Leader Clustering
# --------------------------
def leader_clustering(data, threshold=50):
    leaders = []
    clusters = []
    labels = []

    for point in data:
        assigned = False
        for i, leader in enumerate(leaders):
            dist = np.linalg.norm(point - leader)
            if dist < threshold:
                clusters[i].append(point)
                labels.append(i)
                assigned = True
                break
        if not assigned:
            leaders.append(point)
            clusters.append([point])
            labels.append(len(leaders) - 1)
    return np.array(labels), np.array(leaders)

leader_labels, leader_centroids = leader_clustering(data_np, threshold=50)

# --------------------------
# 3. K-Means Clustering
# --------------------------
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data_np)
kmeans_labels = kmeans.labels_
kmeans_centroids = kmeans.cluster_centers_

# --------------------------
# แสดงผลใน CMD แบบตารางสวยงาม
# --------------------------

def show_result(method_name, labels, centroids):
    print(f"\n{'='*40}")
    print(f"{method_name:^40}")
    print(f"{'='*40}")
    print(f"✅ จำนวนกลุ่มที่พบ: {len(np.unique(labels))}\n")

    # แสดง Labels เป็นตาราง
    df_labels = pd.DataFrame({'Index': range(len(labels)), 'Cluster': labels})
    print("📌 Labels ของข้อมูลแต่ละรายการ:")
    print(df_labels.head(10).to_string(index=False))  # แสดง 10 รายการแรก
    print("... (แสดงเฉพาะ 10 รายการแรก)\n")

    # แสดง Centroids เป็นตาราง
    df_centroids = pd.DataFrame(centroids, columns=features)
    print("📍 Centroids ของแต่ละกลุ่ม:")
    print(df_centroids.round(2).to_string(index=True))
    print("\n")


# เรียกใช้ฟังก์ชันแสดงผล
show_result("1. Sequential Clustering", seq_labels, seq_centroids)
show_result("2. Leader Clustering", leader_labels, leader_centroids)
show_result("3. K-Means Clustering", kmeans_labels, kmeans_centroids)

# --------------------------
# แสดงกราฟทั้ง 3 วิธี
# --------------------------
def plot_clusters(data, labels, centroids, title, feature_x='Age_months', feature_y='Peabody'):
    x_idx = features.index(feature_x)
    y_idx = features.index(feature_y)

    plt.figure(figsize=(7, 5))
    plt.scatter(data[:, x_idx], data[:, y_idx], c=labels, cmap='rainbow', alpha=0.6)
    plt.scatter(centroids[:, x_idx], centroids[:, y_idx], c='black', marker='x', s=100, label='Centroids')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_clusters(data_np, seq_labels, seq_centroids, "1. Sequential Clustering")
plot_clusters(data_np, leader_labels, leader_centroids, "2. Leader Clustering")
plot_clusters(data_np, kmeans_labels, kmeans_centroids, "3. K-Means Clustering")
