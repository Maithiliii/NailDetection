import torch
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
# Constants
SCALING_MM_PER_PX = 0.1  # adjust as needed
# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/nail_model/weights/best.pt', force_reload=True)
# Load and run inference
img_path = 'test_images/1.png'
img = cv2.imread(img_path)
results = model(img)
# Extract detection data
nail_data = []
det = results.xyxy[0]
for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
    x1, y1, x2, y2 = map(int, xyxy)
    width_px = x2 - x1
    height_px = y2 - y1
    width_mm = width_px * SCALING_MM_PER_PX
    height_mm = height_px * SCALING_MM_PER_PX
    nail_id = i + 1  # 🔥 Shift ID to start from 1

    nail_data.append({
        'id': nail_id,
        'bbox': (x1, y1, x2, y2),
        'height': height_mm,
        'weight': width_mm
    })

    # Draw bounding box with height and weight and ID
    label = f"Nail {nail_id}\n{height_mm:.1f}mm H, {width_mm:.1f}mm W"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Match similar nails using clustering
vectors = np.array([[n['height'], n['weight']] for n in nail_data])
num_clusters = max(1, len(vectors) // 2)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(vectors)
for i, label in enumerate(labels):
    nail_data[i]['cluster'] = label
pairs = []
used = set()
for cluster_id in set(labels):
    cluster_nails = [n for n in nail_data if n['cluster'] == cluster_id]
    for i in range(len(cluster_nails)):
        id_i = cluster_nails[i]['id']
        if id_i in used:
            continue
        closest = None
        min_dist = float('inf')
        for j in range(i + 1, len(cluster_nails)):
            id_j = cluster_nails[j]['id']
            if id_j in used:
                continue
            dist = euclidean(
                [cluster_nails[i]['height'], cluster_nails[i]['weight']],
                [cluster_nails[j]['height'], cluster_nails[j]['weight']]
            )
            if dist < min_dist:
                min_dist = dist
                closest = id_j
        if closest is not None:
            used.add(id_i)
            used.add(closest)
            pairs.append((id_i, closest))
# Output results
print("\n Matched Pairs:")
for a, b in pairs:
    print(f"Nail {a} ↔ Nail {b}")
# Show final annotated image
cv2.imshow("Detection with Height/Weight", img)
cv2.waitKey(0)
cv2.destroyAllWindows()