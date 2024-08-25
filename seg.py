import cv2
import matplotlib.pyplot as plt

# 画像とアノテーション画像を読み込む
image_path = 'path_to_your_image.png'
annotation_path = 'path_to_your_annotation.png'

image = cv2.imread(image_path)
annotation = cv2.imread(annotation_path)

# アノテーション画像を透明度50%に設定
alpha = 0.5
overlay = cv2.addWeighted(annotation, alpha, image, 1 - alpha, 0)

# 結果を表示する
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
