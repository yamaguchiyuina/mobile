import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込み
image_path = 'path_to_your_image.png'  # 元の画像
line_image_path = 'path_to_your_line_image.png'  # 黒線の画像

image = cv2.imread(image_path)
line_image = cv2.imread(line_image_path, cv2.IMREAD_GRAYSCALE)  # グレースケールで読み込む

# 黒線を太くする
kernel_size = 5  # カーネルサイズ（線の太さ）
kernel = np.ones((kernel_size, kernel_size), np.uint8)
thick_line_image = cv2.dilate(line_image, kernel, iterations=1)

# 太くした黒線をカラー画像に変換
thick_line_image_colored = cv2.cvtColor(thick_line_image, cv2.COLOR_GRAY2BGR)

# 黒線部分だけを抽出して、元画像に貼り付ける
mask = thick_line_image != 0
image[mask] = thick_line_image_colored[mask]

# 結果を表示
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
