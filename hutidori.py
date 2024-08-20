from PIL import Image, ImageDraw
import numpy as np
import cv2

# 画像を開く
img = Image.open('path_to_your_image.png')
width, height = img.size

# 画像をNumPy配列に変換
img_np = np.array(img)

# 中心座標を計算
center_x, center_y = width // 2, height // 2

# 緑色の検出
# 緑色はRGB(0, 255, 0) 近似値で検出
green_mask = (img_np[:, :, 1] > 100) & (img_np[:, :, 0] < 100) & (img_np[:, :, 2] < 100)

# 中心からの距離を計算
distances = np.sqrt((np.arange(width) - center_x) ** 2 + (np.arange(height)[:, None] - center_y) ** 2)
distances_masked = np.where(green_mask, distances, np.inf)

# 最も近い緑色のピクセルを探す
min_distance_index = np.unravel_index(np.argmin(distances_masked), distances_masked.shape)
min_distance_pixel = min_distance_index[::-1]  # (y, x) → (x, y) に変換

# 緑色の集まりを検出するための処理
# 色検出をし、輪郭を描くためにOpenCVを使用
gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), 50, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 緑色部分を縁取り
draw = ImageDraw.Draw(img)
for contour in contours:
    for point in contour:
        draw.point(tuple(point[0]), fill=(255, 0, 0))  # 縁取りとして赤色を使用

# 結果の画像を保存
img.save('outlined_image.png')

print(f"中央から一番近い緑色のピクセル座標: {min_distance_pixel}")
