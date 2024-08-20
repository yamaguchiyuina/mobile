from PIL import Image

# 画像を開く
img = Image.open('path_to_your_image.png')
pixels = img.load()

width, height = img.size
black_pixel = (0, 0, 0)

# 黒ピクセルが100ピクセル横に連続するy座標を探す
min_y = height
for y in range(height):
    count = 0
    for x in range(width):
        if pixels[x, y] == black_pixel:
            count += 1
        else:
            count = 0

        if count >= 100:
            min_y = min(min_y, y)

# 結果を表示
if min_y < height:
    print(f"黒のピクセルが100ピクセル横に続く最も下のy座標: {min_y}")
else:
    print("100ピクセル横に連続する黒のピクセルは見つかりませんでした。")

from PIL import Image

# 画像を開く
img = Image.open('path_to_your_image.png')

# 指定したy座標 (min_y) より上の部分を切り取る
# ここでは min_y = 100 の場合と仮定しています
y_cutoff = min_y

# 画像サイズを取得
width, height = img.size

# 切り取る範囲を指定
left = 0
top = y_cutoff
right = width
bottom = height

# 画像を切り取る
cropped_img = img.crop((left, top, right, bottom))

# 切り取った画像を保存
cropped_img.save('cropped_image.png')
