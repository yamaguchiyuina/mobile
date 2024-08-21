import numpy as np

def extract_patches(image, patch_size):
    """
    パッチサイズが満たない端を含むパッチをそのまま切り取ります。
    image: 入力画像 (numpy配列)
    patch_size: パッチサイズ (高さと幅が同じ前提)
    """
    height, width, channels = image.shape
    patches = []
    
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image[y:min(y+patch_size, height), x:min(x+patch_size, width), :]
            patches.append(patch)
    
    return patches

# 例として、430x640の画像を使用します
image = np.random.rand(430, 640, 3)  # ランダムな画像を生成

# 端を含むパッチをそのまま利用
patch_size = 256
patches = extract_patches(image, patch_size)

# 各パッチの形状を表示
for i, patch in enumerate(patches):
    print(f"Patch {i}: {patch.shape}")
