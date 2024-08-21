import tensorflow as tf
import numpy as np

def pad_image(image, target_height, target_width):
    """
    画像を指定した高さと幅にゼロパディングで拡張します。
    image: 入力画像 (numpy配列)
    target_height: パッチサイズに合わせた目標の高さ
    target_width: パッチサイズに合わせた目標の幅
    """
    height, width, channels = image.shape

    # 余剰部分の計算
    pad_height = target_height - height % target_height if height % target_height != 0 else 0
    pad_width = target_width - width % target_width if width % target_width != 0 else 0

    # パディングの実行
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    
    return padded_image

def image_to_patches(image, patch_size):
    """
    画像を指定したパッチサイズでスライドして切り取ります。
    image: 入力画像 (numpy配列)
    patch_size: パッチサイズ (高さと幅が同じ前提)
    """
    patches = tf.image.extract_patches(
        images=tf.expand_dims(image, axis=0),
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    patches = tf.reshape(patches, [-1, patch_size, patch_size, image.shape[-1]])
    
    return patches

# 例として、430x640の画像を使用します
image = np.random.rand(430, 640, 3)  # ランダムな画像を生成

# 画像をパディングして、サイズを調整
padded_image = pad_image(image, 256, 256)

# パッチサイズ256で画像をスライドして切り取る
patch_size = 256
patches = image_to_patches(padded_image, patch_size)

# 各パッチの形状を表示
for i, patch in enumerate(patches):
    print(f"Patch {i}: {patch.shape}")
