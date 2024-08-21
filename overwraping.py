import tensorflow as tf

def extract_overlapping_patches(image, patch_size, stride):
    """
    オーバーラップを許容してパッチを生成します。
    image: 入力画像 (numpy配列)
    patch_size: パッチサイズ (高さと幅が同じ前提)
    stride: パッチをスライドさせる量（オーバーラップが生じる）
    """
    patches = tf.image.extract_patches(
        images=tf.expand_dims(image, axis=0),
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    patches = tf.reshape(patches, [-1, patch_size, patch_size, image.shape[-1]])
    
    return patches

# 例として、430x640の画像を使用します
image = np.random.rand(430, 640, 3)  # ランダムな画像を生成

# オーバーラッピングを利用してパッチを切り取る
patch_size = 256
stride = 128  # パッチが50%オーバーラップするように設定
patches = extract_overlapping_patches(image, patch_size, stride)

# 各パッチの形状を表示
for i, patch in enumerate(patches):
    print(f"Patch {i}: {patch.shape}")
