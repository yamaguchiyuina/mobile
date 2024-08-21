from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D
import tensorflow as tf

def bottleneck(x, in_channels, t, c, n, s):
    # ボトルネック層の定義（詳細は省略）
    pass

def MobileNetV2_Segmentation(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = tf.nn.relu6(x)

    x = bottleneck(x, in_channels=32,  t=1, c=16,  n=1, s=1)
    x = bottleneck(x, in_channels=16,  t=6, c=24,  n=2, s=2)
    x = bottleneck(x, in_channels=24,  t=6, c=32,  n=3, s=2)
    x = bottleneck(x, in_channels=32,  t=6, c=64,  n=4, s=2)
    x = bottleneck(x, in_channels=64,  t=6, c=96,  n=3, s=1)
    x = bottleneck(x, in_channels=96,  t=6, c=160, n=3, s=2)
    x = bottleneck(x, in_channels=160, t=6, c=320, n=1, s=1)

    x = Conv2D(1280, kernel_size=1, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.nn.relu6(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = BatchNormalization()(x)

    outputs = Conv2D(3, kernel_size=1, strides=1, padding="same", activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# モデルの作成
model = MobileNetV2_Segmentation()

# モデル全体を保存（アーキテクチャ、重み、オプティマイザの状態を含む）
model.save('mobilenetv2_segmentation.h5')
