# UNetなどのモデルを使用する例
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def build_unet_model(input_shape):
    inputs = Input(shape=input_shape)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    # ... (中間層の省略)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c1)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# 各パッチのサイズに応じてモデルを構築
unet_models = [build_unet_model(patch.shape) for patch in patches]

# 各パッチに対してモデルを適用
for i, (model, patch) in enumerate(zip(unet_models, patches)):
    prediction = model.predict(np.expand_dims(patch, axis=0))
    print(f"Prediction for Patch {i}: {prediction.shape}")
