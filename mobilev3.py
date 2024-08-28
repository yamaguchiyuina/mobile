from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Multiply, Dense, DepthwiseConv2D
import tensorflow.keras.backend as K
import tensorflow as tf

def hard_sigmoid(x):
    return tf.nn.relu6(x + 3.) * (1. / 6.)

def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])

def SE_Block(x, in_channels, r=0.25):
    out = GlobalAveragePooling2D()(x)
    out = Dense(int(in_channels * r))(out)
    out = Activation(hard_sigmoid)(out)
    out = Dense(in_channels, activation='sigmoid')(out)
    out = Multiply()([out, x])
    return out

def mobilenetv3_block(x, exp_size, out_channels, kernel, SE, NL, s, r=0.25):
    activation = hard_swish if NL == 'HS' else 'relu'

    out = Conv2D(filters=exp_size, kernel_size=(1, 1), padding='same')(x)
    out = BatchNormalization()(out)
    out = Activation(activation)(out)

    out = DepthwiseConv2D(kernel_size=kernel, strides=s, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation(activation)(out)

    out = Conv2D(filters=out_channels, kernel_size=(1, 1), padding='same')(out)
    out = BatchNormalization()(out)

    if SE:
        out = SE_Block(out, in_channels=out_channels, r=r)

    if K.int_shape(out) == K.int_shape(x):
        out = Add()([out, x])

    return out

def MobileNetV3_Small(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(16, kernel_size=3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(hard_swish)(x)

    x = mobilenetv3_block(x, exp_size=16,  out_channels=16, kernel=3, SE=True,  NL='RE', s=2)
    x = mobilenetv3_block(x, exp_size=72,  out_channels=24, kernel=3, SE=False, NL='RE', s=2)
    x = mobilenetv3_block(x, exp_size=88,  out_channels=24, kernel=3, SE=False, NL='RE', s=1)
    x = mobilenetv3_block(x, exp_size=96,  out_channels=40, kernel=5, SE=True,  NL='HS', s=2)
    x = mobilenetv3_block(x, exp_size=240, out_channels=40, kernel=5, SE=True,  NL='HS', s=1)
    x = mobilenetv3_block(x, exp_size=240, out_channels=40, kernel=5, SE=True,  NL='HS', s=1)
    x = mobilenetv3_block(x, exp_size=120, out_channels=48, kernel=5, SE=True,  NL='HS', s=1)
    x = mobilenetv3_block(x, exp_size=144, out_channels=48, kernel=5, SE=True,  NL='HS', s=1)
    x = mobilenetv3_block(x, exp_size=288, out_channels=96, kernel=5, SE=True,  NL='HS', s=2)
    x = mobilenetv3_block(x, exp_size=576, out_channels=96, kernel=5, SE=True,  NL='HS', s=1)
    x = mobilenetv3_block(x, exp_size=576, out_channels=96, kernel=5, SE=True,  NL='HS', s=1)

    x = Conv2D(576, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(hard_swish)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024)(x)
    x = Activation(hard_sigmoid)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# モデルの作成
model = MobileNetV3_Small()

# モデル全体を保存
model.save('mobilenetv3_small.h5')
