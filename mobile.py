class MobileNetV2_Segmentation(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, kernel_size=3, strides=2, padding="same")
        self.bn1 = BatchNormalization()
        self.relu1 = tf.nn.relu6

        self.bottleneck1 = bottleneck(in_channels=32,  t=1, c=16,  n=1, s=1)
        self.bottleneck2 = bottleneck(in_channels=16,  t=6, c=24,  n=2, s=2)
        self.bottleneck3 = bottleneck(in_channels=24,  t=6, c=32,  n=3, s=2)
        self.bottleneck4 = bottleneck(in_channels=32,  t=6, c=64,  n=4, s=2)
        self.bottleneck5 = bottleneck(in_channels=64,  t=6, c=96,  n=3, s=1)
        self.bottleneck6 = bottleneck(in_channels=96,  t=6, c=160, n=3, s=2)
        self.bottleneck7 = bottleneck(in_channels=160, t=6, c=320, n=1, s=1)
        self.conv2 = Conv2D(1280, kernel_size=1, strides=1, padding="same")
        self.bn2 = BatchNormalization()
        self.relu2 = tf.nn.relu6

        # 出力形状を合わせるためにアップサンプリング層を追加
        self.upsample1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv3 = Conv2D(256, kernel_size=3, strides=1, padding="same", activation='relu')
        self.bn3 = BatchNormalization()

        self.upsample2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv4 = Conv2D(128, kernel_size=3, strides=1, padding="same", activation='relu')
        self.bn4 = BatchNormalization()

        self.upsample3 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv5 = Conv2D(64, kernel_size=3, strides=1, padding="same", activation='relu')
        self.bn5 = BatchNormalization()

        self.upsample4 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv6 = Conv2D(32, kernel_size=3, strides=1, padding="same", activation='relu')
        self.bn6 = BatchNormalization()

        self.final_conv = Conv2D(3, kernel_size=1, strides=1, padding="same", activation='sigmoid')  # 3チャンネルの出力

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.upsample1(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.upsample2(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = self.upsample3(x)
        x = self.conv5(x)
        x = self.bn5(x)

        x = self.upsample4(x)
        x = self.conv6(x)
        x = self.bn6(x)

        x = self.final_conv(x)
        return x
