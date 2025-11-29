# residual_layer_with_pool + dropout
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, pool_csl):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(0.1)
        )

        self.shortcut = None
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c)
            )
        self.act = nn.GELU()
        self.pool = pool_csl(2)

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x) if self.shortcut else x
        out = self.act(out)
        return self.pool(out)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            ResidualBlock(3, 32, nn.MaxPool2d),
            ResidualBlock(32, 64, nn.MaxPool2d),
            ResidualBlock(64, 128, nn.MaxPool2d),
            ResidualBlock(128, 256, nn.AvgPool2d)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=10)
model.to(device); #send model to available proccessing unit (GPU or CPU)

# residual_layer_with_pool+pre-activation
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, pool_csl):
        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.GELU(),
            nn.Conv2d(in_c, out_c, 3, padding=1),

            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
        )

        self.shortcut = nn.Identity()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c)
            )
        self.pool = pool_csl(2)

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        return self.pool(out)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            ResidualBlock(3, 32, nn.MaxPool2d),
            ResidualBlock(32, 64, nn.MaxPool2d),
            ResidualBlock(64, 128, nn.MaxPool2d),
            ResidualBlock(128, 256, nn.AvgPool2d)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# residual + se
class SEBlock(nn.Module): # squeezes & excites
    def __init__(self, c, r=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, pool_csl):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )

        self.shortcut = None
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c)
            )
        self.act = nn.GELU()
        self.pool = pool_csl(2)
        self.se = SEBlock(out_c)

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out += self.shortcut(x) if self.shortcut else x
        out = self.act(out)
        return self.pool(out)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            ResidualBlock(3, 32, nn.MaxPool2d),
            ResidualBlock(32, 64, nn.MaxPool2d),
            ResidualBlock(64, 128, nn.MaxPool2d),
            ResidualBlock(128, 256, nn.AvgPool2d)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# residual layer with pool
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, pool_csl):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )

        self.shortcut = None
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c)
            )
        self.act = nn.GELU()
        self.pool = pool_csl(2)

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x) if self.shortcut else x
        out = self.act(out)
        return self.pool(out)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            ResidualBlock(3, 32, nn.MaxPool2d),
            ResidualBlock(32, 64, nn.MaxPool2d),
            ResidualBlock(64, 128, nn.MaxPool2d),
            ResidualBlock(128, 256, nn.AvgPool2d)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# double channels in the middle + gelu
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def get_conv_layer(in_c, out_c, pool_cls):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c*2, 3, padding=1),
                nn.BatchNorm2d(out_c*2),
                nn.GELU(),
                
                nn.Conv2d(out_c*2, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
                
                pool_cls(2)
            )
        self.features = nn.Sequential(
            get_conv_layer(3, 32, nn.MaxPool2d),
            get_conv_layer(32, 64, nn.MaxPool2d),
            get_conv_layer(64, 128, nn.MaxPool2d),
            get_conv_layer(128, 256, nn.AvgPool2d),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=10)
model.to(device); #send model to available proccessing unit (GPU or CPU)

# double channels in the middle
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def get_conv_layer(in_c, out_c, pool_cls):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c*2, 3, padding=1),
                nn.BatchNorm2d(out_c*2),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(out_c*2, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                
                pool_cls(2)
            )
        self.features = nn.Sequential(
            get_conv_layer(3, 32, nn.MaxPool2d),
            get_conv_layer(32, 64, nn.MaxPool2d),
            # nn.Dropout2d(0.1),
            get_conv_layer(64, 128, nn.MaxPool2d),
            get_conv_layer(128, 256, nn.AvgPool2d),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=10)
model.to(device); #send model to available proccessing unit (GPU or CPU)

# double conv per pool
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def get_conv_layer(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                
                nn.MaxPool2d(2),
                # nn.Dropout2d(0.1)
            )
        self.features = nn.Sequential(
            get_conv_layer(3, 32),
            get_conv_layer(32, 64),
            get_conv_layer(64, 128),
            get_conv_layer(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=10)
model.to(device); #send model to available proccessing unit (GPU or CPU)

# original
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=10)
model.to(device); #send model to available proccessing unit (GPU or CPU)