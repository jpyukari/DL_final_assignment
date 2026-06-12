import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    ResNet の basic block
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        コンストラクタ．

        Parameters
        ----------
        in_channles: int
            入力のチャネル数
        out_channels:
            出力のチャネル数
        stride: int
            ストライド
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        順伝播処理

        Parameters
        ----------
        x: torch.Tensor
            ブロックへの入力

        Returns
        -------
        out: torch.Tensor
            ブロックへの出力
        """
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out
    

class BottleneckBlock(nn.Module):
    """
    ResNet の bottleneck block
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        """
        コンストラクタ．

        Parameters
        ----------
        in_channles: int
            入力のチャネル数
        out_channels:
            出力のチャネル数
        stride: int
            ストライド
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        """
        順伝播処理

        Parameters
        ----------
        x: torch.Tensor
            ブロックへの入力

        Returns
        -------
        out: torch.Tensor
            ブロックへの出力
        """
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet の実装
    """
    def __init__(self, block, layers):
        """
        コンストラクタ．

        Parameters
        ----------
        block: torch.nn.Module
            利用するブロックのクラス (BasicBlock / BottleneckBlock)
        layers: list
            各ブロックの層数
        """
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        """
        同じ構成を繰り返す部分を生成する．

        Parameters
        ----------
        block: torch.nn.Module
            利用するブロックのクラス (BasicBlock / BottleneckBlock)
        blocks: int
            層数
        out_channels: int
            出力のチャネル数
        stride: int
            ストライド

        Returns
        -------
        layers: torch.nn.ModuleList
            生成した層
        """
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        順伝播処理

        Parameters
        ----------
        x: torch.Tensor
            入力データ

        Returns
        -------
        x: torch.Tensor
            ResNet によって生成される特徴量
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    

def ResNet18():
    """
    ResNet18 を生成する関数．
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    """
    ResNet50 を生成する関数．
    """
    return ResNet(BottleneckBlock, [3, 4, 6, 3])
