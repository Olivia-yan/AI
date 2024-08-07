import sys
from collections import OrderedDict
from functools import partial

import torch.nn as nn

from modules import GlobalAvgPool2d, ResidualBlock
from model.attention.SEAttention import SEAttention

from .util import try_index


class ResNet(nn.Module):
    """Standard residual network

    Parameters
    ----------
    structure : list of int
        Number of residual blocks in each of the four modules of the network
    bottleneck : bool
        If `True` use "bottleneck" residual blocks with 3 convolutions, otherwise use standard blocks
    norm_act : callable
        Function to create normalization / activation Module
    classes : int
        If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
        of the network
    dilation : int or list of int
         List of dilation factors for the four modules of the network, or `1` to ignore dilation
    keep_outputs : bool
        If `True` output a list with the outputs of all modules
    """

    def __init__(
        self,
        structure,
        bottleneck,
        norm_act=nn.BatchNorm2d,
        classes=0,
        output_stride=16,
        keep_outputs=False
    ):
        super(ResNet, self).__init__()
        self.structure = structure
        self.bottleneck = bottleneck
        self.keep_outputs = keep_outputs

        if len(structure) != 4:
            raise ValueError("Expected a structure with four values")
        if output_stride != 8 and output_stride != 16:
            raise ValueError("Output stride must be 8 or 16")

        if output_stride == 16:
            dilation = [1, 1, 1, 2]  # dilated conv for last 3 blocks (9 layers)
        elif output_stride == 8:
            dilation = [1, 1, 2, 4]  # 23+3 blocks (78 layers)
        else:
            raise NotImplementedError

        self.dilation = dilation


        # Initial layers
        layers = [
            ("conv1", nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)), ("bn1", norm_act(64))
        ]
        if try_index(dilation, 0) == 1:
            layers.append(("pool1", nn.MaxPool2d(3, stride=2, padding=1)))
        self.mod1 = nn.Sequential(OrderedDict(layers))

        # Groups of residual blocks
        in_channels = 64
        if self.bottleneck:
            channels = (64, 64, 256)
        else:
            channels = (64, 64)
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                stride, dil = self._stride_dilation(dilation, mod_id, block_id)
                blocks.append(
                    (
                        "block%d" % (block_id + 1),
                        ResidualBlock(
                            in_channels,
                            channels,
                            norm_act=norm_act,
                            stride=stride,
                            dilation=dil,
                            last=block_id == num - 1
                        )
                    )
                )

                # Update channels and p_keep
                in_channels = channels[-1]

            # Create module
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

            # Double the number of channels for the next module
            channels = [c * 2 for c in channels]

        self.out_channels = in_channels

        # Pooling and predictor
        if classes != 0:
            self.classifier = nn.Sequential(
                OrderedDict(
                    [("avg_pool", GlobalAvgPool2d()), ("fc", nn.Linear(in_channels, classes))]
                )
            )

    @staticmethod
    def _stride_dilation(dilation, mod_id, block_id):
        d = try_index(dilation, mod_id)
        s = 2 if d == 1 and block_id == 0 and mod_id > 0 else 1
        return s, d

    def forward(self, x):
        outs = []
        attentions = []

        branch1_x, branch2_x = [], []
        # print('resnetx',x.shape)#[6, 3, 512, 512]
        x = self.mod1(x)
        #attentions.append(x)
        # print('self.mod1=', x.shape)#[6, 64, 128, 128]
        # v100上跑的结果。4得到35.24           16得到34.17     32得到34.45         64得到37.64和37.14
        se = SEAttention(x.device, channel=64, reduction=64)  # 可取1, 2, 4, 8, 16, 32, 64。langchao上跑的结果。4得到34.68           4得到35.74         8得到34.71            64得到35.57和35.79
        x1 = se(x)
        # print(x1.shape)
        outs.append(x1)


        x, xb1, xb2, att = self.mod2(x1)#如果这里处理的是x1，v100上的结果是35.41，langchao结果是35.98
        # print('self.mod2,x=', x.shape)#[6, 256, 128, 128]
        # print('self.mod2,xb1=', xb1.shape)#[6, 64, 128, 128]
        # print('self.mod2,xb2=', xb2.shape)#[6, 64, 128, 128]
        # print('self.mod2,att=', att.shape)#[6, 256, 128, 128]
        # transform_net2 = TransformNet2(x.device)
        # transform_net2 = transform_net2.half()
        # # 通过网络传递 x 来得到与 x1 相同尺寸的输出
        # x01 = transform_net2(x)
        #
        # v100上跑的结果。32得到34点61     64得到36点03      ，128得到33.93   256得到32.8
        se2 = SEAttention(x.device, channel=256,reduction=256)  # 可取1, 2, 4, 8, 16, 32, 64, 128, 256。langchao上跑的结果。 256得到31.73(mod2(x1))
        x2 = se2(x)
        # # print('x01', x01.shape)
        # #这里
        # x2 = x01 + x2
        # print('afterx2', x2.shape)
        attentions.append(att)
        outs.append(x2)
        branch1_x.append(xb1)
        branch2_x.append(xb2)

        x, xb1, xb2, att = self.mod3(x2)
        se3 = SEAttention(x.device, channel=512,reduction=512)  # 可取1, 2, 4, 8, 16, 32, 64, 128, 256，512。
        x3 = se3(x)
        # 这里
        # print('self.mod3,x=', x.shape)#[6, 512, 64, 64]
        # print('self.mod3,xb1=', xb1.shape)#[6, 128, 64, 64]
        # print('self.mod3,xb2=', xb2.shape)#[6, 128, 64, 64]
        # print('self.mod3,att=', att.shape)#[6, 512, 64, 64]
        # transform_net3 = TransformNet3(x.device)
        # transform_net3 = transform_net3.half()
        # # 通过网络传递 x 来得到与 x1 相同尺寸的输出
        # x03 = transform_net3(x)
        # # print('x03', x03.shape)
        # transform_net10 = TransformNet10(x.device)
        # transform_net10 = transform_net10.half()
        # # 通过网络传递 x 来得到与 x1 相同尺寸的输出
        # x10 = transform_net10(x1)
        # # print('x10', x10.shape)
        # # breakss
        # x3 = x03 + x10 + x3
        attentions.append(att)
        outs.append(x3)
        branch1_x.append(xb1)
        branch2_x.append(xb2)

        x, xb1, xb2, att = self.mod4(x3)
        se4 = SEAttention(x.device, channel=1024,reduction=1024)  # 可取1, 2, 4, 8, 16, 32, 64, 128, 256，512，1024。
        x4 = se4(x)
        # print('self.mod4,x=', x.shape)#[6, 1024, 32, 32]
        # print('self.mod4,xb1=', xb1.shape)#[6, 256, 32, 32]
        # print('self.mod4,xb2=', xb2.shape)#[6, 256, 32, 32]
        # print('self.mod4,att=', att.shape)#[6, 1024, 32, 32]
        # transform_net4 = TransformNet4(x.device)
        # transform_net4 = transform_net4.half()
        # # 通过网络传递 x 来得到与 x1 相同尺寸的输出
        # x04 = transform_net4(x)
        # print('x04', x04.shape)
        # transform_net14 = TransformNet14(x.device)
        # transform_net14 = transform_net14.half()
        # # 通过网络传递 x 来得到与 x1 相同尺寸的输出
        # x14 = transform_net14(x1)
        # print('x14', x14.shape)
        # transform_net24 = TransformNet24(x.device)
        # transform_net24 = transform_net24.half()
        # # 通过网络传递 x 来得到与 x1 相同尺寸的输出
        # x24 = transform_net24(x2)
        # print('x24', x24.shape)

        # x4 = x04 + x14 + x24 + x4
        attentions.append(att)
        outs.append(x4)
        branch1_x.append(xb1)
        branch2_x.append(xb2)

        x, xb1, xb2, att = self.mod5(x4)
        # print('self.mod5,x=', x.shape)#[6, 2048, 32, 32]
        # print('self.mod5,xb1=', xb1.shape)#[6, 512, 32, 32]
        # print('self.mod5,xb2=', xb2.shape)#[6, 512, 32, 32]
        # print('self.mod5,att=', att.shape)#[6, 2048, 32, 32]
        #5个全64得到35.89，reduction=channel得到36.28
        se5 = SEAttention(x.device, channel=2048,reduction=2048)  # 可取1, 2, 4, 8, 16, 32, 64, 128, 256，512，1024，2048。
        x5 = se5(x)
        x=x5
        attentions.append(att)
        outs.append(x5)
        branch1_x.append(xb1)
        branch2_x.append(xb2)


        if hasattr(self, "classifier"):
            outs.append(self.classifier(outs[-1]))

        if self.keep_outputs:
            return outs, attentions, branch1_x, branch2_x
        else:
            return outs[-1], attentions, branch1_x, branch2_x


_NETS = {
    "18": {
        "structure": [2, 2, 2, 2],
        "bottleneck": False
    },
    "34": {
        "structure": [3, 4, 6, 3],
        "bottleneck": False
    },
    "50": {
        "structure": [3, 4, 6, 3],
        "bottleneck": True
    },
    "101": {
        "structure": [3, 4, 23, 3],
        "bottleneck": True
    },
    "152": {
        "structure": [3, 8, 36, 3],
        "bottleneck": True
    },
}

__all__ = []
for name, params in _NETS.items():
    net_name = "net_resnet" + name
    setattr(sys.modules[__name__], net_name, partial(ResNet, **params))
    __all__.append(net_name)
