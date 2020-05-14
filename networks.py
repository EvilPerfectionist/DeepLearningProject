import torch.nn.functional as F
import torch
import torch.nn as nn
from spectral_norm import SpectralNorm
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize=None, kernel_size=4, stride=2,
                 padding=1, dropout=0, activation_fn=nn.LeakyReLU(0.2)):
        super(ConvBlock, self).__init__()
        model = [nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding)]

        if normalize == 'batch':
            # + batchnorm
            model.append(nn.BatchNorm2d(out_size))
        elif normalize == 'instance':
            # + instancenorm
            model.append(nn.InstanceNorm2d(out_size))
        elif normalize == 'adain':
            # + Adaptive Instance Normalization
            model.append(AdaptiveInstanceNorm2d(out_size))

        model.append(activation_fn)

        if dropout > 0:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        print("encode" + str(x.shape))
        x = self.model(x)
        return x

class TransConvBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize=None, kernel_size=4, stride=2,
                 padding=1, dropout=0.0, activation_fn=nn.ReLU()):
        super(TransConvBlock, self).__init__()
        model = [nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding)]

        if normalize == 'batch':
            # add batch norm
            model.append(nn.BatchNorm2d(out_size))
        elif normalize == 'instance':
            # add instance norm
            model.append(nn.InstanceNorm2d(out_size))
        elif normalize == 'adain':
            # + Adaptive Instance Normalization
            model.append(AdaptiveInstanceNorm2d(out_size))

        model.append(activation_fn)

        if dropout > 0:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x, skip_input):
        #print(x.shape, skip_input.shape)
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    """Generator architecture."""

    def __init__(self, normalization_type=None, conv_dim = 64, repeat_num = 2):
        super(Generator, self).__init__()
        self.norm = normalization_type

        self.encode_layers = []
        self.encode_layers.append(ConvBlock(1, conv_dim, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0))

        curr_dim = conv_dim
        for i in range(1, 4):
            self.encode_layers.append(ConvBlock(curr_dim, curr_dim * 2, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0))
            curr_dim = curr_dim * 2
        for j in range(repeat_num):
            self.encode_layers.append(ConvBlock(curr_dim, curr_dim, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0))

        self.decode_layers = []
        self.decode_layers.append(TransConvBlock(curr_dim, curr_dim, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0))

        for k in range(repeat_num - 1):
            self.decode_layers.append(TransConvBlock(curr_dim * 2, curr_dim, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0))
        for h in range(1, 4):
            curr_dim = curr_dim // 2
            self.decode_layers.append(TransConvBlock(curr_dim * 4, curr_dim, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0))

        self.final = ConvBlock(curr_dim * 2, 2, normalize=None, kernel_size=1, stride=1, padding=0, dropout=0, activation_fn=nn.Tanh())

        # self.down1 = ConvBlock(1, 64, normalize=self.norm, kernel_size=4, stride=1, padding=0, dropout=0)
        # self.down2 = ConvBlock(64, 64, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        # self.down3 = ConvBlock(64, 128, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        # self.down4 = ConvBlock(128, 256, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        # self.down5 = ConvBlock(256, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        # self.down6 = ConvBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        # self.down7 = ConvBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        #
        # self.up1 = TransConvBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        # self.up2 = TransConvBlock(1024, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        # self.up3 = TransConvBlock(1024, 256, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        # self.up4 = TransConvBlock(512, 128, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        # self.up5 = TransConvBlock(256, 64, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        # self.up6 = TransConvBlock(128, 64, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        # self.final = ConvBlock(
        #     128, 2, normalize=None, kernel_size=1, stride=1, padding=0, dropout=0, activation_fn=nn.Tanh()
        #)

    def forward(self, x):
        #print(x.dtype)
        #x = F.interpolate(x, size=(131, 131), mode='bilinear', align_corners=True)
        encoded_results = []
        for encode_layer in range(self.encode_layers):
            encoded_result = encode_layer(x)
            encoded_results.append(encoded_result)

        flag = len(encoded_results)
        for decode_layer in range(self.decode_layers):
            if flag == len(encoded_results):
                decoded_result = decode_layers(encoded_results[-1], encoded_results[-2])
                encoded_results.pop(-1)
                encoded_results.pop(-1)
            else:
                decoded_result = decode_layer(decoded_result, encoded_results[-1])
                encoded_results.pop(-1)

        x = self.final(decoded_result)
        # d1 = self.down1(x)
        # d2 = self.down2(d1)
        # d3 = self.down3(d2)
        # d4 = self.down4(d3)
        # d5 = self.down5(d4)
        # d6 = self.down6(d5)
        # d7 = self.down7(d6)

        # u1 = self.up1(d7, d6)
        # u2 = self.up2(u1, d5)
        # u3 = self.up3(u2, d4)
        # u4 = self.up4(u3, d3)
        # u5 = self.up5(u4, d2)
        # u6 = self.up6(u5, d1)
        #
        # x = self.final(u6)
        return x

class Discriminator(nn.Module):
    """Discriminator architecture."""

    def __init__(self, normalization_type):
        super(Discriminator, self).__init__()
        self.norm = normalization_type

        self.down1 = ConvBlock(3, 64, normalize=None, kernel_size=4, stride=1, padding=0, dropout=0)
        self.down2 = ConvBlock(64, 128, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down3 = ConvBlock(128, 256, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down4 = ConvBlock(256, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down5 = ConvBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down6 = ConvBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.final = ConvBlock(
            512, 1, normalize=None, kernel_size=4, stride=1, padding=0, dropout=0, activation_fn=nn.Sigmoid()
        )

    def forward(self, x):
        x = F.interpolate(x, size=(131, 131), mode='bilinear', align_corners=True)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        x = self.final(d6)
        x = x.view(x.size()[0], -1)
        #print(x.shape)
        return x

class unet_generator(nn.Module):

    def __init__(self, color_dim = 313, img_size = 128):
        super(unet_generator, self).__init__()

        self.norm = 'adain'

        self.down1 = ConvBlock(1, 64, normalize = self.norm, kernel_size = 4, stride = 1, padding = 0, dropout = 0)
        self.down2 = ConvBlock(64, 64, normalize = self.norm, kernel_size = 4, stride = 2, padding = 1, dropout = 0)
        self.down3 = ConvBlock(64, 128, normalize = self.norm, kernel_size = 4, stride = 2, padding = 1, dropout = 0)
        self.down4 = ConvBlock(128, 256, normalize = self.norm, kernel_size = 4, stride = 2, padding = 1, dropout = 0)
        self.down5 = ConvBlock(256, 512, normalize = self.norm, kernel_size = 4, stride = 2, padding = 1, dropout = 0)
        self.down6 = ConvBlock(512, 512, normalize = self.norm, kernel_size = 4, stride = 2, padding = 1, dropout = 0)
        self.down7 = ConvBlock(512, 512, normalize = None, kernel_size = 4, stride = 2, padding = 1, dropout = 0)

        self.up1 = TransConvBlock(512, 512, normalize = self.norm, kernel_size = 4, stride = 2, padding = 1, dropout = 0)
        self.up2 = TransConvBlock(1024, 512, normalize = self.norm, kernel_size = 4, stride = 2, padding = 1, dropout = 0)
        self.up3 = TransConvBlock(1024, 256, normalize = self.norm, kernel_size = 4, stride = 2, padding = 1, dropout = 0)
        self.up4 = TransConvBlock(512, 128, normalize = self.norm, kernel_size = 4, stride = 2, padding = 1, dropout = 0)
        self.up5 = TransConvBlock(256, 64, normalize = self.norm, kernel_size = 4, stride = 2, padding = 1, dropout = 0)
        self.up6 = TransConvBlock(128, 64, normalize = self.norm, kernel_size = 4, stride = 2, padding = 1, dropout = 0)
        self.final = ConvBlock(
            128, 2, normalize = None, kernel_size = 1, stride = 1, padding = 0, dropout = 0, activation_fn = nn.Tanh()
        )

        self.layers = [self.down1, self.down2, self.down3, self.down4, self.down5, self.down6, self.down7,
                 self.up1, self.up2, self.up3, self.up4, self.up5, self.up6, self.final]

        self.mlp = MLP(color_dim, self.get_num_adain_params(self.layers), self.get_num_adain_params(self.layers), 3)
        self.img_size = img_size

    def forward(self, x, color_feat):

        ### AdaIn params
        adain_params = self.mlp(color_feat)
        self.assign_adain_params(adain_params, self.layers)

        x = F.interpolate(x, size=(self.img_size + 3, self.img_size + 3), mode='bilinear', align_corners=True)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        x = self.final(u6)
        return x

    def get_num_adain_params(self, _module):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for model in _module:
            for m in model.modules():
                if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                    num_adain_params += 2*m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params, _module):
        # assign the adain_params to the AdaIN layers in model
        for model in _module:
            for m in model.modules():
                if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                    mean = adain_params[:, :m.num_features]
                    std = adain_params[:, m.num_features:2*m.num_features]
                    m.bias = mean.contiguous().view(-1)
                    m.weight = std.contiguous().view(-1)
                    if adain_params.size(1) > 2*m.num_features:
                        adain_params = adain_params[:, 2*m.num_features:]

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, act = nn.ReLU(inplace = True)):

        super(MLP, self).__init__()
        self.model = []

        self.model.append(nn.Linear(input_dim, dim))
        self.model.append(act)

        for i in range(n_blk - 2):
            self.model.append(nn.Linear(dim, dim))
            self.model.append(act)

        self.model.append(nn.Linear(dim, output_dim))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class Discriminator2(nn.Module):
    def __init__(self, feature_dim, imsize, conv_dim=64, repeat_num=5):
        super(Discriminator2, self).__init__()

        img_dim = 3
        input_dim = img_dim + feature_dim

        layers = []
        layers.append(nn.Conv2d(input_dim, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(imsize / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(k_size * k_size * curr_dim),
            nn.Linear(k_size * k_size * curr_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, ab_img, l_img, color_feat):
        print(ab_img.shape)
        print(l_img.shape)
        x = torch.cat([ab_img, l_img, color_feat], dim = 1)
        batch_size = x.size(0)
        h = self.main(x)
        out = self.conv1(h)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out
