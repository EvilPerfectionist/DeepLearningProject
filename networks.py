import torch.nn.functional as F
import torch
import torch.nn as nn
from spectral_norm import SpectralNorm
import numpy as np

def weights_init_normal(m):
    """Initialize the weights of a module with Gaussian distribution."""
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        # The init call will throw an AttributeError for Conv2d layers with spectral norm, because
        # they do not have a 'weight' attribute. We can skip the initialization for these layers.
        # These were already initalized in a different manner during their construction.
        try:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        except AttributeError:
            pass
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

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
        elif normalize == 'spectral':
            # conv + spectralnorm
            model = [
                SpectralNorm(nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding))
            ]

        model.append(activation_fn)

        if dropout > 0:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #print(x.shape)
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

    def __init__(self, normalization_type=None):
        super(Generator, self).__init__()
        self.norm = normalization_type

        self.down1 = ConvBlock(1, 64, normalize=self.norm, kernel_size=4, stride=1, padding=0, dropout=0)
        self.down2 = ConvBlock(64, 64, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down3 = ConvBlock(64, 128, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down4 = ConvBlock(128, 256, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down5 = ConvBlock(256, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down6 = ConvBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down7 = ConvBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)

        self.up1 = TransConvBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.up2 = TransConvBlock(1024, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.up3 = TransConvBlock(1024, 256, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.up4 = TransConvBlock(512, 128, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.up5 = TransConvBlock(256, 64, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.up6 = TransConvBlock(128, 64, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.final = ConvBlock(
            128, 2, normalize=None, kernel_size=1, stride=1, padding=0, dropout=0, activation_fn=nn.Tanh()
        )

    def forward(self, x):
        #print(x.dtype)
        x = F.interpolate(x, size=(131, 131), mode='bilinear', align_corners=True)
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

    def __init__(self, input_nc, output_nc, ngf, color_dim = 313):
        super(unet_generator, self).__init__()

        self.e1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.e2 = unet_encoder_block(ngf, ngf * 2)
        self.e3 = unet_encoder_block(ngf * 2, ngf * 4)
        self.e4 = unet_encoder_block(ngf * 4, ngf * 8)
        self.e5 = unet_encoder_block(ngf * 8, ngf * 8)
        self.e6 = unet_encoder_block(ngf * 8, ngf * 8, norm = None)

        self.d1 = unet_decoder_block(ngf * 8, ngf * 8)
        self.d2 = unet_decoder_block(ngf * 8 * 2, ngf * 8, drop_out = None)
        self.d3 = unet_decoder_block(ngf * 8 * 2, ngf * 4, drop_out = None)
        self.d4 = unet_decoder_block(ngf * 4 * 2, ngf * 2, drop_out = None)
        self.d5 = unet_decoder_block(ngf * 2 * 2, ngf, drop_out = None)
        self.d6 = unet_decoder_block(ngf * 2, output_nc, norm = None, drop_out = None)
        self.tanh = nn.Tanh()

        self.layers = [self.e1, self.e2, self.e3, self.e4, self.e5, self.e6,
                 self.d1, self.d2, self.d3, self.d4, self.d5, self.d6]

        self.mlp = MLP(color_dim, self.get_num_adain_params(self.layers), self.get_num_adain_params(self.layers), 3)


    def forward(self, x, color_feat):

        ### AdaIn params
        adain_params = self.mlp(color_feat)
        self.assign_adain_params(adain_params, self.layers)

        ### Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        ### Decoder
        d1_ = self.d1(e6)
        d1 = torch.cat([d1_, e5], dim = 1)

        d2_ = self.d2(d1)
        d2 = torch.cat([d2_, e4], dim = 1)

        d3_ = self.d3(d2)
        d3 = torch.cat([d3_, e3], dim = 1)

        d4_ = self.d4(d3)
        d4 = torch.cat([d4_, e2], dim = 1)

        d5_ = self.d5(d4)
        d5 = torch.cat([d5_, e1], dim = 1)

        d6 = self.d6(d5)

        output = self.tanh(d6)

        return output

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


class unet_encoder_block(nn.Module):

    def __init__(self, input_nc, output_nc, ks = 4, stride = 2, padding = 1, norm = 'adain', act = nn.LeakyReLU(inplace = True, negative_slope = 0.2)):
        super(unet_encoder_block, self).__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, ks, stride, padding)
        m = [act, self.conv]

        if norm == 'adain':
            m.append(AdaptiveInstanceNorm2d(output_nc))

        self.body = nn.Sequential(*m)

    def forward(self, x):
        print(x.shape)
        return self.body(x)

class unet_decoder_block(nn.Module):

    def __init__(self, input_nc, output_nc, ks = 4, stride = 2, padding = 1, norm = 'adain', act = nn.ReLU(inplace = True), drop_out = 0.5):
        super(unet_decoder_block, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_nc, output_nc, ks, stride, padding)
        m = [act, self.deconv]

        if norm == 'adain':
            m.append(AdaptiveInstanceNorm2d(output_nc))

        if drop_out is not None:
            m.append(nn.Dropout(drop_out))

        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


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
    def __init__(self, img_dim, feature_dim, imsize, conv_dim=64, repeat_num=5):
        super(Discriminator2, self).__init__()


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
        x = torch.cat([ab_img, l_img, color_feat], dim = 1)
        batch_size = x.size(0)
        h = self.main(x)
        out = self.conv1(h)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out
