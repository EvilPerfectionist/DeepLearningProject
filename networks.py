import torch.nn.functional as F
import torch
import torch.nn as nn
from spectral_norm import SpectralNorm


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
