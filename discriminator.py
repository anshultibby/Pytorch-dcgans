import torch.nn as nn

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        noise_dim = (args.batchSize, 100)
        noise = torch.rand(sizes= noise_dim)
        self.main = nn.Sequential(OrderedDict([
            # #noise layer
            # ("noise", )
            # input is Z, going into a convolution
            ('conv1', nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False)),
            ('norm1', nn.BatchNorm2d(ngf * 8)),
            ('relu1', nn.ReLU(True)),
            # state size. (ngf*8) x 4 x 4
            ('conv2', nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)),
            ('norm2', nn.BatchNorm2d(ngf * 4)),
            ('relu2', nn.ReLU(True)),
            # state size. (ngf*4) x 8 x 8
            ('conv3', nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
            ('norm3', nn.BatchNorm2d(ngf * 2)),
            ('relu3', nn.ReLU(True)),
            # state size. (ngf*2) x 16 x 16
            ('conv4', nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False)),
            ('norm4', nn.BatchNorm2d(ngf)),
            ('relu4', nn.ReLU(True)),
            # state size. (ngf) x 32 x 32
            ('conv5', nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False)),
            ('out', nn.Tanh())
            # state size. (nc) x 64 x 64
        ]))
		self.apply(weights_init)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



class _netD(nn.Module):

    def __init__(self, ngpu, ndf, nc, nb_label):


        super(netD, self).__init__()
        self.ngpu = ngpu
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0, bias=False)
        self.disc_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, nb_label)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.ndf = ndf
        self.apply(weights_init)



    def forward(self, input):

        x = self.conv1(input)
        x = self.LeakyReLU(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.LeakyReLU(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.LeakyReLU(x)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.LeakyReLU(x)

        x = self.conv5(x)
        x = x.view(-1, self.ndf * 1)
        classify = self.aux_linear(x)
        classify = self.softmax(classify)
        discriminate = self.disc_linear(x)
        discriminate = self.sigmoid(discriminate)
        return classify, discriminate