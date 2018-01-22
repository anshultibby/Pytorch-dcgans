import torch.nn as nn
import torch
from collections import OrderedDict
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, args):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        noise_dim = (args.batchSize, nz)
        noise = torch.rand(noise_dim[0], noise_dim[1])
        self.ReLU = nn.ReLU(True)
        self.Tanh = nn.Tanh() 
        # self.linear = nn.Linear(nz, 512*4*4)
        self.conv1 = nn.ConvTranspose2d(nz, ngf*8, (4,4), 1, 0, bias=False) #1 --> 4
        self.BatchNorm1 = nn.BatchNorm2d(ngf * 8, momentum = 0.1)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, (4,4), 2, 1, bias=False) # 4 --> 8
        self.BatchNorm2 = nn.BatchNorm2d(ngf * 4, momentum = 0.1)

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf*2, (4,4), 2, 1, bias=False) # 8 --> 16
        self.BatchNorm3 = nn.BatchNorm2d(ngf * 2)
        
        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf*2, (3,3), 1, 1, bias=False) # 16 --> 16
        self.BatchNorm4 = nn.BatchNorm2d(ngf * 2)

        self.conv5 = nn.ConvTranspose2d(ngf * 2, ngf*2, (3,3), 1, 1, bias=False) # 16 --> 16
        self.BatchNorm5 = nn.BatchNorm2d(ngf * 2)

        self.conv6 = nn.ConvTranspose2d(ngf * 2, nc, (4,4), 2, 1, bias=False) # 16 --> 32
 
        self.apply(weights_init)

    def forward(self, input):
        
        # print(input.data.shape, '1')
        x = self.conv1(input)
        # print(x.data.shape, '2')
        x = self.BatchNorm1(x)
        # print(x.data.shape, '3')
        x = self.ReLU(x)
        # print(x.data.shape, '4')

        x = self.conv2(x)
        # print(x.data.shape, '5')
        x = self.BatchNorm2(x)
        # print(x.data.shape, '6')
        x = self.ReLU(x)
        # print(x.data.shape, '7')

        x = self.conv3(x)
        # print(x.data.shape, '8')
        x = self.BatchNorm3(x)
        x = self.ReLU(x)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.ReLU(x)

        x = self.conv5(x)
        x = self.BatchNorm5(x)
        x = self.ReLU(x)
        x = self.conv6(x)

        output = self.Tanh(x)
        return output

class _netD(nn.Module):

    def __init__(self, ngpu, ndf, nc, nb_label, args):


        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1= nn.Dropout(p=0.2, inplace=False)
        self.args = args
        self.ndf = ndf

        self.conv1 = nn.Conv2d(nc, 3*ndf, kernel_size = (3,3), stride = 1, padding = 1, bias=False)
        # I am using weight = 0 for now which computes norm per output channel but experiment with setting it to none
        self.conv2 = nn.Conv2d(3*ndf, 3*ndf, kernel_size = (3,3), stride = 1, padding = 1, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ndf * 3, momentum = 0.1)

        self.conv3 = nn.Conv2d(3*ndf, 3*ndf, kernel_size = (3,3), stride = 2, padding = 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ndf * 3, momentum = 0.1)
        self.dropout2 = nn.Dropout2d(p=0.5, inplace=False)
        self.conv4 = nn.Conv2d(3*ndf, 6*ndf, kernel_size = (3,3), stride = 1, padding = 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ndf * 6, momentum = 0.1)
        self.conv5 = nn.Conv2d(6*ndf, 6*ndf, kernel_size = (3,3), stride = 1, padding = 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ndf * 6, momentum = 0.1)
        self.conv6 = nn.Conv2d(6*ndf, 6*ndf, kernel_size = (3,3), stride = 2, padding = 1, bias=False)
        self.BatchNorm5 = nn.BatchNorm2d(ndf * 6, momentum = 0.1)
        self.dropout3 = nn.Dropout2d(p=0.5, inplace=False)
        
        self.conv7 = nn.Conv2d(6*ndf, 12*ndf, kernel_size = (3,3), stride = 1, padding = 0, bias=False)
        self.BatchNorm6 = nn.BatchNorm2d(ndf * 12, momentum = 0.1)
        
        self.conv8 = nn.Conv2d(12*ndf, 12*ndf, kernel_size = (3,3), stride = 1, padding = 0, bias=False)
        self.BatchNorm7 = nn.BatchNorm2d(ndf * 12, momentum = 0.1)
        
        self.conv9 = nn.Conv2d(12*ndf, 12*ndf, kernel_size = (1,1), stride = 1, padding = 0, bias=False)
        self.BatchNorm8 = nn.BatchNorm2d(ndf * 12, momentum = 0.1)
        self.network_linear1 = nn.utils.weight_norm(nn.Linear(6*ndf*6*6, 6*6*ndf))
        self.dropout4 = nn.Dropout2d(p=0.5, inplace=False)

        self.aux_linear = nn.Linear(6144, nb_label)
        self.disc_linear = nn.Linear(6144 , 1)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        # self.sigmoid = nn.Sigmoid()
        
        self.apply(weights_init)



    def forward(self, input):

        x = self.dropout1(input)
        # print(x.data.shape, '0')
        # print(input.data.shape, 'i')
        x = self.conv1(x)
        # print(x.data.shape, '1')
        x = self.LeakyReLU(x)
        # print(x.data.shape, '2')
        x = self.conv2(x)
        x = self.BatchNorm1(x)
        # print(x.data.shape, '3')
        x = self.LeakyReLU(x)
        # print(x.data.shape, '4')
        x = self.conv3(x)
        x = self.BatchNorm2(x)

        # print(x.data.shape, '5')
        x = self.LeakyReLU(x)
        # print(x.data.shape, '6')

        x = self.dropout2(x)
        # print(x.data.shape, '7')
        x = self.conv4(x)
        x = self.BatchNorm3(x)

        # print(x.data.shape, '8')
        x = self.LeakyReLU(x)
        # print(x.data.shape, '9')
        x = self.conv5(x)
        x = self.BatchNorm4(x)

        # print(x.data.shape, '10')
        x = self.LeakyReLU(x)
        # print(x.data.shape, '11')
        x = self.conv6(x)
        x = self.BatchNorm5(x)

        # print(x.data.shape, '12')
        x = self.LeakyReLU(x)
        # print(x.data.shape, '13')
        x = self.dropout3(x)
        # print(x.data.shape, '14')
        x = self.conv7(x)
        x = self.BatchNorm6(x)
        x = self.LeakyReLU(x)
        # print(x.data.shape, '15')
        x = self.conv8(x)
        x = self.BatchNorm7(x)
        # print(x.data.shape, '15')

        x = self.LeakyReLU(x)

        x = self.conv9(x)
        x = self.BatchNorm8(x)
        # print(x.data.shape, '15')
        x = self.LeakyReLU(x)

        before = self.dropout4(x)

        # before = self.dropout5(x)

        before = before.view(before.data.shape[0], -1)

        pool = before
        # print(pool.data.shape, '21')
        # raw_input()
        before2 = self.aux_linear(before)
        # print(before2.data.shape, '21')
        # raw_input()
        discriminate = self.disc_linear(before)
        discriminate = self.sigmoid(discriminate)

        after = self.softmax(before2)
        return discriminate, before2, after, pool 
