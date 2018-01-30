import torch.nn as nn
import torch
from collections import OrderedDict
from torch.autograd import Variable
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.05)
        m.bias.data.fill_(0)
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)

def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins

class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, args):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        noise_dim = (args.batchSize, nz)
        noise = torch.rand(noise_dim[0], noise_dim[1])
        self.ReLU = nn.ReLU(True)
        self.Tanh = nn.Tanh() 
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(nz, 100)
        self.linear2 = nn.Linear(100,500) #4 --> 8

        self.BatchNorm1 = nn.BatchNorm2d(500, momentum = 0.9)

        self.linear3 = nn.Linear(500,500)  # 8 --> 16
        self.BatchNorm2 = nn.BatchNorm2d(500, momentum = 0.9)

        self.linear4 = nn.Linear(500,28**2)
        # self.l2norm = torch.nn.functional.normalize()
        # self.l2norm = torch.nn.functional.normalize(28**2)
        # self.conv3 = nn.utils.weight_norm(nn.ConvTranspose2d(ngf * 4, nc, (4,4), 2, 1, bias=False), dim = 1) # 16 --> 32
        # self.BatchNorm3 = nn.BatchNorm2d(ngf * 2)

        # self.conv4 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1, bias=False)
        # self.BatchNorm4 = nn.BatchNorm2d(ngf * 1)

        # self.conv5 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1, bias=False)

        # self.apply(weights_init)
        self.apply(weights_init)

    def forward(self, input):
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        #     output = self.main(input)
        # return output

        # print(input.data.shape, 'i')
        # x = self.linear1(input)
        # print(x.data.shape, '0')
        # x = x.view(-1, self.ngf*16, 4, 4)
        # print(x.data.shape, '1')
        x = self.linear2(input)
        # print(x.data.shape, '2')
        x = self.BatchNorm1(x)
        # print(x.data.shape, '3')
        x = self.softplus(x)
        # print(x.data.shape, '4')

        x = self.linear3(x)
        # print(x.data.shape, '5')
        x = self.BatchNorm2(x)
        # print(x.data.shape, '6')
        x = self.softplus(x)
        # print(x.data.shape, '7')

        x = self.linear4(x)
        x = torch.nn.functional.normalize(x, dim=1, eps=1e-6)
        output = self.sigmoid(x)
        output = output.view(output.data.shape[0], 28,28)
        return output

class _netD(nn.Module):


    def __init__(self, ngpu, ndf, nc, nb_label, args):


        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1= nn.Dropout(p=0.2, inplace=False)
        self.args = args
        self.ndf = ndf

        self.linear1 = nn.Linear(28**2,1000)
        self.linear2 = nn.Linear(1000, 500)
        self.linear3 = nn.Linear(500, 250)
        self.linear4 = nn.Linear(250,250)
        self.linear5 = nn.Linear(250,250)
        self.aux_linear = nn.Linear(250,10)
        self.disc_linear = nn.Linear(250,1)
        # I am using weight = 0 for now which computes norm per output channel but experiment with setting it to none
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        # self.sigmoid = nn.Sigmoid()
        
        self.apply(weights_init)



    def forward(self, input):

        # print(input.data.shape, '0')
        input = input.view(input.data.shape[0], -1)
        x = gaussian(input, True, 0, 0.3)
        # print(x.data.shape, '1')

        x = self.linear1(x)
        # print(x.data.shape, '2')
        x = gaussian(x, True, 0, 0.5)
        # print(x.data.shape, '3')
        x = self.linear2(x)
        # print(x.data.shape, '3')
        x = gaussian(x, True, 0, 0.5)
        # print(x.data.shape, '4')
        x = self.linear3(x)
        # print(x.data.shape, '5')
        x = gaussian(x, True, 0, 0.5)
        # print(x.data.shape, '6')
        x = self.linear4(x)
        # print(x.data.shape, '7')
        x = gaussian(x, True, 0, 0.5)
        # print(x.data.shape, '8')
        last = self.linear5(x)
        # print(x.data.shape, '9')
        x = gaussian(last, True, 0, 0.5)
        # print(x.data.shape, '10')
        before = self.aux_linear(x)
        discriminate = self.disc_linear(x)
        after = self.softmax(before)
        return discriminate, before, after, last

# class _netD(nn.Module):

#     def __init__(self, ngpu, ndf, nc, nb_label, args):


#         super(_netD, self).__init__()
#         self.ngpu = ngpu
#         self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
#         self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
#         self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
#         self.BatchNorm2 = nn.BatchNorm2d(ndf * 2)
#         self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
#         self.BatchNorm3 = nn.BatchNorm2d(ndf * 4)
#         self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
#         self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
#         self.conv5 = nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0, bias=False)
#         self.disc_linear = nn.Linear(ndf * 1, 1)
#         self.aux_linear = nn.Linear(ndf * 1, nb_label)
#         self.softmax = nn.Softmax()
#         self.sigmoid = nn.Sigmoid()
#         self.ndf = ndf
#         self.apply(weights_init)



#     def forward(self, input):

#         x = self.conv1(input)
#         x = self.LeakyReLU(x)

#         x = self.conv2(x)
#         x = self.BatchNorm2(x)
#         x = self.LeakyReLU(x)

#         x = self.conv3(x)
#         x = self.BatchNorm3(x)
#         x = self.LeakyReLU(x)

#         x = self.conv4(x)
#         x = self.BatchNorm4(x)
#         x = self.LeakyReLU(x)

#         before = self.conv5(x)
#         before = before.view(-1, self.ndf * 1)

#         before2 = self.aux_linear(before)
#         after = self.softmax(before2)
#         discriminate = self.disc_linear(before)
#         discriminate = self.sigmoid(discriminate)
#         return discriminate, before, after