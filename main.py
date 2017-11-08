from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from discriminator import _netD, _netG
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if args.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(args.imageSize),
                                   transforms.CenterCrop(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif args.dataset == 'lsun':
    dataset = dset.LSUN(db_path=args.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(args.imageSize),
                            transforms.CenterCrop(args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif args.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    testset = dset.CIFAR10(root=args.dataroot, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif args.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
                            transform=transforms.ToTensor())
assert dataset
train = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(args.workers))

ngpu = int(args.ngpu)
nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
if args.dataset == 'mnist':
    nc = 1
    nb_label = 10
else:
    nc = 3
    nb_label = 10






netG = _netG(ngpu, nz, ngf, nc, args)

if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
print(netG)


netD = _netD(ngpu, ndf, nc, nb_label, args)

if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)

d_criterion = nn.BCELoss()
c_criterion = nn.NLLLoss()

input = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)

noise = torch.FloatTensor(args.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(args.batchSize, nz, 1, 1).normal_(0, 1)
d_label = torch.FloatTensor(args.batchSize,1)
c_label = torch.LongTensor(args.batchSize,1)


real_label = 1
fake_label = 0

if args.cuda:
    netD.cuda()
    netG.cuda()
    netD = torch.nn.parallel.DataParallel(netD)
    d_criterion.cuda()
    c_criterion.cuda()
    input, d_label = input.cuda(), d_label.cuda()
    c_label = c_label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()



fixed_noisev = Variable(fixed_noise)
d_labelv = Variable(d_label)
c_labelv = Variable(c_label)
noisev = Variable(noise)
inputv = Variable(input)

fixed_noise_ = np.random.normal(0,1, (args.batchSize, nz))
random_label = np.random.randint(0, nb_label, args.batchSize)
print('fixed label:{}'.format(random_label))
random_onehot = np.zeros((args.batchSize, nb_label))
random_onehot[np.arange(args.batchSize), random_label] = 1
fixed_noise_[np.arange(args.batchSize), :nb_label] = random_onehot[np.arange(args.batchSize)]


fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise_ = fixed_noise_.resize_(args.batchSize,nz,1,1)
fixed_noise.copy_(fixed_noise_)

#costs
output

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))



def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels).cpu().sum()
    return correct, len(labels)
    
for epoch in range(args.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        img, label = data
        batch_size = img.size(0)
        if args.cuda:
            img = img.cuda()
        input.resize_(img.size()).copy_(img)
        d_label.resize_(batch_size,1).fill_(real_label)
        c_label.resize_(batch_size,1).copy_(label)


        inputv = Variable(input)
        d_labelv = Variable(d_label)
        c_labelv = Variable(c_label)
        # labelv = Variable(label)

        before, after = netD(inputv)
        d_errD_real = d_criterion(d_output, d_labelv)
        c_errD_real = c_criterion(c_output, c_labelv)
        errD_real = d_errD_real + c_errD_real
        errD_real.backward()
        D_x = d_output.data.mean()

        correct, length = test(c_output, c_label)

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)

        label = np.random.randint(0, nb_label, batch_size)
        noise_ = np.random.normal(0,1, (batch_size, nz))
        label_onehot = np.zeros((batch_size, nb_label))
        label_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]

        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, nz, 1, 1)
        noise.copy_(noise_)

        c_label.resize_(batch_size).copy_(torch.from_numpy(label))

        noisev = Variable(noise)
        fake = netG(noisev)

        d_label = d_label.fill_(fake_label)

        c_output, d_output = netD(fake.detach())
        d_errD_fake = d_criterion(d_output, d_label)
        c_errD_fake = c_criterion(c_output, c_label)
        errD_fake = d_errD_fake + c_errD_fake

        errD_fake.backward()
        D_G_z1 = d_output.data.mean()
        errD = d_errD_real + d_errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        d_labelv = Variable(d_label.fill_(real_label))  # fake labels are real for generator cost
        d_output, c_output = netD(fake)
        d_errG = d_criterion(d_output, d_labelv)
        c_errG = c_criterion(c_output, c_label)

        errG = d_errG + c_errG
        errG.backward()
        D_G_z2 = d_output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, args.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % args.outf,
                    normalize=True)
            fake = netG(fixed_noisev)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
