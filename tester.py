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
import helper
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=40, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--niter', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate, default=0.0003')
parser.add_argument('--beta1', type=float, default=0.7, help='beta1 for adam. default=0.7')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--split', type=float, default=0.1, help='what percentage of data to be considered unlabelled' )

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
rng = np.random.RandomState(args.manualSeed)
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
elif args.dataset == 'mnist':
    dataset = dset.MNIST(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                               ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #seems like torchvision is buggy for 1 channel normalize
                           ]))
elif args.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
                            transform=transforms.ToTensor())
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=True, num_workers=int(0))

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(0))

txs = []
tys = []
for data in dataloader:
    img, label = data
    txs.append(img)
    tys.append(label)
    
# testx = []
# for data in testset:
#     img = data
#     testx.append(img)


#Making the labelled/unlabelled split
splitinds = int(args.split*np.shape(txs)[0])
x_unlab = torch.squeeze(torch.stack(txs, dim=0))
y_unlab = torch.stack(tys, dim=0)
x_lab = torch.squeeze(torch.stack(txs[splitinds:],dim=0))
y_lab = torch.stack(tys[splitinds:],dim=0)

labset = torch.utils.data.TensorDataset(x_lab, y_lab)
unlabset = torch.utils.data.TensorDataset(x_unlab, y_unlab)

labloader = torch.utils.data.DataLoader(labset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(args.workers), drop_last = True)

unlabloader = torch.utils.data.DataLoader(unlabset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(args.workers), drop_last = True)
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
     # original saved file with DataParallel
    state_dict = torch.load(args.netG)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    netG.load_state_dict(new_state_dict)
    # netG.load_state_dict(torch.load(args.netG))
print(netG)


netD = _netD(ngpu, ndf, nc, nb_label, args)

if args.netD != '' and args.cuda:    
    # original saved file with DataParallel
    state_dict = torch.load(args.netD)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    netD.load_state_dict(new_state_dict)
print(netD)



input = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)

d_label = torch.FloatTensor(args.batchSize,1)
c_label = torch.LongTensor(args.batchSize,1)



if args.cuda:
    netD.cuda()
    netG.cuda()
    netD = torch.nn.parallel.DataParallel(netD, device_ids=[0, 1])
    netG = torch.nn.parallel.DataParallel(netG, device_ids=[0, 1])
    input, d_label = input.cuda(), d_label.cuda()
    c_label = c_label.cuda()



d_labelv = Variable(d_label)
c_labelv = Variable(c_label)
inputv = Variable(input)

netD.eval()
#costs
# output


def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels).cpu().sum()
    return correct, len(labels)
    

    
total_correct = 0
total_length = 0
for data in testloader:

    img, label = data
    c_label.resize_(args.batchSize).copy_(label)
    input.resize_(img.size()).copy_(img)
    inputv = Variable(input)


    discriminate, before, after, last  = netD(inputv)
    correct_unl, length_unl = test(after, c_label)

    total_correct += correct_unl
    total_length += length_unl

error = 1.0 - float(total_correct)/float(total_length)
print('Error rate: %.4f, total_length: %.4d, total_correct: %.4d '
          % (error, total_length, total_correct ))

