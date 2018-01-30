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
from discriminator_heart import _netD, _netG
import numpy as np 
import helper

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=40, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate, default=0.0003')
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
elif args.dataset == 'ecg':
    trainx, trainy = helper.read_dataset(args.dataroot + "/training_data4.hdf5")
    testx, testy = helper.read_dataset(args.dataroot + "/test_data4.hdf5")
    trainx = torch.from_numpy(trainx[:,10:,30:-20,:])
    testx = torch.from_numpy(testx[:,10:,30:-20,:])

    trainy = torch.from_numpy(np.argmax(trainy, axis=1))
    testy = torch.from_numpy(np.argmax(testy, axis=1))
    dataset = torch.utils.data.TensorDataset(trainx, trainy)
    testset = torch.utils.data.TensorDataset(testx, testy)
    
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
x_unlab = torch.squeeze(torch.stack(txs, dim=0), dim=4)
# print(x_unlab.size())
# input()
y_unlab = torch.stack(tys, dim=0)
x_lab = torch.squeeze(torch.stack(txs[splitinds:],dim=0), dim =4)
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
elif args.dataset == 'ecg':
    nc = 1
    nb_label = 15
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

d_criterion = nn.BCEWithLogitsLoss()
c_criterion = nn.CrossEntropyLoss()
gen_criterion = nn.MSELoss()

input = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
input2 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)

noise = torch.FloatTensor(args.batchSize, nz)
fixed_noise = torch.FloatTensor(args.batchSize, nz, 1,1).normal_(0, 1)
d_label = torch.FloatTensor(args.batchSize,1)
c_label = torch.LongTensor(args.batchSize,1)


real_label = 1
fake_label = 0

if args.cuda:
    netD.cuda()
    netG.cuda()
    netD = torch.nn.parallel.DataParallel(netD, device_ids=[0, 3])
    netG = torch.nn.parallel.DataParallel(netG, device_ids=[0, 3])
    d_criterion.cuda()
    c_criterion.cuda()
    gen_criterion.cuda()
    input, d_label = input.cuda(), d_label.cuda()
    input2 = input2.cuda()
    c_label = c_label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()



fixed_noisev = Variable(fixed_noise)
d_labelv = Variable(d_label)
c_labelv = Variable(c_label)
noisev = Variable(noise)
inputv = Variable(input)
input2v = Variable(input2)

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
# output

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=50, gamma=0.2)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=50, gamma=0.2)


def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels).cpu().sum()
    return correct, len(labels)
    
def feature_loss(X1, X2):

    m1 = torch.mean(X1, 0)
    m2 = torch.mean(X2, 0)
    loss = torch.mean(torch.abs(m1 - m2))
    return loss    

loss_g = list()
loss_d = list()
for epoch in range(args.niter):
    # schedulerG.step()
    # schedulerD.step()

    x_lab = [] 
    y_lab = []
    for data in labloader:
        img, label = data
        x_lab.append(img)
        y_lab.append(label)
    num_labs = len(x_lab)


    x_unlab = []
    y_unlab = []    
    for data in unlabloader:
        img, label = data
        x_unlab.append(img)
        y_unlab.append(label)
    
    total_correct_unl = 0
    total_length_unl = 0

    total_correct_lab = 0
    total_length_lab = 0

    for i, img2 in enumerate(x_unlab):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with labelled
        netD.zero_grad()
        i2 = i % num_labs
        batch_size = args.batchSize
        label = y_lab[i2]

        unl_label = y_unlab[i]
        img = x_lab[i2]

        if args.cuda:
            img = img.cuda()

        input.resize_(img.size()).copy_(img)
        input2.resize_(img.size()).copy_(img2)



        d_label.resize_(batch_size,1).fill_(real_label)


        c_label.resize_(batch_size).copy_(label)

   
        inputv = Variable(input)
        input2v = Variable(input2)
        d_labelv = Variable(d_label)
        c_labelv = Variable(c_label)
        labelv = Variable(label)

        discriminate, before, after, last = netD(inputv)

        c_errD_labelled = c_criterion(before, c_labelv)

        errD_real = c_errD_labelled
        errD_real.backward()
        
        input.resize_(img.size()).copy_(img2)
        inputv = Variable(input)


        discriminate2, before2, after2, last2 = netD(inputv)

        l_lab = Variable(before.data[torch.from_numpy(np.arange(batch_size)).cuda(),c_label])

        l_unl = helper.log_sum_exp(before2)



        D_x = 0.5*discriminate.data.mean() + 0.5*discriminate2.data.mean()

        correct, length = test(after, c_label)
        c_label.resize_(batch_size).copy_(unl_label)

        correct_unl, length_unl = test(after2, c_label)


        total_correct_unl += correct_unl
        total_length_unl += length_unl

        total_correct_lab += correct
        total_length_lab += length

        # train with fake
        noise.resize_(batch_size, nz,1,1).normal_(0, 1)

        # noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)

        label = np.random.randint(0, nb_label, batch_size)
        noise_ = np.random.normal(0,1, (batch_size, nz))
        label_onehot = np.zeros((batch_size, nb_label))
        label_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]

        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, nz, 1, 1)
        noise.copy_(noise_)


        noisev = Variable(noise)
        fake = netG(noisev)


        d_label = d_label.fill_(real_label)
        d_labelv = Variable(d_label)
        loss_unl = d_criterion(discriminate2, d_labelv)
        loss_unl.backward()



        d_label = d_label.fill_(fake_label)
        d_labelv = Variable(d_label)
        discriminate3, before3, after3, last3 = netD(fake.detach())
        loss_fake = d_criterion(discriminate3, d_labelv)
        loss_fake.backward()

        # z_exp_unl = helper.log_sum_exp(before2)
        # z_exp_fake = helper.log_sum_exp(before3)

        # l_gen = helper.log_sum_exp(before3)
        # softplus = torch.nn.Softplus()
  
        errD_fake = loss_unl + loss_fake


        # errD_fake.backward(retain_graph = True)
        D_G_z1 = discriminate3.data.mean()
        errD = errD_real + errD_fake
        # errD.backward(retain_graph = True)
        optimizerD.step()


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # print(input2v.data)
        # input.resize_(img.size()).copy_(img)
        # inputv = Variable(input)
        # discriminate, before, after = netD(inputv)

        fake = netG(noisev)
        discriminate, before, after, last  = netD(fake)
        discriminate2, before2, after2, last2  = netD(inputv.detach())
        # print(before)
        # gen_loss = feature_loss(before, before2)

        d_labelv = Variable(d_label.fill_(real_label))  # fake labels are real for generator cost
        # # d_output, c_output = netD(fake)
        # d_errG = d_criterion(discriminate2, d_labelv)
        # c_errG = c_criterion(c_output, c_label)

        # m1 = torch.mean(last, 0)
        # m2 = torch.mean(last2, 0)
        # # print(m1 - m2)
        # loss_gen = torch.mean(torch.abs(m1-m2))
        last2v = Variable(last2.data, requires_grad = False)
        gen_loss = gen_criterion(torch.mean(last, 0), torch.mean(last2v,0))
        # gen_loss = d_criterion(discriminate, d_labelv)

        # errG = loss_gen
        errG = gen_loss
        errG.backward()
        D_G_z2 = discriminate.data.mean()
        optimizerG.step()


        loss_d.append(errD.detach().data)
        loss_g.append(errG.detach().data)
        print('[%d/%d][%d/%d] Loss_D: %.4f Fake_Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Correct_l: %.4f Correct_unl: %.4f Length: %.4f'
              % (epoch, args.niter, i, len(x_unlab),
                 errD.data[0], errD_fake.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, correct, correct_unl, length))
        if i % 100 == 0:
            vutils.save_image(img,
                    '%s/real_samples.png' % args.outf,
                    normalize=True)
            fake = netG(fixed_noisev)
            # print(fake.data)
            vutils.save_image(fake.data,'%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                    normalize=True)

    lab_training_error = 1.0 - float(total_correct_lab)/float(total_length_lab)
    unlab_training_error = 1.0 - float(total_correct_unl)/float(total_length_unl)        
    print('[%d/%d] Labelled_Training_Error: %.4f Unlabelled_Training_Error: %.4f'
              % (epoch, args.niter,lab_training_error, unlab_training_error))
    # do checkpointing
    d = np.array(loss_d)
    g = np.array(loss_g)
    np.save('%s/LossG_epoch_%d.npy' % (args.outf, epoch), g)
    np.save('%s/LossD_epoch_%d.npy' % (args.outf, epoch), d)
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
