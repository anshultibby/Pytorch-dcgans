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
import skimage.io as skio
import skimage.transform
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
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate, default=0.0003')
parser.add_argument('--beta1', type=float, default=0.7, help='beta1 for adam. default=0.7')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--split', type=float, default=0.1, help='what percentage of data to be considered unlabelled' )
parser.add_argument('--experiment', type=float, default=0, help='additional flag set to 1 to run experiments')
parser.add_argument('--specific', type=int, default=-1, help='additional flag to print out results for only one category')
parser.add_argument('--loop', type=int, default=0, help='additional flag to test code at multiple epochs')
parser.add_argument('--folder', default='', help='for extra folder location to load gen and dis from')
args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.dataset == 'ecg' or args.dataset == 'lvh':
    from discriminator_fisher import _netD, _netG
else:
    from discriminator import _netD, _netG

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
rng = np.random.RandomState(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True
# if args.cuda:
#     print("Using CUDA")
#     cutorch.setDevice(3)


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
    # testx = torch.from_numpy(testx)
    # testx = np.array(testx)
    # test =[]
    # for i, img in enumerate(testx):
    #     img = skimage.transform.resize(img,(110,110))
    #     test.append(img)
    # trainy = torch.from_numpy(np.argmax(trainy, axis=1))
    testy = torch.from_numpy(np.argmax(testy, axis=1))
    testx = torch.from_numpy(testx[:,10:,30:-20,:])
    dataset = torch.utils.data.TensorDataset(trainx, trainy)
    print(testx.size())
    testset = torch.utils.data.TensorDataset(testx, testy)

elif args.dataset == 'lvh':
    testx, testy = helper.read_dataset(args.dataroot + "/NEW_a4c_lvh_test.hdf5")

    newtestx = []
    newtesty = []
    for i,img in enumerate(testx):
        if args.specific != -1:
            # print(testy.size)
            if np.argmax([testy[i]], axis=1)[0] == args.specific:
                newtestx.append(skimage.transform.resize(img[:,:,0],(110,110)))
                newtesty.append(args.specific)
        else: 
            newtestx.append(skimage.transform.resize(img[:,:,0],(110,110)))
            newtesty.append(np.argmax([testy[i]], axis=1)[0])

    testx = torch.from_numpy(np.array(newtestx))
    testy = torch.from_numpy(np.array(newtesty))
    testset = torch.utils.data.TensorDataset(testx, testy)

    trainx1, trainy1 = helper.read_dataset(args.dataroot + "/NEW_a4c_lvh_train_1.hdf5")
    trainx2, trainy2 = helper.read_dataset(args.dataroot + "/NEW_a4c_lvh_train_2.hdf5")

    trainx = []
    unlabx = []
    trainy = []
    for i, img in enumerate(trainx1):
        trainx.append(skimage.transform.resize(img[:,:,0],(110,110)))
        trainy.append(trainy1[i])
    for i, img in enumerate(trainx2):
        trainx.append(skimage.transform.resize(img[:,:,0], (110,110)))
        trainy.append(trainy2[i])

    trainx = torch.from_numpy(np.array(trainx))
    labels = np.argmax(trainy, axis=1)
    trainy = torch.from_numpy(labels)
    dataset = torch.utils.data.TensorDataset(trainx, trainy)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=int(0))




    txs = []
    tys = []
    for data in dataloader:
        img, label = data
        txs.append(img)
        tys.append(label)


    x_lab = torch.stack(txs)
    y_lab = torch.stack(tys,dim=0)

    labset = torch.utils.data.TensorDataset(x_lab, y_lab)
    labloader = torch.utils.data.DataLoader(labset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(args.workers), drop_last = False)


    x_lab = []
    y_lab = []    
    for data in labloader:
        img, label = data
        x_lab.append(img)
        y_lab.append(label)
    

elif args.dataset == 'mnist':
    dataset = dset.MNIST(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                               ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #seems like torchvision is buggy for 1 channel normalize
                           ]))
elif args.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
                            transform=transforms.ToTensor())
# assert dataset

if args.dataset == 'mnist':
    nc = 1
    nb_label = 10
elif args.dataset == 'ecg':
    nc = 1
    nb_label = 15
elif args.dataset == 'lvh':
    nc = 1
    nb_label = 2
else:
    nc = 3
    nb_label = 10

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
#                                          shuffle=True, num_workers=int(0))

testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=int(0))

# txs = []
# tys = []
# for data in dataloader:
#     img, label = data
#     txs.append(img)
#     tys.append(label)


def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    # print(repr(pred.cpu().numpy()))
    # print(repr(labels.cpu().numpy()))

    correct = pred.eq(labels).cpu().sum()
    return correct, len(labels)
    
batch_size = args.batchSize

    
# txs2 = []
# tys2 = []
# for data in testloader:
#     img, label = data
#     txs2.append(img)
#     tys2.append(label)

# txs2 = np.array(txs2)
# tys2 = np.array(tys2)
# if args.experiment != 0:
#     cats = dict()
#     for i in range(nb_label):
#         cats[i] = []
#     for i,lab in enumerate(tys2):
#     #     print(cats)
#         lab = lab.numpy()[0]
#     #     print(lab)
#         if lab in cats:
#             a = cats[lab]
#             a.append(i)
#             cats[lab] = a
#         else:
#             a = [i]
#             cats[lab] = a

#     labelled_inds = []
#     unlabelled_inds = []        
#     for key in cats.keys():
#             # print(key)
#             if key == args.experiment:
#                 labelled_inds.extend(cats[key][:])
#                 a = cats[key][:]
#                 index = int(len(a))
#                 print(len(a), "len", args.experiment)
#                 unlabelled_inds.extend(a[:index])
#             # elif key == 8:
#             #     labelled_inds.extend(cats[key][:])
#             #     a = cats[key][:]
#             #     index  = int(1*len(a))
#             #     print(len(a), "len 8")
#             #     unlabelled_inds.extend(a[:index])
#         #Loading the testdata properly
#     testx = torch.squeeze(torch.stack(txs2[unlabelled_inds], dim=0))
#     testy = torch.stack(tys2[unlabelled_inds], dim=0)

# #Loading the testdata properly
# print(np.shape(txs2))
# testx = torch.squeeze(torch.stack(txs2, dim=0), dim=4)
# testy = torch.stack(tys2, dim=0)
# testset = torch.utils.data.TensorDataset(testx, testy)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize,
                                         shuffle=False, num_workers=int(0), drop_last = False)



#Making the labelled/unlabelled split
# splitinds = int(args.split*np.shape(txs)[0])
# x_unlab = torch.squeeze(torch.stack(txs, dim=0), dim=4)
# y_unlab = torch.stack(tys, dim=0)
# x_lab = torch.squeeze(torch.stack(txs[splitinds:],dim=0), dim=4)
# y_lab = torch.stack(tys[splitinds:],dim=0)

# labset = torch.utils.data.TensorDataset(x_lab, y_lab)
# unlabset = torch.utils.data.TensorDataset(x_unlab, y_unlab)



# labloader = torch.utils.data.DataLoader(labset, batch_size=args.batchSize,
#                                          shuffle=True, num_workers=int(args.workers), drop_last = True)

# unlabloader = torch.utils.data.DataLoader(unlabset, batch_size=args.batchSize,
#                                          shuffle=True, num_workers=int(args.workers), drop_last = True)
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
elif args.dataset == 'lvh':
    nc = 1
    nb_label = 2
else:
    nc = 3
    nb_label = 10







if args.loop != 0:

    
    fixed_noise = torch.FloatTensor(args.batchSize, nz, 1,1).normal_(0, 1)

    fixed_noisev = Variable(fixed_noise)

    fixed_noise_ = np.random.normal(0,1, (args.batchSize, nz))
    random_label = np.random.randint(0, nb_label, args.batchSize)
    print('fixed label:{}'.format(random_label))
    random_onehot = np.zeros((args.batchSize, nb_label))
    random_onehot[np.arange(args.batchSize), random_label] = 1
    fixed_noise_[np.arange(args.batchSize), :nb_label] = random_onehot[np.arange(args.batchSize)]


    fixed_noise_ = (torch.from_numpy(fixed_noise_))
    fixed_noise_ = fixed_noise_.resize_(args.batchSize,nz,1,1)
    fixed_noise.copy_(fixed_noise_)

    netDname = args.netD[-6:-4]
    netD = _netD(ngpu, ndf, nc, nb_label, args)
    input = torch.FloatTensor(args.batchSize, nc, args.imageSize, args.imageSize)

    d_label = torch.FloatTensor(args.batchSize,1)
    c_label = torch.LongTensor(args.batchSize,1)

    errors = list()
    epoch_arr = list()
    loss_arr = list()

    d_criterion = nn.BCEWithLogitsLoss()
    c_criterion = nn.CrossEntropyLoss()

    folder = args.folder
    for i in range(args.loop):

        epoch_arr.append(i)

        netG = _netG(ngpu, nz, ngf, nc, args)
        netGname = folder + "/netG_epoch_" + str(i) + ".pth"  
         # original saved file with DataParallel
        state_dict = torch.load(netGname)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        netG.load_state_dict(new_state_dict)
        # netG.load_state_dict(torch.load(args.netG))
        # print(netG)



        netD = _netD(ngpu, ndf, nc, nb_label, args)
        netDname = folder + "/netD_epoch_" + str(i) + ".pth"  
        # original saved file with DataParallel
        state_dict = torch.load(netDname)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        netD.load_state_dict(new_state_dict)
    # print(netD)




        if args.cuda:
            netD.cuda()
            netG.cuda()
            netD = torch.nn.parallel.DataParallel(netD, device_ids=[0])
            netG = torch.nn.parallel.DataParallel(netG, device_ids=[0])
            input, d_label = input.cuda(), d_label.cuda()
            c_label = c_label.cuda()



        d_labelv = Variable(d_label)
        c_labelv = Variable(c_label)
        inputv = Variable(input)
        noise = torch.FloatTensor(args.batchSize, nz)

        real_label = 1
        fake_label = 0
        netD.eval()
        #costs
        # output

        total_correct = 0
        total_length = 0
        loss = 0
        for data in testloader:
            noise = torch.FloatTensor(args.batchSize, nz)
            noise.resize_(batch_size, nz,1,1).normal_(0, 1)
            # noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)

            

            img, label = data
            # print(img.size())
            c_label.resize_(label.size()[0]).copy_(label)
            c_labelv = Variable(c_label)
            input.resize_(img.size()).copy_(img)
            inputv = Variable(input)


            discriminate, before, after, last  = netD(inputv)
            correct_unl, length_unl = test(after, c_label)

            img = x_lab[0]
            label = y_lab[0]
            c_label.resize_(label.size()[0]).copy_(label)
            c_labelv = Variable(c_label)
            input.resize_(img.size()).copy_(img)
            inputv = Variable(input)

            discriminate, before, after, last  = netD(inputv)
            discriminate3, before3, after3, last3 = netD(fake.detach())
            
            d_label.resize_(discriminate.size())
            d_label = d_label.fill_(real_label)
            d_labelv = Variable(d_label)

            

            loss_lab = c_criterion(before, c_labelv)
            loss_unl = d_criterion(discriminate, d_labelv)

            d_label.resize_(discriminate3.size())
            d_label = d_label.fill_(fake_label)
            d_labelv = Variable(d_label)
            loss_fake = d_criterion(discriminate3, d_labelv)


            loss = loss_lab + loss_unl + loss_fake
            total_correct += correct_unl
            total_length += length_unl


        vutils.save_image(img, '%s/real_samples.png' % args.outf, normalize=True)    
        fake = netG(fixed_noisev)
        # print(fake.data)
        vutils.save_image(fake.data,'%s/fake_samples_' +  args.folder + 'epoch_%03d.png' % (args.outf, i), normalize=True)

        error = 1.0 - float(total_correct)/float(total_length)
        errors.append(error)
        loss_arr.append(total_correct)

        # print(i)
        # c_errD_labelled = c_criterion(before, c_labelv)
        # print('Error rate: %.4f, total_length: %.4d, total_correct: %.4d '
        #           % (error, total_length, total_correct ))
    # print(errors)
    # print(loss_arr)
else:    

    netD = _netD(ngpu, ndf, nc, nb_label, args)

    if args.netD != '' and args.cuda:
        if args.experiment == 0:    
            # original saved file with DataParallel
            state_dict = torch.load(args.netD)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            netD.load_state_dict(new_state_dict)
            # print(netD)
        else:   
            netD1 = _netD(ngpu, ndf, nc, nb_label, args)

    netG = _netG(ngpu, nz, ngf, nc, args)
    
    if args.netG != '' and args.cuda:
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
        # print(netG)


input = torch.FloatTensor(args.batchSize, nc, args.imageSize, args.imageSize)

d_label = torch.FloatTensor(args.batchSize,1)
c_label = torch.LongTensor(args.batchSize,1)



if args.cuda:
    netD.cuda()
    netG.cuda()
    netD = torch.nn.parallel.DataParallel(netD, device_ids=[0])
    netG = torch.nn.parallel.DataParallel(netG, device_ids=[0])
    input, d_label = input.cuda(), d_label.cuda()
    c_label = c_label.cuda()



d_labelv = Variable(d_label)
c_labelv = Variable(c_label)
inputv = Variable(input)

netD.eval()
#costs
# output


total_correct = 0
total_length = 0
for data in testloader:

    img, label = data
    # print(img.size())
    c_label.resize_(label.size()[0]).copy_(label)
    input.resize_(img.size()).copy_(img)
    inputv = Variable(input)


    discriminate, before, after, last  = netD(inputv)
    correct_unl, length_unl = test(after, c_label)

    
    total_correct += correct_unl
    total_length += length_unl

error = 1.0 - float(total_correct)/float(total_length)
print('Error rate: %.4f, total_length: %.4d, total_correct: %.4d '
          % (error, total_length, total_correct ))


