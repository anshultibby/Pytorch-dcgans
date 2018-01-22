# Pytorch-dcgans
Install the latest version of torch, Cudnn 5.1 and Cuda 8.0 for the code to run.

The cifar_meatching code is from Goodfellow's github and is used as reference for writing the pytorch code.

On the Cifar 10 dataset this model achieved an error rate of 19% after running for 200 epochs for a 10-90 labelled/unlabelled split. This gives us 5000 labelled samples. The paper on improved GANs training reported an error rate of 16-20% for a split containing 4000 labelled samples and 16-19% for a split containing 8000 labelled samples so this is well within the benchmark results.
