# RopeNet-
Implementation for RopeNet 
Matlab version: R2018a

Before running:

Manually create a folder that corresponds to the directory name 'save_dir'.
Adjust lambda of RPCA according to the paper. For LeNet5, adjust lam_conv1, lam_conv2, lam_fc1 and lam_fc2.
For ResNet, adjust lam_conv1, lam_linear, lam_layer1, lam_layer2 and lam_layer3. For VGG, just adjust rank_level.
Run LeNet_rpca, resnet_rpca and vgg_rpca to conduct RPCA on LeNet, ResNet and VGG, respectively.

python version: 3.7
torch version:1.9

Run svd_rpca_cifar to evaluate on ResNet and VGG.
Note that for --decompose and --fine_tune, please manually modify line143, 148, 149, 152, 153 to decide weights loading directory and define Rope-Net.

options:
--save-mat: train the network and save weights in mat form.
--decompose: evaluate on a non-finetuned Rope-Net.
--fine_tune: evaluate on a finetuned Rope-Net.
--sp: sparsity for the sparse branch of Rope-Net, entered in decimal form.
--dataset: cifar10 or cifar100. Default cifar10,

Run svd_rpca_LeNet to evaluate on LeNet5.
Note that for --decompose and --fine_tune, please manually modify line172 to decide weights loading directory and modify line193 or 1ine199 to decide sparsity
for the sparse branch of Rope-Net.

options:
--save-mat: train the network and save weights in mat form.
--decompose: evaluate on a non-finetuned Rope-Net.
--fine_tune: evaluate on a finetuned Rope-Net.
