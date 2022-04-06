Matlab version: R2018a

Before running:
1. Manually create a folder that corresponds to the directory name 'save_dir'.
2. Adjust lambda of RPCA according to the paper. For LeNet5, adjust lam_conv1, lam_conv2, lam_fc1 and lam_fc2. 
   For ResNet, adjust lam_conv1, lam_linear, lam_layer1, lam_layer2 and lam_layer3. For VGG, just adjust rank_level.

Run LeNet_rpca, resnet_rpca and vgg_rpca to conduct RPCA on LeNet, ResNet and VGG, respectively.