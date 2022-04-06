lam_conv1 = 0.16;
lam_conv2 = 0.055;
lam_fc1 = 0.045;
lam_fc2 = 0.12;
save_dir='/rl3/';
conv_decompose(conv1_weight, lam_conv1, ['mat_weights_mnist',save_dir,'conv1']);
conv_decompose(conv2_weight, lam_conv2, ['mat_weights_mnist',save_dir,'conv2']);
fc_decompose(fc1_weight, fc1_bias, lam_fc1, ['mat_weights_mnist',save_dir,'fc1']);
fc_decompose(fc2_weight, fc2_bias, lam_fc2, ['mat_weights_mnist',save_dir,'fc2']);

