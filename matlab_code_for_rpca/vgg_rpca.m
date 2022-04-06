dataset = 'cifar100'; 
depth = 19;
rank_level = 6;  % changed according to compression rate (1~6)

network = ['vgg',num2str(depth)];
save_dir = [network, '_', num2str(rank_level), '/']; 
if rank_level == 1
    lambda = [0.13, 0.04, 0.05, 0.028, 0.035, 0.02, 0.02, 0.02, 0.02, 0.014, 0.013, 0.013, 0.014, 0.014, 0.013, 0.013, 0.04, 0.04, 0.046];
    if depth == 19
        lambda(10) = 0.013; % avoiding non convergence
    end
elseif rank_level == 2
    lambda = [0.12, 0.035, 0.04, 0.025, 0.03, 0.019, 0.017, 0.017, 0.017, 0.013, 0.012, 0.012, 0.013, 0.013, 0.012, 0.012, 0.035, 0.035, 0.045];
elseif rank_level == 3
    lambda = [0.11, 0.032, 0.035, 0.023, 0.025, 0.018, 0.016, 0.016, 0.016, 0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011, 0.03, 0.03, 0.044];
elseif rank_level == 4
    lambda = [0.1, 0.03, 0.03, 0.022, 0.022, 0.017, 0.015, 0.015, 0.015, 0.011, 0.01, 0.01, 0.011, 0.011, 0.01, 0.01, 0.028, 0.028, 0.042];
elseif rank_level == 5
    lambda = [0.09, 0.028, 0.028, 0.021, 0.02, 0.016, 0.014, 0.013, 0.014, 0.01, 0.009, 0.009, 0.01, 0.01, 0.009, 0.009, 0.027, 0.027, 0.041];
elseif rank_level == 6
    lambda = [0.08, 0.026, 0.025, 0.02, 0.018, 0.015, 0.013, 0.012, 0.013, 0.009, 0.008, 0.008, 0.009, 0.009, 0.008, 0.008, 0.025, 0.025, 0.04];
end
if strcmp(network, 'vgg11')
    conv1 = features_0_weight; bias1 = features_0_bias; conv3 = features_3_weight; bias3 = features_3_bias;
    conv5 = features_6_weight; bias5 = features_6_bias; conv6 = features_8_weight; bias6 = features_8_bias;
    conv9 = features_11_weight; bias9 = features_11_bias; conv10 = features_13_weight; bias10 = features_13_bias;
    conv13 = features_16_weight; bias13 = features_16_bias; conv14 = features_18_weight; bias14 = features_18_bias;
elseif strcmp(network, 'vgg13')
    conv1 = features_0_weight; bias1 = features_0_bias; conv2 = features_2_weight; bias2 = features_2_bias;
    conv3 = features_5_weight; bias3 = features_5_bias; conv4 = features_7_weight; bias4 = features_7_bias;
    conv5 = features_10_weight; bias5 = features_10_bias; conv6 = features_12_weight; bias6 = features_12_bias;
    conv9 = features_15_weight; bias9 = features_15_bias; conv10 = features_17_weight; bias10 = features_17_bias;
    conv13 = features_20_weight; bias13 = features_20_bias; conv14 = features_22_weight; bias14 = features_22_bias;
elseif strcmp(network, 'vgg16')
    conv1 = features_0_weight; bias1 = features_0_bias; conv2 = features_2_weight; bias2 = features_2_bias;
    conv3 = features_5_weight; bias3 = features_5_bias; conv4 = features_7_weight; bias4 = features_7_bias;
    conv5 = features_10_weight; bias5 = features_10_bias; conv6 = features_12_weight; bias6 = features_12_bias;
    conv7 = features_14_weight; bias7 = features_14_bias; conv9 = features_17_weight; bias9 = features_17_bias; 
    conv10 = features_19_weight; bias10 = features_19_bias; conv11 = features_21_weight; bias11 = features_21_bias;
    conv13 = features_24_weight; bias13 = features_24_bias; conv14 = features_26_weight; bias14 = features_26_bias;
    conv15 = features_28_weight; bias15 = features_28_bias;
elseif strcmp(network, 'vgg19')
    conv1 = features_0_weight; bias1 = features_0_bias; conv2 = features_2_weight; bias2 = features_2_bias;
    conv3 = features_5_weight; bias3 = features_5_bias; conv4 = features_7_weight; bias4 = features_7_bias;
    conv5 = features_10_weight; bias5 = features_10_bias; conv6 = features_12_weight; bias6 = features_12_bias;
    conv7 = features_14_weight; bias7 = features_14_bias; conv8 = features_16_weight; bias8 = features_16_bias;
    conv9 = features_19_weight; bias9 = features_19_bias; conv10 = features_21_weight; bias10 = features_21_bias; 
    conv11 = features_23_weight; bias11 = features_23_bias; conv12 = features_25_weight; bias12 = features_25_bias;
    conv13 = features_28_weight; bias13 = features_28_bias; conv14 = features_30_weight; bias14 = features_30_bias;
    conv15 = features_32_weight; bias15 = features_32_bias; conv16 = features_34_weight; bias16 = features_34_bias;
end

params = 0;
flops = 0;
r1 = conv_decompose(conv1, lambda(1), ['mat_weights_',dataset,'/',save_dir,'conv1_1'], bias1);
params = 91 * r1 + 64 + params;
flops = 93184 * r1 + flops;
if depth > 11
    r2 = conv_decompose(conv2, lambda(2), ['mat_weights_',dataset,'/',save_dir,'conv1_2'], bias2);
    params = 640 * r2 + 64 + params;
    flops = 655360 * r2 + flops;
end
r3 = conv_decompose(conv3, lambda(3), ['mat_weights_',dataset,'/',save_dir,'conv2_1'], bias3);
params = 704 * r3 + 128 + params;
flops = 180224 * r3 + flops;
if depth > 11
    r4 = conv_decompose(conv4, lambda(4), ['mat_weights_',dataset,'/',save_dir,'conv2_2'], bias4);
    params = 1280 * r4 + 128 + params;
    flops = 327680 * r4 + flops;
end
r5 = conv_decompose(conv5, lambda(5), ['mat_weights_',dataset,'/',save_dir,'conv3_1'], bias5);
params = 1408 * r5 + 256 + params;
flops = 90112 * r5 + flops;
r6 = conv_decompose(conv6, lambda(6), ['mat_weights_',dataset,'/',save_dir,'conv3_2'], bias6);
params = 2560 * r6 + 256 + params;
flops = 163840 * r6 + flops;
if depth > 13
    r7 = conv_decompose(conv7, lambda(7), ['mat_weights_',dataset,'/',save_dir,'conv3_3'], bias7);
    params = 2560 * r7 + 256 + params;
    flops = 163840 * r7 + flops;
end
if depth > 16
    r8 = conv_decompose(conv8, lambda(8), ['mat_weights_',dataset,'/',save_dir,'conv3_4'], bias8);
    params = 2560 * r8 + 256 + params;
    flops = 163840 * r8 + flops;
end
r9 = conv_decompose(conv9, lambda(9), ['mat_weights_',dataset,'/',save_dir,'conv4_1'], bias9);
params = 2816 * r9 + 512 + params;
flops = 45056 * r9 + flops;
r10 = conv_decompose(conv10, lambda(10), ['mat_weights_',dataset,'/',save_dir,'conv4_2'], bias10);
params = 5120 * r10 + 512 + params;
flops = 81920 * r10 + flops;
if depth > 13
    r11 = conv_decompose(conv11, lambda(11), ['mat_weights_',dataset,'/',save_dir,'conv4_3'], bias11);
    params = 5120 * r11 + 512 + params;
    flops = 81920 * r11 + flops;
end
if depth > 16
    r12 = conv_decompose(conv12, lambda(12), ['mat_weights_',dataset,'/',save_dir,'conv4_4'], bias12);
    params = 5120 * r12 + 512 + params;
    flops = 81920 * r12 + flops;
end
r13 = conv_decompose(conv13, lambda(13), ['mat_weights_',dataset,'/',save_dir,'conv5_1'], bias13);
params = 5120 * r13 + 512 + params;
flops = 20480 * r13 + flops;
r14 = conv_decompose(conv14, lambda(14), ['mat_weights_',dataset,'/',save_dir,'conv5_2'], bias14);
params = 5120 * r14 + 512 + params;
flops = 20480 * r14 + flops;
if depth > 13
    r15 = conv_decompose(conv15, lambda(15), ['mat_weights_',dataset,'/',save_dir,'conv5_3'], bias15);
    params = 5120 * r15 + 512 + params;
    flops = 20480 * r15 + flops;
end
if depth > 16
    r16 = conv_decompose(conv16, lambda(16), ['mat_weights_',dataset,'/',save_dir,'conv5_4'], bias16);
    params = 5120 * r16 + 512 + params;
    flops = 20480 * r16 + flops;
end
r17 = fc_decompose(classifier_1_weight, classifier_1_bias, lambda(17), ['mat_weights_',dataset,'/',save_dir,'fc1']);
params = 1025 * r17 + 512 + params;
flops = 1024 * r17 + flops;
r18 = fc_decompose(classifier_4_weight, classifier_4_bias, lambda(18), ['mat_weights_',dataset,'/',save_dir,'fc2']);
params = 1025 * r18 + 512 + params;
flops = 1024 * r18 + flops;
r19 = fc_decompose(classifier_6_weight, classifier_6_bias, lambda(19), ['mat_weights_',dataset,'/',save_dir,'fc3']);
params = 523 * r19 + 10 + params;
flops = 522 * r19 + flops;

disp(['params_low_rank=',num2str(params)])
disp(['FLOPs_low_rank=',num2str(flops)])

    