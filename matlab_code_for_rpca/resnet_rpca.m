num_blocks = 3;  % (depth-2)/6
dataset = 'cifar100'; 
save_dir = 'resnet20_r008/';  % changed according to compression rate

lam_conv1 = 0.15;
if strcmp(dataset,'cifar10')
    lam_linear = 0.14;
else
    lam_linear = 0.09;
end

if num_blocks == 3
    lam_layer1 = 0.07;
    lam_layer2 = 0.047;
    lam_layer3 = 0.0315;
elseif num_blocks == 5
    lam_layer1 = 0.07;
    lam_layer2 = 0.048;
    lam_layer3 = 0.032;
elseif num_blocks == 9
    lam_layer1 = 0.07;
    lam_layer2 = 0.048;
    lam_layer3 = 0.032;
elseif num_blocks == 18
    lam_layer1 = 0.07;
    lam_layer2 = 0.048;
    lam_layer3 = 0.032;
end

params = 0;
flops = 0;
r_conv1 = conv_decompose(conv1_weight, lam_conv1, ['mat_weights_',dataset,'/',save_dir,'conv1']);
params = 43 * r_conv1 + params;
flops = 44032 * r_conv1 + flops;
r_fc = fc_decompose(linear_weight, linear_bias, lam_linear, ['mat_weights_',dataset,'/',save_dir,'linear']);
params = 74 * r_fc + 10 + params;
flops = 74 * r_fc + flops;

for i = 0:(num_blocks-1)
    for j = 1:3
        if (j==2 && i ==0) || (j==3 && i ==0)
            r1 = conv_decompose(eval(['layer',num2str(j),'_',num2str(i),'_conv1_weight']), eval(['lam_layer', num2str(j-1)]), ['mat_weights_',dataset,'/',save_dir,num2str(j), '_', num2str(i), '_conv1']);
        else
            r1 = conv_decompose(eval(['layer',num2str(j),'_',num2str(i),'_conv1_weight']), eval(['lam_layer', num2str(j)]), ['mat_weights_',dataset,'/',save_dir,num2str(j), '_', num2str(i), '_conv1']);
        end
        if j==1
            params = 160 * r1 + params;
            flops = 163840 * r1 + flops;
        elseif (j==2 && i ==0)
            params = 176 * r1 + params;
            flops = 45056 * r1 + flops;
        elseif (j==2 && i ~=0)
            params = 320 * r1 + params;
            flops = 81920 * r1 + flops;
        elseif (j==3 && i ==0)
            params = 352 * r1 + params;
            flops = 22528 * r1 + flops;
        elseif (j==3 && i ~=0)
            params = 640 * r1 + params;
            flops = 40960 * r1 + flops;
        end
        r2 = conv_decompose(eval(['layer',num2str(j),'_',num2str(i),'_conv2_weight']), eval(['lam_layer', num2str(j)]), ['mat_weights_',dataset,'/',save_dir,num2str(j), '_', num2str(i), '_conv2']);
        if j==1
            params = 160 * r2 + params;
            flops = 163840 * r2 + flops;
        elseif j==2
            params = 320 * r2 + params;
            flops = 81920 * r2 + flops;
        elseif j==3
            params = 640 * r2 + params;
            flops = 40960 * r2 + flops;
        end
    end
end
        
if strcmp(dataset,'cifar100')
    params = 90 * r_fc + 90 + params;
    flops = 90 * r_fc + flops;
end
disp(['params_low_rank=',num2str(params)])
disp(['FLOPs_low_rank=',num2str(flops)])