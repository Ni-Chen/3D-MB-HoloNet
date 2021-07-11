% 
close all; clear; clc;

addpath(genpath('./function/'));  % Add funtion path with sub-folders
out_dir = '../keras/output/';
paper_dir = '../../papers/Optica/figure/';

format long;

%% Show holoNet result
train_names = strings(4);

% train_names(1) =  'Nz3_ppv1e-03~5e-03_dz1200um_L5_B32_N2000_lr0.001_G0.001_R0.0015';
train_names(1) =  'Nz7_ppv1e-03~5e-03_dz1200um_L6_B32_N2000_lr0.001_G0.001_R0.001';
train_names(2) =  'Nz15_ppv1e-03~5e-03_dz1200um_L5_B64_N3800_lr0.001_G0.001_R0.002';
train_names(3) =  'Nz32_ppv5e-04~1e-03_dz725um_L8_B16_N2500_lr0.00025_G0.001_R0.002';


%%
for idx = 1:1
    train_name = train_names(idx);
    train_name = convertStringsToChars(train_name);
    
    load([out_dir, 'predict_', train_name,  '.mat'])
    load([out_dir, 'gt_', train_name,  '.mat'])

    gt = single(gt);
    % Measure image quality
    %     mse(idx) = immse(predict(:), gt(:));
    %     peaksnr(idx) = psnr(predict(:), gt(:));
    ncc = corr(predict(:), gt(:));
    acc = loc_acc(gt, predict);
    ssimval = ssim(predict(:),gt(:));
    

    data_index = 10;

    oriObj = squeeze(gt(data_index, :, :, :));
    preObj = squeeze(predict(data_index, :, :, :));


    temp1 = shiftdim(oriObj, 1);
    figure; show3d(temp1, 0.0); axis equal;  colorbar off; %set(gcf, 'Color', 'None');
%     export_fig([paper_dir, train_name, '_data', num2str(data_index), '_gt.png'],'-transparent');
%     plotly_fig = fig2plotly(gcf, 'offline', true);

%     figure; scatter_3d(temp)

    temp = shiftdim(abs(preObj), 1);
    temp2 = (temp- min(temp(:)))./(max(temp(:))-min(temp(:)));
%     figure; show3d(temp, 0.0); axis equal; colorbar; colorbar off; %set(gcf, 'Color', 'None');
%     export_fig([paper_dir, train_name, '_data', num2str(data_index), '_pred.png'],'-transparent');
  
    figure; scatter_3d(temp1, temp2);
end

function acc_rate = loc_acc(gt, pred)
        
    N_rand = sum(gt(:) ==1);

    idxs = find(gt==1);
    count = 0;
    for idx = 1: N_rand
        if pred(idxs(idx)) > 0
            count = count+1;
        end      
    end
    
    acc_rate = count/N_rand;    
    
end


function scatter_3d(cube1, cube2)
    [X,Y,Z] = ndgrid(1:size(cube1,1), 1:size(cube1,2), 1:size(cube1,3));
    pointsize = 20;
    scatter3(X(:), Y(:), Z(:), pointsize, cube1(:), 'r', 'filled');
%     hold on;
%     scatter3(X(:), Y(:), Z(:), pointsize, cube2(:), 'g', 'filled');
end
