clc;clear;close all;

data_dir = '../model/';
paper_dir = '../../papers/Optica/figure/data/';

if ~isfolder(paper_dir)
    mkdir(paper_dir)
end

addpath(genpath('./function/'));  % Add funtion path with sub-folders
out_dir = '../output/';

format long;

%% Show holoNet result

data_dir = ['./data/train_', 'data_11_rand_50db_dz1mm', '/'];

%%
load([out_dir, 'predict_', train_name,  '.mat'])
load([out_dir, 'gt_', train_name,  '.mat'])

gt = single(gt);
% Measure image quality
%     mse(idx) = immse(predict(:), gt(:));
%     peaksnr(idx) = psnr(predict(:), gt(:));
ncc = corr(predict(:), gt(:));
% acc = loc_acc(gt, predict);
ssimval = ssim(predict(:),gt(:));

data_index = 1;

oriObj = squeeze(gt(data_index, :, :, :));
preObj = squeeze(predict(data_index, :, :, :));


temp = shiftdim(oriObj, 1);
figure; show3d(temp, 0.0); title('Ground truth'); axis equal;  %set(gcf, 'Color', 'None');
% export_fig([out_dir, train_name, '_data', num2str(data_index), '_gt.png']);
% figure; show3d(temp, 0.01); view(0, 0); colorbar off; axis equal;
% figure; show3d(temp, 0.01); view(0, -90); colorbar off; axis equal;


temp = shiftdim(abs(preObj), 1);
% temp = temp./max(temp(:));
figure; show3d(temp, 0.0); title('Prediction'); axis equal; %set(gcf, 'Color', 'None');
% export_fig([out_dir, train_name, '_data', num2str(data_index), '_pred.png']);
% figure; show3d(temp, 0.01); view(0, 0); colorbar off; axis equal;
% figure; show3d(temp, 0.01); view(0, -90); colorbar off; axis equal;


%%
noise_levels= [10 20 25 35 40];

for idx = 1:length(noise_levels)
    noise_level = noise_levels(idx);
    
    load([out_dir, 'predict_', train_name, '_test_', num2str(noise_level), 'db.mat'])
    load([out_dir, 'gt_', train_name, '_test_', num2str(noise_level), 'db.mat'])
    
    gt = single(gt);
    % Measure image quality
%     mse(idx) = immse(predict(:), gt(:));
%     peaksnr(idx) = psnr(predict(:), gt(:));
    ncc(idx) = corr(predict(:), gt(:));
%     acc(idx) = loc_acc(gt, predict);
    ssimval(idx) = ssim(predict(:),gt(:));

    data_index = 1;

    oriObj = squeeze(gt(data_index, :, :, :));
    preObj = squeeze(predict(data_index, :, :, :));

%     N_rand = oriObj(oriObj == 1);
%     size(N_rand)

    %%
    temp = shiftdim(oriObj, 1);
    figure; show3d(temp, 0.0); title('Ground truth'); axis equal;  %set(gcf, 'Color', 'None');
    % export_fig([out_dir, train_name, '_data', num2str(data_index), '_gt.png']);
    % figure; show3d(temp, 0.01); view(0, 0); colorbar off; axis equal;
    % figure; show3d(temp, 0.01); view(0, -90); colorbar off; axis equal;

    temp = shiftdim(abs(preObj), 1);
    % temp = temp./max(temp(:));
    figure; show3d(temp, 0.0); title('Prediction'); axis equal; %set(gcf, 'Color', 'None');
    % export_fig([out_dir, train_name, '_data', num2str(data_index), '_pred.png']);
    % figure; show3d(temp, 0.01); view(0, 0); colorbar off; axis equal;
    % figure; show3d(temp, 0.01); view(0, -90); colorbar off; axis equal;
end

% mse
% peaksnr
ncc
ssimval


figure;
plot(noise_levels, ncc, ssimval);

t = [noise_levels ncc ssimval];
save([paper_dir 'test_noise.dat'], 't', '-ascii');

%%
ppvs = [0.5 2 6 8 10 12 16 18 20];

for idx = 1:length(ppvs)
    ppv = ppvs(idx);
    
    load([out_dir, 'predict_', train_name, '_test_ppv', num2str(ppv), '.mat'])
    load([out_dir, 'gt_', train_name, '_test_ppv', num2str(ppv), '.mat'])
    
    gt = single(gt);
    % Measure image quality
%     mse(idx) = immse(predict(:), gt(:));
%     peaksnr(idx) = psnr(predict(:), gt(:));
    ncc(idx) = corr(predict(:), gt(:));
%     acc(idx) = loc_acc(gt, predict);
    ssimval(idx) = ssim(predict(:),gt(:));



    data_index = 1;

    oriObj = squeeze(gt(data_index, :, :, :));
    preObj = squeeze(predict(data_index, :, :, :));

%     N_rand = oriObj(oriObj == 1);
%     size(N_rand)

    %%
    temp = shiftdim(oriObj, 1);
    figure; show3d(temp, 0.0); title('Ground truth'); axis equal;  %set(gcf, 'Color', 'None');
    % export_fig([out_dir, train_name, '_data', num2str(data_index), '_gt.png']);
    % figure; show3d(temp, 0.01); view(0, 0); colorbar off; axis equal;
    % figure; show3d(temp, 0.01); view(0, -90); colorbar off; axis equal;

    temp = shiftdim(abs(preObj), 1);
    % temp = temp./max(temp(:));
    figure; show3d(temp, 0.0); title('Prediction'); axis equal; %set(gcf, 'Color', 'None');
    % export_fig([out_dir, train_name, '_data', num2str(data_index), '_pred.png']);
    % figure; show3d(temp, 0.01); view(0, 0); colorbar off; axis equal;
    % figure; show3d(temp, 0.01); view(0, -90); colorbar off; axis equal;
end


t = [noise_levels ncc ssimval];
save([paper_dir 'test_ppvs.dat'], 't', '-ascii');





