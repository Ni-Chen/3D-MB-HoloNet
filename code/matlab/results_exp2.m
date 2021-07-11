close all; clear; clc;

addpath(genpath('./function/'));  % Add funtion path with sub-folders
out_dir = '../keras/output/gold/';
paper_dir = '../../papers/Optica/figure/';

format long;


%%
train_name = 'exp_Nz11_ppv1e-05~5e-05_dz1mm_L5_B32_N1500_lr0.0005_G0.0001_R0.0001'; 


indir = '../data/x_shift/';  load([indir, 'holo_data.mat']); 
plot_ratio = round((params.Nz*params.dz)/(params.Nx*params.pps));


matchStr = regexp(train_name,'\_','split');
obj_name = [matchStr{1} '_' matchStr{2}]

load([out_dir, train_name,  '_gt.mat'])
load([out_dir, train_name,  '_predict.mat'])
load([out_dir, train_name,  '_data.mat'])

gt = single(gt);

gt = (gt- min(gt(:)))./(max(gt(:))-min(gt(:)));
predict = (predict- min(predict(:)))./(max(predict(:))-min(predict(:)));

ncc = corr(predict(:), gt(:))
ssimval = ssim(predict(:),gt(:))
maeval = sum(abs(predict(:)-gt(:)))/length(predict(:))

data_num = size(gt,1);
data_index = [1, 3, 9, 15]
% data_index = 1:16

flag = 0;
for idx =  data_index
    flag = flag+1;
    
    oriObj = squeeze(gt(idx, :, :, :));
    preObj = squeeze(predict(idx, :, :, :));
    holo = squeeze(data(idx, :, :));
    

    pred_single_temp = shiftdim(abs(preObj), 1);
    
    pred_single = zeros(params.Ny, params.Nx, params.Nz*plot_ratio);
    
    for is = 1: params.Nz
         pred_single(:,:, is*plot_ratio) = pred_single_temp(:,:,is);
    end     
    
    pred_single = (pred_single- min(pred_single(:)))./(max(pred_single(:))-min(pred_single(:)));
    
    figure; show3d(pred_single, 0.0); axis equal; colorbar; colorbar off; %set(gcf, 'Color', 'None');
    set(gca,'FontSize', 16);
    xticks([1 size(pred_single,2)/2 size(pred_single,2)]);
    yticks([1 size(pred_single,1)/2 size(pred_single,1)]);
    zticks([1 size(pred_single,3)/2 size(pred_single,3)]);
    set_ticks(((1:params.Nx) - round(params.Nx/2))*params.pps*1e3, ((1:params.Ny) - round(params.Ny/2))*params.pps*1e3, params.z*1e3, 'mm');    
    export_fig([paper_dir, obj_name, '_data', num2str(flag), '_pred.png'],'-transparent');
    

    figure; imagesc(pred_single(:,:,72)); axis image; colorbar off;  
%     figure; imagesc(sum(pred_single,3)); axis image; colorbar off;  

    set(gca,'FontSize', 16);
    xticks([1 size(pred_single,2)/2 size(pred_single,2)]);
    yticks([1 size(pred_single,1)/2 size(pred_single,1)]);
    
    x_tick = ((1:params.Nx) - round(params.Nx/2))*params.pps*1e3;
    y_tick = ((1:params.Ny) - round(params.Ny/2))*params.pps*1e3;       
    
    x_ticks = {floor(min(x_tick)*10)/10, 0, floor(max(x_tick)*10)/10};
    set(gca,'xticklabels', (x_ticks));
    y_ticks = {floor(min(y_tick)*10)/10, 0, floor(max(y_tick)*10)/10};
    set(gca,'yticklabels', (y_ticks));

    xlabel(['x (mm)']);
    ylabel(['y (mm)']);
    zlabel(['z (mm)']);
    
    export_fig([paper_dir, obj_name, '_data', num2str(flag), '_pred_top.png'],'-transparent');
    
    imwrite((mat2gray(holo)), [paper_dir 'exp2_holo' num2str(flag) '.png']);
    
end


function set_ticks(x_tick, y_tick, z_tick, tick_unit)

    x_ticks = {floor(min(x_tick)*10)/10, 0, floor(max(x_tick)*10)/10};
    set(gca,'xticklabels', (x_ticks));
    y_ticks = {floor(min(y_tick)*10)/10, 0, floor(max(y_tick)*10)/10};
    set(gca,'yticklabels', (y_ticks));
    z_ticks = {floor(min(z_tick)),  floor((min(z_tick) + (max(z_tick)-min(z_tick))/2) *10)/10, floor(max(z_tick)*10)/10};
    set(gca,'zticklabels',(z_ticks));
    
    xlabel(['x (', tick_unit, ')'], 'fontsize', 14, 'Rotation', 23);
    ylabel(['y (', tick_unit, ')'], 'fontsize', 14, 'Rotation', -35.5);
    zlabel(['z (', tick_unit, ')'], 'fontsize', 14);
end
