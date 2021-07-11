close all; clear; clc;

addpath(genpath('./function/'));  % Add funtion path with sub-folders
out_dir = '../keras/output/gold/';
paper_dir = '../../papers/Optica/figure/';

format long;

%%

train_name = 'exp_Nz30_ppv1e-04~5e-04_dz3mm_L5_B32_N1000_lr0.002_G0.0001_R0.0005';

% train_name = 'exp_Nz30_ppv1e-04~5e-04_dz3mm_L5_B32_N1500_lr0.0005_G0.0001_R0.0001';



data_index = [1,2,3,4];
indir = '../data/test_data_particle/';
load([indir, 'holo_data.mat']);
plot_ratio = round((params.Nz*params.dz)/(params.Nx*params.pps));

matchStr = regexp(train_name,'\_','split');
obj_name = [matchStr{1} '_' matchStr{2}]

load([out_dir, train_name,  '_gt.mat'])
load([out_dir, train_name,  '_predict.mat'])

load([paper_dir, 'fasta_recons.mat']);

gt = single(deconv_recon);

gt = (gt- min(gt(:)))./(max(gt(:))-min(gt(:)));
predict = (predict- min(predict(:)))./(max(predict(:))-min(predict(:)));

ncc = corr(predict(:), gt(:))
ssimval = ssim(predict(:),gt(:))
maeval = sum(abs(predict(:)-gt(:)))/length(predict(:))

for idx =  data_index
    oriObj = squeeze(gt(idx, :, :, :));
    preObj = squeeze(predict(idx, :, :, :));
    
    
    gt_single_temp = oriObj;
    gt_single = zeros(params.Ny, params.Nx, params.Nz*plot_ratio);    
    for is = 1: params.Nz
        gt_single(:,:, is*plot_ratio) = gt_single_temp(:,:,is);
    end
    
    gt_single = (gt_single- min(gt_single(:)))./(max(gt_single(:))-min(gt_single(:)));
    %     figure; show3d(gt_single, 0.0); axis equal;  colorbar off; %set(gcf, 'Color', 'None');
    %     xlabel('x (voxel)', 'fontsize', 14,'Rotation', 23); ylabel('y (voxel)', 'fontsize', 14, 'Rotation',-35.5); zlabel('z (voxel)', 'fontsize', 14);
    %         export_fig([paper_dir, train_name, '_data', num2str(data_index), '_gt.png'],'-transparent');
    
    
    pred_single_temp = shiftdim(abs(preObj), 1);    
    pred_single = zeros(params.Ny, params.Nx, params.Nz*plot_ratio);    
    for is = 1: params.Nz
        pred_single(:,:, is*plot_ratio) = pred_single_temp(:,:,is);
    end    
    pred_single = (pred_single- min(pred_single(:)))./(max(pred_single(:))-min(pred_single(:)));

    plot_location(gt_single, pred_single);
    set_ticks(((1:params.Nx) - round(params.Nx/2))*params.pps*1e3, ((1:params.Ny) - round(params.Ny/2))*params.pps*1e3, params.z*1e3, 'mm'); axis image;
    xlim([0 size(gt_single,2)])
    ylim([0 size(gt_single,1)])
    zlim([0 size(gt_single,3)])
    
    export_fig([paper_dir, obj_name, '_data' num2str(idx), '_compare.png'], '-transparent');
    
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

function plot_location(s1,s2)
    figure; 
    [idxy1, idxx1, idxz1] = ind2sub(size(s1),find(s1 > 0.2));
    scatter3(idxy1, idxx1, idxz1, 50, 'r');  hold on;

    
    [idxy2, idxx2, idxz2] = ind2sub(size(s2), find(s2 ==1));
    col = [0 0.5 0.5];
    scatter3(idxy2, idxx2, idxz2, 35, 'o', 'MarkerEdgeColor',col, 'MarkerFaceColor',col); 
    
    xticks([1 size(s1,2)/2 size(s1,2)]);
    yticks([1 size(s1,1)/2 size(s1,1)]);
    zticks([1 size(s1,3)/2 size(s1,3)]);

    lgd = legend('Deconvolution', 'Mo-HoloNet', 'Orientation','horizontal', 'Location', 'north');
    set(lgd,'position', [0.299999996985708   0.93   0.435000006028584   0.047619048527309]);

    box on,
    ax = gca; ax.BoxStyle = 'full';
    set(gca,'FontSize', 14);
    
end