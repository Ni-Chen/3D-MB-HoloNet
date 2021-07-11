close all; 
clear; clc;

addpath(genpath('./function/'));  % Add funtion path with sub-folders
out_dir = '../keras/output/gold/';
paper_dir = '../../papers/OE/figure/';

format long;

%%

train_name = 'exp_Nz49_ppv1e-04~5e-04_dz1mm_L5_B32_N1500_lr0.0005_G0.0001_R0.0001';

data_index = [1];
indir = '../data/20200806/';
load([indir, 'holo_data.mat']);
plot_ratio = round((params.Nz*params.dz)/(params.Nx*params.pps));

matchStr = regexp(train_name,'\_','split');
obj_name = [matchStr{1} '_' matchStr{2}]

load([out_dir, train_name,  '_gt.mat'])
load([out_dir, train_name,  '_predict.mat'])
load([out_dir, train_name,  '_data.mat'])

load([paper_dir, 'exp3_fasta_recons.mat']);

gt = single(deconv_recon);

gt = (gt- min(gt(:)))./(max(gt(:))-min(gt(:)));
predict = (predict- min(predict(:)))./(max(predict(:))-min(predict(:)));

ncc = corr(predict(:), gt(:))
ssimval = ssim(predict(:),gt(:))
maeval = sum(abs(predict(:)-gt(:)))/length(predict(:))

for idx =  1:size(gt,1)
    oriObj = squeeze(gt(idx, :, :, :));
    preObj = squeeze(predict(idx, :, :, :));
    holo = squeeze(data(idx, :, :));
    imwrite((mat2gray(holo)), [paper_dir 'exp3_holo' num2str(idx) '.png']);
    
    
    gt_single_temp = oriObj;
%     gt_single_temp = shiftdim((oriObj), 1);    
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

%     plot_location(gt_single, pred_single);
    plot_location(pred_single);      
    set_ticks(((1:params.Nx) - round(params.Nx/2))*params.pps*1e3, ((1:params.Ny) - round(params.Ny/2))*params.pps*1e3, params.z*1e3, 'mm');
    xlim([0 size(gt_single,2)])
    ylim([0 size(gt_single,1)])
    zlim([0 size(gt_single,3)])
    set(gca,'DataAspectRatio',[1 1 2]);        
    xlabel(['x (mm)'], 'Position', [55.6,-3.3,-103.7], 'Rotation', 23);
    ylabel(['y (mm)'], 'Position', [-18.5,61.4,-80.2], 'Rotation', -35.5);    
    export_fig([paper_dir, obj_name, '_data' num2str(idx), '.png'], '-transparent');
    
    %% Export video
    plot_location(pred_single);      
    set_ticks(((1:params.Nx) - round(params.Nx/2))*params.pps*1e3, ((1:params.Ny) - round(params.Ny/2))*params.pps*1e3, params.z*1e3, 'mm');
    xlim([0 size(gt_single,2)])
    ylim([0 size(gt_single,1)])
    zlim([0 size(gt_single,3)])
    set(gca,'DataAspectRatio',[1 1 2]);        
    xlabel('');
    ylabel('');  
    zlabel('');
%     if ~exist([paper_dir, obj_name, '_data' num2str(idx) '/'], 'dir'),  mkdir([paper_dir, obj_name, '_data' num2str(idx) '/']),   end
    OptionZ.FrameRate=30;OptionZ.Duration=5.5;OptionZ.Periodic=true;
    CaptureFigVid([-20,10;-110,10;-190,80;-290,10;-380,10],...
                  [paper_dir, obj_name, '_data' num2str(idx)],...
                  OptionZ);
    
    %%   
    plot_location(pred_single);  axis image;  view(90, 90);   set(gca,'FontSize', 16);  
    set_ticks_normal(((1:params.Nx) - round(params.Nx/2))*params.pps*1e3, ((1:params.Ny) - round(params.Ny/2))*params.pps*1e3, params.z*1e3, 'mm');
    xlim([0 size(gt_single,2)])
    ylim([0 size(gt_single,1)])
    xlabel(['y (mm)']); ylabel(['x (mm)']);    
    export_fig([paper_dir, obj_name, '_data' num2str(idx), '_top.png'], '-transparent');
    
        
    plot_location(pred_single);  axis image;  view(-180, 0);   set(gca,'FontSize', 16);  set(gca,'DataAspectRatio',[1 1 4]);
    set_ticks_normal(((1:params.Nx) - round(params.Nx/2))*params.pps*1e3, ((1:params.Ny) - round(params.Ny/2))*params.pps*1e3, params.z*1e3, 'mm');
    xlim([1 size(gt_single,2)])
    ylim([1 size(gt_single,1)])
    zlim([1 size(gt_single,3)])
    xlabel(['x (mm)']); ylabel(['z (mm)']);    
    export_fig([paper_dir, obj_name, '_data' num2str(idx), '_side.png'], '-transparent');
    
    
    plot_location_cmp(gt_single,pred_single)
    set_ticks(((1:params.Nx) - round(params.Nx/2))*params.pps*1e3, ((1:params.Ny) - round(params.Ny/2))*params.pps*1e3, params.z*1e3, 'mm');
    xlim([1 size(gt_single,2)])
    ylim([1 size(gt_single,1)])
    zlim([1 size(gt_single,3)])
    xlabel('x (mm)', 'Position',[62.41042361419295,-6.918782137523067,-95.05372089108687],'Rotation', 22); 
    ylabel('y (mm)', 'Position',[-22.02557381798124,60.36246947773816,-88.2071741886648],'Rotation',-40); 
    set(gca,'DataAspectRatio',[1 1 2]);       
    export_fig([paper_dir, obj_name, '_data' num2str(idx), '_cmp.png'], '-transparent');
    
    
    plot_location_cmp(gt_single,pred_single), axis image;  view(90, 90);   set(gca,'FontSize', 16);  
    set_ticks_normal(((1:params.Nx) - round(params.Nx/2))*params.pps*1e3, ((1:params.Ny) - round(params.Ny/2))*params.pps*1e3, params.z*1e3, 'mm');
    xlim([0 size(gt_single,2)])
    ylim([0 size(gt_single,1)])
    xlabel(['y (mm)']); ylabel(['x (mm)']);  
    legend off;
     export_fig([paper_dir, obj_name, '_data' num2str(idx), '_cmp_xy.png'], '-transparent');
    
    
    plot_location_cmp(gt_single,pred_single), axis image;  view(90, 0);   set(gca,'FontSize', 16);  set(gca,'DataAspectRatio',[1 1 4]);
    set_ticks_normal(((1:params.Nx) - round(params.Nx/2))*params.pps*1e3, ((1:params.Ny) - round(params.Ny/2))*params.pps*1e3, params.z*1e3, 'mm');
    xlim([1 size(gt_single,2)])
    ylim([1 size(gt_single,1)])
    zlim([1 size(gt_single,3)])
   legend off;
    export_fig([paper_dir, obj_name, '_data' num2str(idx), '_cmp_xz.png'], '-transparent');
end


function [gt_num, missing_gt_num, error_num] = plot_location_cmp(gt, pred)
    figure;
    col = [0 0.5 0.5];

    [idxy1, idxx1, idxz1] = ind2sub(size(gt),find(gt>0.4));
    gt_num = size(idxy1,1);
    scatter3(idxy1, idxx1, idxz1, 45, 'r');  hold on;

    [idxy2, idxx2, idxz2] = ind2sub(size(pred),find(pred > 0.5));
    scatter3(idxy2, idxx2, idxz2, 30, 'o','MarkerEdgeColor',col, 'MarkerFaceColor',col); hold on;

    r2 = @(x,y,z,X,Y,Z) (x-X').^2 + (y-Y').^2 + (z-Z').^2;
    dist = r2(idxx1, idxy1, idxz1, idxx2, idxy2, idxz2);
    dist_min_1 = min(dist,[],2);
    dist_min_2 = min(dist,[],1)';

    % Error
    ind_out_1 = (dist_min_1 > 0);
    ind_out_2 = (dist_min_2 > 0);

    %         scatter3(idxy1(ind_out_1), idxx1(ind_out_1), idxz1(ind_out_1), 10, ...
    %             'o', 'MarkerEdgeColor',col, 'MarkerFaceColor',col); hold on;
    %         scatter3(idxy2(ind_out_2), idxx2(ind_out_2), idxz2(ind_out_2), 10, ...
    %             'o','MarkerEdgeColor',col, 'MarkerFaceColor',col);

    % unpredicted
    tmp_ind = 1:numel(ind_out_1);
    missing_gt_ind = tmp_ind(~logical(sum(dist < 1, 2)));
    missing_gt_num = numel(missing_gt_ind);

    % error unpredicted
    tmp_ind = 1:numel(ind_out_2);
    error_ind = tmp_ind(~logical(sum(dist < 1, 1)));
    error_num = numel(error_ind);


    xticks([1 size(gt,2)/2 size(gt,2)]);
    yticks([1 size(gt,1)/2 size(gt,1)]);
    zticks([1 size(gt,3)/2 size(gt,3)]);

    set(gca,'FontSize', 16);

    lgd=legend('Deconvolution', 'Predition', 'Error', 'Orientation','horizontal', 'Location', 'north');
    set(lgd,'position', [0.299999996985708   0.93   0.435000006028584   0.047619048527309]);

%     xlabel('x (mm)', 'Position',[62.41042361419295,-6.918782137523067,-95.05372089108687],'Rotation', 22); 
%     ylabel('y (mm)', 'Position',[-22.02557381798124,60.36246947773816,-88.2071741886648],'Rotation',-28); 
    zlabel('z (mm)');

    box on,
    ax = gca;
    ax.BoxStyle = 'full';
    %         grid on;

end

function set_ticks(x_tick, y_tick, z_tick, tick_unit)

    x_ticks = {floor(min(x_tick)*10)/10, 0, floor(max(x_tick)*10)/10};
    set(gca,'xticklabels', (x_ticks));
    y_ticks = {floor(min(y_tick)*10)/10, 0, floor(max(y_tick)*10)/10};
    set(gca,'yticklabels', (y_ticks));
    z_ticks = {floor(min(z_tick)),  floor((min(z_tick) + (max(z_tick)-min(z_tick))/2) *10)/10, floor(max(z_tick)*10)/10};
    set(gca,'zticklabels',(z_ticks));
    
    xlabel(['x (', tick_unit, ')'], 'Position',[-18.07688945007203,56.25850209044529,-91.15889879022643],'Rotation', 19);

    ylabel(['y (', tick_unit, ')'], 'Position',[41.59327963503006,58.31351338947252,40.321941193629755], 'Rotation', -30);
    zlabel(['z (', tick_unit, ')'], 'fontsize', 16);
end

function set_ticks_normal(x_tick, y_tick, z_tick, tick_unit)

    x_ticks = {floor(min(x_tick)*10)/10, 0, floor(max(x_tick)*10)/10};
    set(gca,'xticklabels', (x_ticks));
    y_ticks = {floor(min(y_tick)*10)/10, 0, floor(max(y_tick)*10)/10};
    set(gca,'yticklabels', (y_ticks));
    z_ticks = {floor(min(z_tick)),  floor((min(z_tick) + (max(z_tick)-min(z_tick))/2) *10)/10, floor(max(z_tick)*10)/10};
    set(gca,'zticklabels',(z_ticks));
    
    xlabel(['x (', tick_unit, ')']);

    ylabel(['y (', tick_unit, ')']);
    zlabel(['z (', tick_unit, ')']);
end

% function plot_location_cmp(s1,s2)
%     figure; 
%     [idxy1, idxx1, idxz1] = ind2sub(size(s1),find(s1 > 0.8));
%     scatter3(idxy1, idxx1, idxz1, 50, 'r');  hold on;
% 
% %     
%     [idxy2, idxx2, idxz2] = ind2sub(size(s2), find(s2 ==1));
%     col = [0 0.5 0.5];
%     scatter3(idxy2, idxx2, idxz2, 35, 'o', 'MarkerEdgeColor',col, 'MarkerFaceColor',col); 
%     
%     xticks([1 size(s1,2)/2 size(s1,2)]);
%     yticks([1 size(s1,1)/2 size(s1,1)]);
%     zticks([1 size(s1,3)/2 size(s1,3)]);
% 
%     lgd = legend('Deconvolution', 'Mo-HoloNet', 'Orientation','horizontal', 'Location', 'north');
%     set(lgd,'position', [0.299999996985708   0.93   0.435000006028584   0.047619048527309]);
% 
%     box on,
%     ax = gca; ax.BoxStyle = 'full';
%     set(gca,'FontSize', 14);
%     
% end



function plot_location(s1)
     col = [0 0.5 0.5];
    figure; 
    [idxy1, idxx1, idxz1] = ind2sub(size(s1),find(s1 > 0.8));
    scatter3(idxy1, idxx1, idxz1, 50, 'o', 'MarkerEdgeColor',col, 'MarkerFaceColor',col);  hold on;

%     
%     [idxy2, idxx2, idxz2] = ind2sub(size(s2), find(s2 ==1));
%     col = [0 0.5 0.5];
%     scatter3(idxy2, idxx2, idxz2, 35, 'o', 'MarkerEdgeColor',col, 'MarkerFaceColor',col); 
    
    xticks([1 size(s1,2)/2 size(s1,2)]);
    yticks([1 size(s1,1)/2 size(s1,1)]);
    zticks([1 size(s1,3)/2 size(s1,3)]);

%     lgd = legend('Deconvolution', 'Mo-HoloNet', 'Orientation','horizontal', 'Location', 'north');
%     set(lgd,'position', [0.299999996985708   0.93   0.435000006028584   0.047619048527309]);

    box on,
    ax = gca; ax.BoxStyle = 'full';
    set(gca,'FontSize', 16);
    
end