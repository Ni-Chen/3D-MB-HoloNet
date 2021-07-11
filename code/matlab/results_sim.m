close all; clear; clc;

addpath(genpath('./function/'));  % Add funtion path with sub-folders
out_dir = '../keras/output/';
paper_dir = '../../papers/Optica/figure/';

format long;


%%

% train_name = 'sim_Nz7_ppv1e-03~5e-03_dz1200um_L5_B32_N1000_lr0.002_G0.0001_R0.0005'; data_index = [11];
% train_name = 'sim_Nz15_ppv5e-04~2.5e-03_dz1200um_L5_B32_N1000_lr0.002_G0.0001_R0.0001'; data_index = [22];
% train_name = 'sim_Nz32_ppv2e-04~1e-03_dz725um_L5_B32_N1000_lr0.001_G0.0001_R0.0002'; data_index = [32];


% train_name = 'sim_Nz30_ppv1e-04~5e-04_dz3mm_L5_B32_N1000_lr0.002_G0.0001_R0.0001'; data_index = [30];

train_name = 'sim_Nz64_ppv1.5e-06~4.8e-04_dz500um_L5_B32_N1000_lr0.001_G0.0001_R0.0001'; data_index = [32, 34];

% train_name = 'sim_Nz11_ppv1e-05~5e-05_dz1mm_L5_B32_N1000_lr0.001_G0.0001_R0.0001'; data_index = [30 40];



matchStr = regexp(train_name,'\_','split');
obj_name = [matchStr{1} '_' matchStr{2}]

load([out_dir, train_name,  '_gt.mat'])
load([out_dir, train_name,  '_predict.mat'])

gt = single(gt);

gt = (gt- min(gt(:)))./(max(gt(:))-min(gt(:)));
predict = (predict- min(predict(:)))./(max(predict(:))-min(predict(:)));

ncc = corr(predict(:), gt(:))
ssimval = ssim(predict(:),gt(:))
maeval = sum(abs(predict(:)-gt(:)))/length(predict(:))

[Ny, Nx, Nz] = size(shiftdim(squeeze(gt(1, :, :, :)),1));
plot_ratio = 5;

% for idx =  data_index
%     oriObj = squeeze(gt(idx, :, :, :));
%     preObj = squeeze(predict(idx, :, :, :));
%     
%     
%     gt_single_temp = shiftdim(oriObj, 1);
%     gt_single = zeros(Ny, Nx, Nz*plot_ratio);    
%     for is = 1: Nz
%         gt_single(:,:, is*plot_ratio) = gt_single_temp(:,:,is);
%     end
%     gt_single = (gt_single- min(gt_single(:)))./(max(gt_single(:))-min(gt_single(:)));
%     
%     figure; show3d(gt_single, 0.0);  colorbar off; %set(gcf, 'Color', 'None');
%     xlabel('x (voxel)', 'fontsize', 14,'Rotation', 23); ylabel('y (voxel)', 'fontsize', 14, 'Rotation',-35.5); zlabel('z (voxel)', 'fontsize', 14);
%     export_fig([paper_dir, obj_name, '_data', num2str(idx), '_gt.png'],'-transparent');
%     
%     pred_single_temp = shiftdim(abs(preObj), 1);
%     pred_single = zeros(Ny, Nx, Nz*plot_ratio);    
%     for is = 1: Nz
%         pred_single(:,:, is*plot_ratio) = pred_single_temp(:,:,is);
%     end
%     
%     pred_single = (pred_single- min(pred_single(:)))./(max(pred_single(:))-min(pred_single(:)));
%     figure; show3d(pred_single, 0.0); colorbar; colorbar off; %set(gcf, 'Color', 'None');
%     xlabel('x (voxel)', 'fontsize', 14,'Rotation', 23); ylabel('y (voxel)', 'fontsize', 14, 'Rotation',-35.5); zlabel('z (voxel)', 'fontsize', 14);
% %     set_ticks(((1:params.Nx) - round(params.Nx/2))*params.pps*1e3, ((1:params.Ny) - round(params.Ny/2))*params.pps*1e3, params.z*1e3, 'mm');
%     export_fig([paper_dir, obj_name, '_data', num2str(idx), '_pred.png'],'-transparent');
%     
%     diff = abs(pred_single - gt_single);
%     figure; show3d(diff, 0.0); colorbar; colormap(1-hot);
%     xlabel('x (voxel)', 'fontsize', 14,'Rotation', 23); ylabel('y (voxel)', 'fontsize', 14, 'Rotation',-35.5); zlabel('z (voxel)', 'fontsize', 14);
%     export_fig([paper_dir, obj_name, '_data', num2str(idx), '_diff.png'], '-transparent');
% 
% end


for idx =  1:50
    idx
    oriObj = squeeze(gt(idx, :, :, :));
    tmp = find(oriObj>0);
    Np = length(tmp(:))
    
%     preObj = squeeze(predict(idx, :, :, :));
%     
%     
%     gt_single = shiftdim(oriObj, 1);
%     gt_single = (gt_single- min(gt_single(:)))./(max(gt_single(:))-min(gt_single(:)));
%     figure; show3d(gt_single, 0.0);  colorbar off; %set(gcf, 'Color', 'None');
% 
%     pred_single = shiftdim(abs(preObj), 1);
%     pred_single = (pred_single- min(pred_single(:)))./(max(pred_single(:))-min(pred_single(:)));
%     figure; show3d(pred_single, 0.0); colorbar; colorbar off; %set(gcf, 'Color', 'None');
% 
%      figure;
%     [gt_num, missing_gt_num, error_num] = plot_location_compare(gt_single, pred_single); axis equal;
% %     set(gca,'DataAspectRatio',[2 2 1]);
%     acc_val = (gt_num -missing_gt_num)/gt_num
%     
%     
% %     export_fig([paper_dir, obj_name, '_compare.png'], '-transparent');
%      export_fig([paper_dir, obj_name, '_data' num2str(idx) '_compare.png'], '-transparent');

end


function [gt_num, missing_gt_num, error_num] = plot_location_compare(gt, pred)
   

        [idxy1, idxx1, idxz1] = ind2sub(size(gt),find(gt==1));
        gt_num = size(idxy1,1);
        scatter3(idxy1, idxx1, idxz1, 60, 'r');  hold on;

        [idxy2, idxx2, idxz2] = ind2sub(size(pred),find(pred > 0.1));
        scatter3(idxy2, idxx2, idxz2, 30, 'b'); hold on;

        r2 = @(x,y,z,X,Y,Z) (x-X').^2 + (y-Y').^2 + (z-Z').^2;
        dist = r2(idxx1, idxy1, idxz1, idxx2, idxy2, idxz2);
        dist_min_1 = min(dist,[],2);
        dist_min_2 = min(dist,[],1)';

        % Error
        ind_out_1 = (dist_min_1 > 0);
        ind_out_2 = (dist_min_2 > 0);

        col = [0 0.5 0.5];
        scatter3(idxy1(ind_out_1), idxx1(ind_out_1), idxz1(ind_out_1), 10, ...
            'o', 'MarkerEdgeColor',col, 'MarkerFaceColor',col); hold on;
        scatter3(idxy2(ind_out_2), idxx2(ind_out_2), idxz2(ind_out_2), 10, ...
            'o','MarkerEdgeColor',col, 'MarkerFaceColor',col);

        % unpredicted
        tmp_ind = 1:numel(ind_out_1);
        missing_gt_ind = tmp_ind(~logical(sum(dist < 1, 2)));
        missing_gt_num = numel(missing_gt_ind);

        % error unpredicted
        tmp_ind = 1:numel(ind_out_2);
        error_ind = tmp_ind(~logical(sum(dist < 1, 1)));
        error_num = numel(error_ind);


        set(gca,'FontSize',16);
        
        lgd=legend('Ground-Truth', 'Predition', 'Error', 'Orientation','horizontal', 'Location', 'north');
        set(lgd,'position', [0.299999996985708   0.93   0.435000006028584   0.047619048527309]);
         
        xlabel('x (voxel)','Rotation', 23); ylabel('y (voxel)', 'Rotation',-35.5); zlabel('z (voxel)');

        box on,
        ax = gca;
        ax.BoxStyle = 'full';
        grid on;

end






