close all; clear; clc;

addpath(genpath('./function/'));  % Add funtion path with sub-folders
out_dir = '../keras/output/gold/';
paper_dir = '../../papers/Optica/figure/';

format long;


%% Show holoNet result
noise_levels= [10 15 20 25 30 35 40 45 50];

for idx = 1:length(noise_levels)
    noise_level = noise_levels(idx);
    
    load([out_dir, 'test_noise_', num2str(noise_level), 'db_pred.mat'])
    load([out_dir, 'test_noise_', num2str(noise_level), 'db_gt.mat'])
    
    gt = single(gt);
%     gt = (gt- min(gt(:)))./(max(gt(:))-min(gt(:)));
    predict = (predict- min(predict(:)))./(max(predict(:))-min(predict(:)));
    
    % Measure image quality
%     peaksnr(idx) = psnr(predict(:), gt(:));
    ncc(idx) = corr(predict(:), gt(:));
    ssimval(idx) = ssim(predict(:),gt(:));
    maeval(idx) = sum(abs(predict(:)-gt(:)))/length(predict(:));

    data_index = 31;

    oriObj = squeeze(gt(data_index, :, :, :));
    preObj = squeeze(predict(data_index, :, :, :));

    N_rand = oriObj(oriObj == 1);
    size(N_rand)
%%
    gt_single = shiftdim(oriObj, 1);
%     figure; show3d(gt_single, 0.0); title('Ground truth'); axis equal; colorbar off; %set(gcf, 'Color', 'None');
%     export_fig([paper_dir, 'test_noise',  num2str(noise_level), 'db_gt.png'],'-transparent');


    pred_single = shiftdim(abs(preObj), 1);
%     pred_single = pred_single./max(pred_single(:));
%     figure; show3d(pred_single, 0.0); title('Prediction'); axis equal; colorbar off;%set(gcf, 'Color', 'None');
%     export_fig([paper_dir, 'test_noise',  num2str(noise_level),  'db_pred.png'],'-transparent');
    
    
    [gt_num, missing_gt_num, error_num] = plot_location_compare(gt_single, pred_single);
%     axis image;
%     xlabel('x (voxel)', 'fontsize', 14,'Rotation', 23); ylabel('y (voxel)', 'fontsize', 14, 'Rotation',-35.5); zlabel('z (voxel)', 'fontsize', 14);
%     set(gca,'FontSize',14);
    
    acc(idx) = (gt_num -missing_gt_num)/gt_num ;
    export_fig([paper_dir 'test_noise_' num2str(noise_level), 'db_compare.png'], '-transparent');

end


t = [roundn(noise_levels',-5), roundn(ncc',-5), roundn(ssimval',-5), roundn(maeval',-5),  roundn(acc',-5)];
t = array2table(t);
t.Properties.VariableNames(1:5) = {'noise', 'PCC', 'SSIM', 'MAE', 'ACC'};
writetable(t, [paper_dir '/data/test_noise.csv'])

%%
ppvs = [1 2 3 4 5 6 7 8 9 10];

for idx = 1:length(ppvs)
    ppv = ppvs(idx);
    
    load([out_dir, 'test_ppv',  num2str(ppv), '_pred.mat'])
    load([out_dir, 'test_ppv',  num2str(ppv), '_gt.mat'])
    
    gt = single(gt);
    gt = (gt- min(gt(:)))./(max(gt(:))-min(gt(:)));
    predict = (predict- min(predict(:)))./(max(predict(:))-min(predict(:)));
    
    % Measure image quality
%     mse(idx) = immse(predict(:), gt(:));
%     peaksnr(idx) = psnr(predict(:), gt(:));
    ncc(idx) = corr(predict(:), gt(:));
%     acc(idx) = loc_acc(gt, predict);
    ssimval(idx) = ssim(predict(:),gt(:));
    maeval(idx) = sum(abs(predict(:)-gt(:)))/length(predict(:));


    data_index = 1;

    oriObj = squeeze(gt(data_index, :, :, :));
    preObj = squeeze(predict(data_index, :, :, :));

%     N_rand = oriObj(oriObj == 1);
%     size(N_rand)
%%
    gt_single = shiftdim(oriObj, 1);
%     figure; show3d(gt_single, 0.0); title('Ground truth'); axis equal; colorbar off; %set(gcf, 'Color', 'None');
%     export_fig([paper_dir, 'test_ppv',  num2str(ppv), '_data_gt.png'],'-transparent');
    % figure; show3d(pred_single, 0.01); view(0, 0); colorbar off; axis equal;
    % figure; show3d(pred_single, 0.01); view(0, -90); colorbar off; axis equal;

    pred_single = shiftdim(abs(preObj), 1);
    % pred_single = pred_single./max(pred_single(:));
%     figure; show3d(pred_single, 0.0); title('Prediction'); axis equal; colorbar off;%set(gcf, 'Color', 'None');
%     export_fig([paper_dir, 'test_ppv',  num2str(ppv), '_data_pred.png'],'-transparent');
    % figure; show3d(pred_single, 0.01); view(0, 0); colorbar off; axis equal;
    % figure; show3d(pred_single, 0.01); view(0, -90); colorbar off; axis equal;
    
      
    [gt_num, missing_gt_num, error_num] = plot_location_compare(gt_single, pred_single);

    
    acc(idx) = (gt_num -missing_gt_num)/gt_num ;
    export_fig([paper_dir 'test_ppv' num2str(ppv), '_compare.png'], '-transparent');
end


% acc = single(acc);
t = [roundn(ppvs',-5), roundn(ncc',-5), roundn(ssimval',-5), roundn(maeval',-5),  roundn(acc',-5)];
t = array2table(t);
t.Properties.VariableNames(1:5) = {'ppv', 'PCC', 'SSIM', 'MAE', 'ACC'};
writetable(t, [paper_dir '/data/test_ppv.csv'])




function [gt_num, missing_gt_num, error_num] = plot_location_compare(gt, pred)
    figure;

        [idxy1, idxx1, idxz1] = ind2sub(size(gt),find(gt==1));
        gt_num = size(idxy1,1);
        scatter3(idxy1, idxx1, idxz1, 60, 'r');  hold on;

        [idxy2, idxx2, idxz2] = ind2sub(size(pred),find(pred > 0.5));
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
