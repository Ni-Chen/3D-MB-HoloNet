function [gt_num, missing_gt_num, error_num] = plot_location_compare(gt, pred)
    norm_0_1 = @(img) (img- min(img(:)))./(max(img(:))-min(img(:)));
    gt = norm_0_1(gt);
    pred = norm_0_1(pred);
    
    
    figure;
    if nargin ==2
        [idxy1, idxx1, idxz1] = ind2sub(size(gt),find(gt==1));
        gt_num = size(idxy1,1);
        scatter3(idxy1, idxx1, idxz1, 50, 'r');  hold on;

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


        legend('Ground-Truth', 'Predition', 'Error', 'Location', 'northwest');
        set(gca,'FontSize',14);
        xlabel('x (voxel)', 'fontsize', 14,'Rotation', 23); ylabel('y (voxel)', 'fontsize', 14, 'Rotation',-35.5); zlabel('z (voxel)', 'fontsize', 14);

        set(gca,'FontSize',14);
        %     view(-25, 20);
        box on,
        ax = gca;
        ax.BoxStyle = 'full';
        grid on;
%         grid minor;
%         axis image;
    else
        [idxy1, idxx1, idxz1] = ind2sub(size(gt),find(gt>0.5));
        gt_num = size(idxy1,1);
        scatter3(idxy1, idxx1, idxz1, 'r');  hold on;

        missing_gt_num = 0;

        error_num = 0;
        set(gca,'FontSize',14);
        xlabel('x (voxel)', 'fontsize', 14,'Rotation', 23); ylabel('y (voxel)', 'fontsize', 14, 'Rotation',-35.5); zlabel('z (voxel)', 'fontsize', 14);

        %     view(-25, 20);
        box on,
        ax = gca;
        ax.BoxStyle = 'full';
        grid on;
%         grid minor;
%         axis image;
    end
end





