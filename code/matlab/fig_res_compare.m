% 
close all; clear; clc;

addpath(genpath('./function/'));  % Add funtion path with sub-folders
out_dir = '../keras/output/';

format long;

%% Show holoNet result

obj_name = 'Nz3_ppv5_30db_dz';    % done, 8(100)-->2(100)
train_setting = '_L9_B32_N500_lr0.01_G0.0001_R1e-05';
paper_dir = '../../papers/Optica/figure/data/';


dzs = [1200 600 300 150];

for idx = 1:length(dzs)
    dz = dzs(idx);
    
    load([out_dir, 'predict_', obj_name, num2str(dz), 'um', train_setting, '.mat']);
    load([out_dir, 'gt_', obj_name, num2str(dz), 'um', train_setting, '.mat']);
        
    gt = single(gt);
    % Measure image quality
    nccval(idx) = corr(predict(:), gt(:));
    ssimval(idx) = ssim(predict(:),gt(:));
    maeval(idx) = sum(abs(predict(:)-gt(:)))/length(predict(:));


    data_index = 1;

    oriObj = squeeze(gt(data_index, :, :, :));
    obj_pred = squeeze(predict(data_index, :, :, :));
    
    oriObj = shiftdim(oriObj, 1);
    obj_pred = shiftdim(abs(obj_pred), 1);

    
    %%
    run(['../data/particle_' obj_name  num2str(dz) 'um_sys_param.m']);
    [Ny, Nx, Nz] = size(oriObj);
    z_range = z0 + ((1:Nz)-round(Nz/2))*dz;   % axial depth span of the object
    [otf3d, psf3d, pupil] = OTF3D(Ny, Nx, lambda, pps, z_range);    % generate otf3d of the system
    [label, holo1] = gaborHolo(oriObj, otf3d, 'Gaussian', 30);
    obj_bp = abs(iMatProp3D(holo1, otf3d)); 
    
    figure; 
    subplot(311); imagesc(plotdatacube(oriObj)); %title('original object'); 
    axis image; drawnow; colormap(hot); axis off;
    subplot(312); imagesc(plotdatacube(obj_bp)); %title('BP reconstruction'); 
    axis image; drawnow; colormap(hot); axis off;
    subplot(313); imagesc(plotdatacube(abs(obj_pred)));% title('MHoloNet reconstruction'); 
    axis image; drawnow; colormap(hot); axis off;

    export_fig([paper_dir, 'res_compare_dz', num2str(dz*1e6), 'um.eps'], '-transparent');
end


t = [roundn(dzs',-4), roundn(nccval',-4), roundn(ssimval',-4), roundn(maeval',-4)];
t = array2table(t);
t.Properties.VariableNames(1:4) = {'dz', 'PCC','SSIM', 'MAE'};
writetable(t, [paper_dir 'res_compare.csv'])
