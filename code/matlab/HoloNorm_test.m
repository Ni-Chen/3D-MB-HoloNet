
close all; clear; clc;

addpath(genpath('./function/'));  % Add funtion path with sub-folders


norm_0_1 = @(img) (img- min(img(:)))./(max(img(:))-min(img(:)));
norm_std = @(img) (img - mean(img(:))) / std(img(:));
norm_std2d = @(img) (img - mean(img)) ./ std(img);
inv_norm_std = @(img, m, s) img.*s + m;

holoDataType = 1;
objType = 'particle';  % 'particle', 'lines'

%%
noise_type  = 'Gaussian';
point_size = 0;      % pixel size of one random point
data_num = 1;        % number of train data

Nxy = 32;              % lateral size
Nz = 15;               % axial size

lambda = 633e-9;       % Illumination wavelength
pps    = 20e-6;         % pixel pitch of CCD camera
z0     = 10e-3;        % Distance between the hologram and the center plane of the 3D object
dz     = 1.2e-3;         % depth interval of the object slices
z_range = z0 + ((1:Nz)-round(Nz/2))*dz;   % axial depth span of the object



NA = pps*Nxy/2/z0

delta_x = lambda/(NA)
delta_z = 2*lambda/(NA^2)



% fixed ppvs
noise_level = 30;   % DB of the noise

ppv_min = 2e-3;
ppv_max = ppv_min;

params.lambda = lambda;
params.pp_holo = pps;
params.z = z_range;
params.Ny = Nxy;
params.Nx = Nxy;

%% Generate training data

[otf3d, psf3d, pupil] = OTF3D(Nxy, Nxy, lambda, pps, z_range);    % generate otf3d of the system


N_random = randi([round(ppv_min*Nxy*Nxy*Nz) round(ppv_max*Nxy*Nxy*Nz)], 1, 1) % particle concentration
obj = randomScatter(Nxy, Nz, N_random);   % randomly located particles

% obj_norm = holo_norm(obj);
obj_norm = (obj);
[label, data] = gaborHolo(obj_norm, otf3d, noise_type, noise_level);


holo_complex_noNoise = MatProp3D(obj, otf3d);    % propagate the object to the hologram plane 
holo_noNoise = (holo_complex_noNoise).*conj(holo_complex_noNoise);
% holo_noNoise = norm_0_1(holo_noNoise);
imhist(holo_noNoise)
   

data_norm = norm_0_1(data); imhist(data_norm);
data_norm = norm_std(data_norm); 
figure; 
subplot(221); imhist(data);
subplot(222); imagesc(data); axis image; colorbar;title(['holo: \mu ' num2str(mean2(data)), ', \sigma ', num2str(std2(data))]);
subplot(223); imhist(data_norm);
subplot(224); imagesc(data_norm); axis image; colorbar;title(['norm_0_1(holo): \mu ' num2str(mean2(data_norm)), ', \sigma ', num2str(std2(data_norm))]);

label_std = norm_std2d(label); 
% figure; 
% subplot(121); imhist(label);title(['holo: \mu ' num2str(mean2(label)), ', \sigma ', num2str(std2(label))]);
% subplot(122); imhist(label_std);title(['holo: \mu ' num2str(mean2(label_std)), ', \sigma ', num2str(std2(label_std))]);
figure; show3d(label, 0.0); axis equal; 
figure; show3d(label_std, 0.0); axis equal;  
 
% label_std_inverse = inv_norm_std(label_std, mean(label(:)), std(label(:)));
% figure; show3d(label_std_inverse, 0.0); axis equal; 

figure;
data_ori = data;
subplot(241); imagesc((data_ori)); title(['holo: \mu ' num2str(mean2(data_ori)), ', \sigma ', num2str(std2(data_ori))]); axis image; drawnow; colormap(hot); colorbar; caxis([min(data_ori(:)) max(data_ori(:))]);axis off;
subplot(242); imhist(data_ori);
temp = abs(iMatProp3D(data_ori, otf3d)); 
% temp = norm_0_1(temp);
subplot(243); imagesc(plotdatacube(temp)); title(['Reconstruction: \mu ' num2str(mean2(temp)), ', \sigma ', num2str(std2(temp))]); axis image; drawnow; colormap(hot); colorbar;caxis([min(temp(:)) max(temp(:))]); axis off;
subplot(244); imhist(temp);


% temp = data-min(data(:));
data_norm1 = norm_0_1(data);
subplot(245); imagesc((data_norm1)); title(['holo_{0,1}: \mu ' num2str(mean2(data_norm1)), ', \sigma ', num2str(std2(data_norm1))]); axis image; drawnow; colormap(hot); colorbar; caxis([min(data_norm1(:)) max(data_norm1(:))]);axis off;
subplot(246); imhist(data_norm1);
temp1 = abs(iMatProp3D(data_norm1, otf3d));
% temp1 = norm_0_1(temp1);
subplot(247); imagesc(plotdatacube(temp1)); title(['Reconstruction: \mu ' num2str(mean(temp1(:))), ', \sigma ', num2str(std(temp1(:)))]); axis image; drawnow; colormap(hot); colorbar;caxis([min(temp1(:)) max(temp1(:))]); axis off;
subplot(248);  imhist(abs(iMatProp3D(data_norm1, otf3d)));



data_norm3 = norm_std(norm_0_1(data));
% figure; imhist(data_norm3);
% data_norm3 = zscore(data);
figure;
subplot(121); imagesc((data_norm3)); title(['holo\_{z\_score}: \mu ' num2str(mean2(data_norm3)), ', \sigma ', num2str(std2(data_norm3))]); axis image; drawnow; colormap(hot); colorbar; caxis([min(data_norm3(:)) max(data_norm3(:))]);axis off;
temp3 = abs(iMatProp3D(data_norm3, otf3d));
temp3 = norm_0_1(temp3);
subplot(122); imagesc(plotdatacube(temp3)); title(['Reconstruction: \mu ' num2str(mean(temp3(:))), ', \sigma ', num2str(std(temp3(:)))]); axis image; drawnow; colormap(hot); colorbar; caxis([min(temp3(:)) max(temp3(:))]);axis off;


% min(data_norm3(:))
% max(data_norm3(:))
% 
% min(temp3(:))
% max(temp3(:))
% 
% obj_norm = norm_std(obj);
% figure;
% subplot(211);
% imagesc(plotdatacube(obj)); title(['3D object']); axis image; drawnow; colormap(hot); colorbar; 
% subplot(212);
% imagesc(plotdatacube(obj_norm)); title(['3D object notm']); axis image; drawnow; colormap(hot); colorbar; 
% 
% mean2(obj_norm)
% std2(obj_norm)
