close all; 
clear; clc;

addpath(genpath('./function/'));  % Add funtion path with sub-folders
data_dir = '../data/';

rng(0);

norm_0_1 = @(img) (img- min(img(:)))./(max(img(:))-min(img(:)));
norm_max = @(img) img./max(abs(img(:)));
norm_std = @(img) (img - mean(img(:))) / std(img(:));
inv_norm_std = @(img, m, s) img.*s + m;
norm_min_max =  @(img, min_val, max_val) (img - min(img(:)))*(max_val-min_val) / (max(img(:))-min(img(:))) + min_val;

norm_holo =  @(img) (norm_0_1(img));


%%
noise_type  = 'Gaussian';
point_size = 0;        % pixel size of one random point
data_num = 1;       % number of train holo

% System parameters
Nxy = 256;              % lateral size
new_size = 128
Nz = 3;                % axial size
lambda = 660e-9;       % Illumination wavelength
pps    = 3.45e-6;        % pixel pitch of CCD camera
sr     = 25e-6;
z0     = 19.7e-3;        % Distance between the hologram and the center plane of the 3D object
dz     = 1e-3;       % depth interval of the object slices

% z_range = z0 + ((1:Nz)-round(Nz/2))*dz;   % axial depth span of the object
z_range = z0 + (0:Nz-1)*dz;   % axial depth span of the object


delta_x = lambda/( pps*Nxy/2/z0)
delta_z = 2*lambda/(( pps*Nxy/2/z0)^2)


% fixed ppvs
noise_level = 40;   % DB of the noise

ppv_min = 4e-5*4;
ppv_max = ppv_min;


if ppv_min == ppv_max
    ppv_text = [num2str(ppv_min,'%.e')];
else
    ppv_text = [num2str(ppv_min,'%.e') '~' num2str(ppv_max,'%.1e')];
end

params.lambda = lambda;
params.pps = pps;
params.z = z_range;
params.Ny = Nxy;
params.Nx = Nxy;

params.Nz = length(params.z);
params.z0 = min(params.z);
params.dz = params.z(2)-params.z(1);


%% Generate training holo

data = zeros( data_num, Nxy, Nxy);
label = zeros( data_num, Nxy, Nxy, Nz);

N_random = randi([round(ppv_min*Nxy*Nxy) round(ppv_max*Nxy*Nxy)], 1, 1) % particle concentration
% N_random = 30;
% obj = randomScatter(Nxy, Nz, sr, N_random);   % randomly located particles
[obj, pos] = generate_particle_vol(Nxy, Nxy, Nz, params.pps, dz, N_random, sr);
 

% figure; show3d(obj, 0.0);  colorbar off;  axis image;


%%
otf3d_fresnel = ProjKernel(params);

%%  Fresnel
% transmistance function: t = exp(-a_obj)exp(-i phi), a_obj is absorption, t=(1-a_obj)exp(-j phi) = 1+t_obj
t_o = (1-obj);
[holo] = gaborHolo_fresnel(t_o, otf3d_fresnel,  noise_level);

figure; imagesc((holo)); title('Hologram'); axis image; drawnow; colormap(hot); colorbar; axis off;

imwrite(mat2gray(holo), ['C:/Users/CHENN/Desktop/',  'temp.png']);

%% Resize hologram

[holo, pps_new] = holoResize(holo, params.pps, new_size, 0);
[params.Ny, params.Nx] = size(holo);
params.pps = pps_new;

otf3d_fresnel = ProjKernel(params);
A2 = @(volume) (ForwardProjection(volume, otf3d_fresnel));
AT2 = @(plane) (BackwardProjection(plane, otf3d_fresnel));


% holo  = norm_std(norm_0_1((holo)));
holo = -holo;

% figure;
% temp  = (((holo)));
% subplot(121); imhist(temp); title('holo');
% temp = (real(AT2(holo)));
% subplot(122);imhist(temp);title('BP');


figure; imagesc(plotdatacube(obj)); title('Original Transfer function'); axis image; axis off;


scale_size = new_size/Nxy;
pos_new = [ceil(pos(:,1)*scale_size) ceil(pos(:,2)*scale_size) pos(:,3)];
obj_new = position2volume(pos_new, new_size, new_size, Nz, params.pps, dz, sr);

figure; imagesc(plotdatacube(obj_new)); title('Resized Transfer function'); axis image; axis off;



%%
% figure; imagesc(plotdatacube(obj_new)); title('Transfer function'); axis image; drawnow; colormap(hot); colorbar; axis off;
figure; imagesc((-holo)); title('Hologram'); axis image; drawnow; colormap(hot); colorbar; axis off;
temp = abs(real(AT2(holo))); %temp = (temp- min(temp(:)))./(max(temp(:))-min(temp(:))); 
% temp = norm_std((temp)); 
% temp = relu(real(AT2(holo)));
% temp = sigmoid(real(AT2(holo)));
temp = (temp);  %temp  = norm_0_1(temp);
figure; imagesc(plotdatacube(temp)); title('Gabor reconstruction: Real'); axis image;  axis off;
% temp = abs(AT2(holo));  %temp  = norm_0_1(temp);
% figure; imagesc(plotdatacube(temp)); title('Gabor reconstruction: Mag'); axis image; drawnow; colormap(hot); colorbar; axis off;

