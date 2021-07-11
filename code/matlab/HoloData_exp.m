close all;
clear; clc;

format long;

addpath(genpath('./function/'));  % Add funtion path with sub-folders
data_dir = '../data/';

%%
objType = 'exp';  
data_num = 2000;       % number of training data


lambda = 660e-9;    % Illumination wavelength
pps    = 3.45e-6;   % pixel pitch of CCD camera
sr     = 25e-6;     % radius of particles

noise_level = 50;   % DB of the noise


%%
Nz = 3; z0 = 18.7e-3; dz = 1e-3;  ppv_min = 5e-5; ppv_max = 2e-4;  Nxy = 256; crop_size = Nxy; new_size = 128; %pps = pps*2
% Nz = 11; z0 = 15.7e-3; dz = 1e-3;  ppv_min = 1e-5; ppv_max = 5e-5; Nxy = 256; crop_size = 256; new_size = 128;
% Nz = 51; z0 = 10e-3; dz = 1e-3;  ppv_min = 0.5e-5; ppv_max = 5e-5; Nxy = 256; crop_size = Nxy; new_size = 128; 
% Nz = 49; z0 = 12e-3; dz = 1e-3;  ppv_min = 1e-4; ppv_max = 5e-4; Nxy = 128; crop_size = Nxy; new_size = 128; pps = pps*((1024)/128);

% Nz = 30; z0 = 12e-3; dz = 3e-3;  ppv_min = 1e-4; ppv_max = 5e-4; Nxy = 128; crop_size = 128; new_size = 128; sr = 50e-6; lambda = 632.8e-9;  pps =pps*(2048/128);


z_range = z0 + (0:Nz-1)*dz;   % axial depth span of the object


%%
delta_x = lambda/( pps*Nxy/2/z0)
delta_z = 2*lambda/(( pps*Nxy/2/z0)^2)

if ppv_min == ppv_max
    ppv_text = [num2str(ppv_min,'%.e')];
else
    ppv_text = [num2str(ppv_min,'%.e') '~' num2str(ppv_max,'%.e')];
end

params.lambda = lambda;
params.pps = pps;
params.z = z_range;
params.Ny = Nxy;
params.Nx = Nxy;

params.Nz = length(params.z);
params.z0 = min(params.z);
params.dz = params.z(2)-params.z(1);

data_dir = [data_dir, objType, '_Nz', num2str(Nz),'_ppv', ppv_text, '_',  num2str(noise_level), 'db', '_dz', num2str(dz*1e3),'mm'];


%% Generate training holo data
scale_size = new_size/crop_size;

data = zeros( data_num, new_size, new_size);
label = zeros( data_num, new_size, new_size, Nz);

otf3d_ori = ProjKernel(params);
figure;
for iData = 1:data_num
    disp([num2str(iData/data_num*100), '% is finished...']);
    
    N_random = randi([round(ppv_min*Nxy*Nxy*Nz) round(ppv_max*Nxy*Nxy*Nz)], 1, 1) % particle concentration
    
    [obj_ori, pos] = generate_particle_vol(Nxy, Nxy, Nz, params.pps, dz, N_random, sr);
    
    %%  Fresnel
    % transmistance function: t = exp(-a_obj)exp(-i phi), a_obj is absorption, t=(1-a_obj)exp(-j phi) = 1+t_obj
    t_o = (1-obj_ori);
    [holo_ori] = gaborHolo(t_o, otf3d_ori, noise_level);
    
    
    %% Resize hologram
    if crop_size ~= new_size
        holo_ori = imcrop(holo_ori,  [1 1 crop_size-1 crop_size-1]);
        [holo, pps_new] = holoResize(holo_ori, params.pps, new_size, 0);
        
        
        %% Resize label
        pos_new = [ceil(pos(:,1)*scale_size) ceil(pos(:,2)*scale_size) pos(:,3)];
        obj = position2volume(pos_new, new_size, new_size, Nz, pps_new, dz, sr);
    else
        holo = holo_ori;
        pps_new = params.pps;
        obj = obj_ori;
    end
    
    data(iData,:,:) = -holo;
    label(iData,:,:,:) = obj;
    
    temp = -holo;
%     ['data: max:' num2str(max(temp(:))) ', min:' num2str(min(temp(:))) ', mean:' num2str(mean(temp(:))) ', std:' num2str(std(temp(:)))]

    %%
    imagesc(plotdatacube(obj)); title('Resized Transfer function'); axis image; drawnow; colormap(hot); colorbar; axis off;
    
end

['data: max:' num2str(max(data(:))) ', min:' num2str(min(data(:))) ', mean:' num2str(mean(data(:))) ', std:' num2str(std(data(:)))]


[params.Ny, params.Nx] = size(holo);
params.pps = pps_new;
otf3d = ProjKernel(params);

save([data_dir, '.mat'], 'data', 'label', 'otf3d');

%% display
A2 = @(volume) (ForwardProjection(volume, otf3d));
AT2 = @(plane) (BackwardProjection(plane, otf3d));

% figure; imagesc(plotdatacube(obj_ori(1:crop_size, 1:crop_size,: ))); title('Original Transfer function'); axis image; drawnow; colormap(hot); colorbar; axis off;
% figure; imagesc(plotdatacube(obj)); title('Resized Transfer function'); axis image; drawnow; colormap(hot); colorbar; axis off; grid on;

figure; imagesc((holo_ori)); title('Ori hologram'); axis image; drawnow; colormap(hot); colorbar; axis off;
% figure; imagesc((holo)); title('Resized hologram'); axis image; drawnow; colormap(hot); colorbar; axis off;

figure; show3d(abs(real(AT2(squeeze(data(iData,:,:))))), 0.); colorbar; axis equal;

figure;
subplot(311); imagesc(plotdatacube(squeeze(label(iData,:,:,:)))); title('Last object'); axis image; drawnow; colormap(hot); colorbar; axis off;
subplot(312); imagesc(squeeze(-data(iData,:,:))); title('Hologram'); axis image; drawnow; colormap(hot); colorbar; axis off;
subplot(313); imagesc(plotdatacube(abs(real(AT2(squeeze(data(iData,:,:))))))); title('Gabor reconstruction'); axis image; drawnow; colormap(hot); colorbar; axis off;

