%% Depth resolution of hologram reconstruction

close all; clear; clc;

addpath(genpath('./function/'));  % Add funtion path with sub-folders

paper_dir = '../../papers/Optica/figure/';

holoDataType = 1;
objType = 'particle';  % 'particle', 'lines'

%%
noise_type  = 'Gaussian';
point_size = 0;      % pixel size of one random point
data_num = 1;        % number of train data

Nxy = 32;              % lateral size
Nz = 3;               % axial size

lambda = 633e-9;       % Illumination wavelength
pps    = 20e-6;         % pixel pitch of CCD camera
z0     = 10e-3;        % Distance between the hologram and the center plane of the 3D object


% Resolution calculation
NA = pps*Nxy/2/z0

delta_x = lambda/(NA)
delta_z = 2*lambda/(NA^2)


%%
dz     = delta_z;         % depth interval of the object slices
z_range = z0 + ((1:Nz)-round(Nz/2))*dz;   % axial depth span of the object

% fixed ppvs
noise_level = 30;   % DB of the noise

ppv_min = 5e-3;
ppv_max = ppv_min;

params.lambda = lambda;
params.pp_holo = pps;
params.z = z_range;
params.Ny = Nxy;
params.Nx = Nxy;

%% Generate  data
N_random = randi([round(ppv_min*Nxy*Nxy*Nz) round(ppv_max*Nxy*Nxy*Nz)], 1, 1) % particle concentration
obj = randomScatter(Nxy, Nz, N_random, point_size);   % randomly located particles


%%
figure;
subplot(411); imagesc(plotdatacube(obj)); title('Object'); axis image; drawnow; colormap(hot); axis off;


%% dz = depth resolution
[otf3d, psf3d, pupil] = OTF3D(Nxy, Nxy, lambda, pps, z_range);    % generate otf3d of the system
obj_norm = (obj);
[label, holo0] = gaborHolo(obj_norm, otf3d, noise_type, noise_level);
temp0 = real(iMatProp3D(holo0, otf3d)); 

% subplot(423); imagesc((holo0));  axis image; drawnow; colormap(hot); colorbar; caxis([min(holo0(:)) max(holo0(:))]);axis off;
subplot(412); imagesc(plotdatacube(temp0)); title(['Reconstruction: \Delta z = \delta z']); axis image; drawnow; colormap(hot);  axis off;

%% dz = 0.5 x depth resolution
dz     = delta_z/2;         % depth interval of the object slices
z_range = z0 + ((1:Nz)-round(Nz/2))*dz;   % axial depth span of the object
params.z = z_range;

[otf3d, psf3d, pupil] = OTF3D(Nxy, Nxy, lambda, pps, z_range);    % generate otf3d of the system
obj_norm = (obj);
[label, holo1] = gaborHolo(obj_norm, otf3d, noise_type, noise_level);
temp1 = real(iMatProp3D(holo1, otf3d)); 

% subplot(425); imagesc((holo1));  axis image; drawnow; colormap(hot); colorbar; caxis([min(holo1(:)) max(holo1(:))]);axis off;
subplot(413); imagesc(plotdatacube(temp1)); title(['Reconstruction: \Delta z =0.5 \times \delta z']); axis image; drawnow; colormap(hot); axis off;

%% dz = 0.25 x depth resolution
dz     = delta_z/4;         % depth interval of the object slices
z_range = z0 + ((1:Nz)-round(Nz/2))*dz;   % axial depth span of the object
params.z = z_range;

[otf3d, psf3d, pupil] = OTF3D(Nxy, Nxy, lambda, pps, z_range);    % generate otf3d of the system
obj_norm = (obj);
[label, holo2] = gaborHolo(obj_norm, otf3d, noise_type, noise_level);
temp2 = real(iMatProp3D(holo2, otf3d)); 

% subplot(427); imagesc((holo2));  axis image; drawnow; colormap(hot); colorbar; caxis([min(holo2(:)) max(holo2(:))]);axis off;
subplot(414); imagesc(plotdatacube(temp2)); title(['Reconstruction: \Delta z = 0.25 \times\delta z']); axis image; drawnow; colormap(hot); axis off;



export_fig([paper_dir, 'res_conv.eps'], '-transparent');