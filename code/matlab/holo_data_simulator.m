close all; clear; clc;

addpath(genpath('./function/'));  % Add funtion path with sub-folders
data_dir = '../data/';


%% Modify
sr = 0;                % pixel size of half random point
data_num = 2000;       % number of train data_single

Nxy = 32;  Nz = 7;  dz = 1.2e-3;     % depth interval of the object slices

%%
objType = 'sim';
lambda = 660e-9;     % Illumination wavelength
pps    = 20e-6;      % pixel pitch of CCD camera
z0     = 5e-3;       % Distance between the hologram and the center plane of the 3D object

z_range = z0 + (0:Nz-1)*dz;   % axial depth span of the object

NA = pps*Nxy/2/z0;

delta_x = lambda/(NA)
delta_z = 2*lambda/(NA^2)

noise_level = 50;   % DB of the noise

group_num = 1;

ppv_min = 1e-3;
ppv_max = 5e-3;

if ppv_min == ppv_max
    ppv_text = [num2str(ppv_min,'%.e')];
else
    ppv_text = [num2str(ppv_min,'%.e') '~' num2str(ppv_max,'%.e')];
end

data_dir = [data_dir, objType, '_Nz', num2str(Nz), '_ppv', ppv_text, '_', num2str(noise_level), 'db', '_dz', num2str(dz*1e6),'um'];

params.lambda = lambda;
params.pps = pps;
params.z = z_range;
params.Ny = Nxy;
params.Nx = Nxy;

% [otf3d] = OTF3D(params);    % generate otf3d of the system
otf3d = ProjKernel(params);
%% Generate training data_single

data = zeros( data_num, Nxy, Nxy);
label = zeros( data_num, Nxy, Nxy, Nz);

figure;
for iData = 1:data_num
    disp([num2str(iData/data_num*100), '% is finished...']);
    
    N_random = randi([round(ppv_min*Nxy*Nxy*Nz) round(ppv_max*Nxy*Nxy*Nz)], 1, 1) % particle concentration
    obj = randomScatter(Nxy, Nz, sr, N_random);   % randomly located particles
    imagesc(plotdatacube(obj)); title(['3D object with particle of ' num2str(N_random) ]); axis image; drawnow; colormap(hot);
    
    
    t_o = (1-obj);
    [data_single] = gaborHolo(t_o, otf3d, noise_level);
    
    
    data(iData,:,:) = -data_single;
    label(iData,:,:,:) = obj;
    
end
save([data_dir, '.mat'], 'data', 'label', 'otf3d');



AT = @(plane) (BackwardProjection(plane, otf3d));


figure;
subplot(311); imagesc(plotdatacube(squeeze(label(iData,:,:,:)))); title('Last object'); axis image; drawnow; colormap(hot); colorbar; axis off;
subplot(312); imagesc(-squeeze(data(iData,:,:))); title('Hologram'); axis image; drawnow; colormap(hot); colorbar; axis off;
temp = abs(real(AT(squeeze(data(iData,:,:)))));  %temp = (temp- min(temp(:)))./(max(temp(:))-min(temp(:)));
subplot(313); imagesc(plotdatacube(temp)); title('Gabor reconstruction'); axis image; drawnow; colormap(hot); colorbar; axis off;