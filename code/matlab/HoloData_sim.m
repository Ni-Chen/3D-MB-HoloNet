close all; clear; clc;

addpath(genpath('./function/'));  % Add funtion path with sub-folders
data_dir = '../data/';


%% Modify
sr = 0;                % pixel size of half random point
data_num = 2000;       % number of train data_single

holoDataType = 4;

Nxy = 32;  Nz = 7;  dz = 1.2e-3;     % depth interval of the object slices
% Nxy = 64;  Nz = 32;  dz = 0.725e-3;       % depth interval of the object slices


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
switch holoDataType
    case 1
        if Nz==3
            ppv_min = 1e-3;
            ppv_max = 5e-3;            
        elseif Nz==7
            ppv_min = 1e-3;
            ppv_max = 5e-3;
        elseif Nz==15
            ppv_min = 5e-4;
            ppv_max = 25e-4;
        elseif Nz==32
            ppv_min = 2e-4;
            ppv_max = 1e-3;
        end
        
        if ppv_min == ppv_max
            ppv_text = [num2str(ppv_min,'%.e')];
        else
            ppv_text = [num2str(ppv_min,'%.e') '~' num2str(ppv_max,'%.e')];
        end
        
        data_dir = [data_dir, objType, '_Nz', num2str(Nz), '_ppv', ppv_text, '_', num2str(noise_level), 'db', '_dz', num2str(dz*1e6),'um'];
       
    case 2
        % Test data_single with varying noise level
        noise_levels = [10 15 20 25 30 35 40 45 50];
        group_num = length(noise_levels);
        data_num = 100;
        
        ppv_min = 1e-3;
        ppv_max = 5e-3;
        
        if ppv_min == ppv_max
            ppv_text = num2str(ppv_min,'%.e');
        else
            ppv_text = [num2str(ppv_min,'%.e') '~' num2str(ppv_max,'%.e')];
        end        
        
        data_dir = [data_dir, 'test_noise_', objType, '_Nz', num2str(Nz), '_ppv', ppv_text, '_dz', num2str(dz*1e6),'um'];
        
    case 3
        % Test data_single with varying ppvs
        ppvs = [1 2 3 4 5 6 7 8 9 10]*1e-3;
        group_num = length(ppvs);
        
        data_num = 100;         % number of test data_single
         
        data_dir = [data_dir, 'test_ppv_', objType, '_Nz', num2str(Nz), '_', num2str(noise_level), 'db',  '_dz', num2str(dz*1e6),'um'];
        
    case 4 % Comparison
        Nxy = 64;              % lateral size
        Nz = 64;                % axial size
        
           
        lambda = 632e-9;       % Illumination wavelength
        pps    = 20e-6;
        dz     = 50e-6;         % depth interval of the object slices
        
        
        z0 = 10e-3

        z_range = z0 + (0:Nz-1)*dz;   % axial depth span of the object
        
        NA = pps*Nxy/2/z0
        
        delta_x = lambda/(NA)
        delta_z = 2*lambda/(NA^2)
        
        
%         ppv_min = 1.9e-4*128*128/Nxy/Nxy/Nz;  
%         ppv_max = 6.1e-2*128*128/Nxy/Nxy/Nz;  
        
        ppv_min = 1.9e-4/128;  
        ppv_max = 6.1e-2/128;
%         ppv_max = ppv_min;  
        
        
        if ppv_min == ppv_max
            ppv_text = [num2str(ppv_min,'%.1e')];
        else
            ppv_text = [num2str(ppv_min,'%.1e') '~' num2str(ppv_max,'%.1e')];
        end
        
        data_dir = [data_dir, objType, '_Nz', num2str(Nz),'_ppv', ppv_text, '_',  num2str(noise_level), 'db', '_dz', num2str(dz*1e6),'um'];
        
end

params.lambda = lambda;
params.pps = pps;
params.z = z_range;
params.Ny = Nxy;
params.Nx = Nxy;

% [otf3d] = OTF3D(params);    % generate otf3d of the system
otf3d = ProjKernel(params);
%% Generate training data_single
if group_num>1
    data = zeros(group_num, data_num, Nxy, Nxy);
    label = zeros(group_num, data_num, Nxy, Nxy, Nz);

    for idx = 1:group_num
        if exist('noise_levels', 'var')
            noise_level = noise_levels(idx);
        end
        if exist('ppvs', 'var')
            ppv = ppvs(idx);
            ppv_min = ppv;
            ppv_max = ppv;
        end

        figure;
        for iData = 1:data_num
            disp([num2str(iData/data_num*100), '% is finished...']);

            N_random = randi([round(ppv_min*Nxy*Nxy*Nz) round(ppv_max*Nxy*Nxy*Nz)], 1, 1) % particle concentration
            obj = randomScatter(Nxy, Nz, sr, N_random);   % randomly located particles

            imagesc(plotdatacube(obj)); title(['3D object with particle of ' ]); axis image; drawnow; colormap(hot);
            
            t_o = (1-obj);
            [data_single] = gaborHolo(t_o, otf3d, noise_level);

            data(idx, iData,:,:) = -data_single;
            label(idx, iData,:,:,:) = obj;

        end
    end
    save([data_dir, '.mat'], 'data', 'label', 'otf3d');
else
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
end


AT = @(plane) (BackwardProjection(plane, otf3d));


figure;
subplot(311); imagesc(plotdatacube(squeeze(label(iData,:,:,:)))); title('Last object'); axis image; drawnow; colormap(hot); colorbar; axis off;
subplot(312); imagesc(-squeeze(data(iData,:,:))); title('Hologram'); axis image; drawnow; colormap(hot); colorbar; axis off;
temp = abs(real(AT(squeeze(data(iData,:,:)))));  %temp = (temp- min(temp(:)))./(max(temp(:))-min(temp(:)));
subplot(313); imagesc(plotdatacube(temp)); title('Gabor reconstruction'); axis image; drawnow; colormap(hot); colorbar; axis off;