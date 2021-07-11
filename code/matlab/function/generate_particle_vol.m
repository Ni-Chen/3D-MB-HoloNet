function [vol, pos] = generate_particle_vol(Nx, Ny, Nz, dxy, dz, Np, sr)
 
    N_pad = 5;
    pad_size = ([N_pad N_pad 0]);
    Nxy_pad = Nx - 2*N_pad;
    
    Np_z = 1;
    for iz =1:Nz
        ix = randi(Nxy_pad, 1, 1);
        iy = randi(Nxy_pad, 1, 1);
        
        pos(iz,:) = [iy ix iz];
    end
    Np_left = Np - Np_z*Nz;
    for ip = 1:Np_left
        ix = randi(Nxy_pad, 1, 1);
        iy = randi(Nxy_pad, 1, 1);
        iz = randi(Nz, 1, 1);
        
        pos(Nz+ip,:) = [iy ix iz];
        
    end       

    pos = round(pos + pad_size);    
        
    % reject same locations
    [pos, ~, ~] = unique(pos(:, 1:3), 'rows');
    
    % reject neighboring particle overlapping
    t = sum(squareform(pdist(pos)) < 2 * sr, 1) - 1;
    pos(logical(t(1:round(size(pos, 1) / 2))), :) = [];
  
    pos(pos(:,1)<1 | pos(:,2)<1, :) = NaN;    
    pos(isnan(sum(pos,2)),:) = []; % remove NaNs

    vol = position2volume(pos, Nx, Ny, Nz, dxy, dz, sr);

end


