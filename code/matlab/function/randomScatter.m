function vol = randomScatter(Nxy, Nz, sr, Np)
%{
------------------------------------------------
Generates randomly distributed particle volume

Inputs:
    sr -> radius of one single scatter

Example:
    im = randomScatter(128, 128, 20, 1);

Copyright (C) 2019, Ni Chen, nichen@snu.ac.kr
------------------------------------------------
%}

%     rng(0);    


    vol = zeros(Nxy, Nxy, Nz);

    N_pad = 1;
    Nxy_pad = Nxy - 2*N_pad;


    for iz =1:Nz
        ix = randi(Nxy_pad, 1, 1);
        iy = randi(Nxy_pad, 1, 1);
        
        pos(iz,:) = [iy ix iz];
    end
    Np_left = Np - Nz;
    for ip = 1:Np_left
        ix = randi(Nxy_pad, 1, 1);
        iy = randi(Nxy_pad, 1, 1);
        iz = randi(Nz, 1, 1);
        
        pos(Nz+ip,:) = [iy ix iz];        
    end
    
    pos = (pos + [N_pad N_pad 0]);         
    
    pos(pos(:,1)<1 | pos(:,2)<1 | pos(:,3)<1 | pos(:,1)>Nxy | pos(:,2)>Nxy | pos(:,3)>Nz, :) = NaN;    
    pos(isnan(sum(pos,2)),:) = []; % remove NaNs      
    
    vol = zeros(Nxy, Nxy, Nz);
    for idx = 1: size(pos,1)
        tmp = pos(idx,:);
        vol(sub2ind([Nxy,Nxy,Nz], tmp(1),tmp(2),tmp(3))) = 1;
    end


end



