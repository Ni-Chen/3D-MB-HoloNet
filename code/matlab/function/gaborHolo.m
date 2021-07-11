function [holo] = gaborHolo(obj, otf3d,  noise_level)
%----------------------------------------------
%
% Generate gabor hologram of 3D objects
%
% Input:   noise_type -> type of noise
%           - 'Gaussian'  level is PSNR
%           - 'Poisson'   level is average number of photons per pixel
%
% Outputs: obj   -> 3D object
%          otf3d   -> otf3d of the system
%          holo  -> blurred and noisholo data
%
    
    %% Hologram generation
    Nz = size(obj,3);
    holo_field = ForwardProjection(obj, otf3d);
    holo_noNoise = abs(holo_field).^2;
    
    ref_field = ForwardProjection(ones(size(obj,1),size(obj,2)), otf3d);
    ref_nonoise = abs(ref_field).^2;
    
    % Add noise
%     noise = max(holo_noNoise(:)).*10^(-noise_level/20).*random('Normal', zeros(size(holo_noNoise)), ones(size(holo_noNoise)));
%     holo = holo_noNoise + noise;
%     
%     noise = max(ref_nonoise(:)).*10^(-noise_level/20) .*random('Normal', zeros(size(ref_nonoise)), ones(size(ref_nonoise)));
%     ref = ref_nonoise + noise;
    
    
    holo = holo_noNoise;
    ref = ref_nonoise;
    
%     holo = awgn(holo_noNoise,noise_level,'measured');
%     ref = awgn(ref_nonoise,noise_level,'measured');
    
 
    holo = (holo - ref)./(sqrt(ref)+eps);
    
    holo = awgn(holo,noise_level,'measured');
    
    
end