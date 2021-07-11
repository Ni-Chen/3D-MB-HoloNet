%{ 
Propagate 2D real field to 3D volume

Inputs:
  holo    : ny-by-nx 2D real data (often the residual hologram)
  params  : Input parameters structure. Must contain the following
      nx, ny, nz  : Size of the volume (in voxels) in x, y, and z
      z_list      : List of reconstruction planes (um)
      pp_holo     : Pixel size (um) of the image
      wavelength  : Illumination wavelength (um)

Outputs:
  volume  : 3D estimated optical field, complex valued

%}


function volume = BackwardProjection(holo, otf) 

    holo = real(holo);   % For adjoint check
    
    volume = ifft2(fft2(holo) .* conj(otf));
    
%     volume = real(volume);
end
