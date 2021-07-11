%{
Simulate forward formation of a hologram from a 3D optical field

Inputs:
  volume  : 3D estimated optical field, generally complex valued
  params  : Input parameters structure. Must contain the following
      - Nx, Ny, nz  : Size of the volume (in voxels) in x, y, and z
      - z_list      : List of reconstruction planes (um)
      - pp_holo     : Pixel size (um) of the image
      - wavelength  : Illumination wavelength (um)

Outputs:
  holo    : Ny-by-Nx 2D real hologram (estimated image)
%}

function holo = ForwardProjection(volume, otf)

    Fholo3d = (fft2(volume) .* otf);
    
    Fholo = sum(Fholo3d, 3); 
    holo = ifft2(Fholo);
    
end