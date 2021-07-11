function spatialcube = plotdatacube(data)
    data = padarray(data, [1 1], max(data(:)), 'both');

    [Ny, Nx, Nz] = size(data);
    
    if Nz >10
        cols = ceil(sqrt(Nz));
    else
        cols = Nz;
    end
    
    rows = ceil(Nz/cols);

    spatialcube = zeros(rows*Ny, cols*Nx);

    figscount = 1;
    for r = 1:rows
        for c = 1:cols
            if figscount <= Nz
                spatialcube((r-1)*Ny+1:(r-1)*Ny+Ny, (c-1)*Nx+1:(c-1)*Nx+Nx) = squeeze(data(:, :, figscount)); 
            else
                spatialcube((r-1)*Ny+1:(r-1)*Ny+Ny, (c-1)*Nx+1:(c-1)*Nx+Nx) = zeros(Ny, Nx);
            end
            figscount = figscount+1;
        end
    end

end