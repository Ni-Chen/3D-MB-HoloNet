function vol = position2volume(pos, Nx, Ny, Nz, dxy, dz, sr)

    if size(pos, 1) == 3 && size(pos, 2) == 3
        pos = pos';
    end

    pos = round(pos); % need improvement
    pos(pos(:,1)<1 | pos(:,2)<1, :) = NaN;
    pos(pos(:,1)>Ny | pos(:,2)>Nx, :) = NaN;

    if round(sr/dxy)<2
        vol = zeros(Ny, Nx, Nz);
        for idx = 1: size(pos,1)
            tmp = pos(idx,:);
            vol(sub2ind([Ny,Nx,Nz], tmp(1),tmp(2),tmp(3))) = 1;
        end
    else
        N = [Ny Nx Nz];
        pos(isnan(sum(pos,2)),:) = []; % remove NaNs
        pos = arrayfun(@(idx) min(max(1, pos(:, idx)), N(idx)), 1:3, 'un', 0);
        pos = cat(2, pos{:});

        X = ((1:Nx) - round((Nx + 1) / 2)) * dxy;
        Y = ((1:Ny) - round((Ny + 1) / 2)) * dxy;
        Z = ((1:Nz) - round((Nz + 1) / 2)) * dz;

        [x, y, z] = meshgrid(X, Y, Z);

        vol = zeros(Ny, Nx, Nz);
        for ip = 1:size(pos, 1)
            r2 = (x - X(pos(ip, 2))).^2 + (y - Y(pos(ip, 1))).^2 + (z - Z(pos(ip, 3))).^2;
            tmp = double(r2 <= sr^2);
            vol = vol + tmp;
        end
        vol(vol > 1) = 1; % dealing with overlapping
    end

end
