function ortho = orthoView(vol)
    orthoview_xy = @(vol) squeeze(vol(:, :, round(size(vol,3)/2)));
    orthoview_xz = @(vol) rot90(squeeze(vol(round(size(vol,1)/2), :, :)));
    orthoview_yz = @(vol) squeeze(vol(:, round(size(vol,2)/2), :));

    % Concat xy, xz, yz images together
    ortho = [orthoview_yz(vol) orthoview_xy(vol); zeros(size(vol,3), size(vol,3)) orthoview_xz(vol)];
end