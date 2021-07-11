function imwrite3D(vol, filename, zrange)
    vol = vol./max(vol(:));
    for slice_idx = 1: size(vol,3)
        slice_img = vol(:,:, slice_idx);
        slice_img = uint8(255 * mat2gray(slice_img));    
        rgbImage = ind2rgb(slice_img, inferno);
        imwrite(rgbImage, [filename num2str(slice_idx) '_' num2str(zrange(slice_idx)) 'um.png']);
    end
end