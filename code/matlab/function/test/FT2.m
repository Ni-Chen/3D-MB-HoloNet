function x_ft = FT2(x)

%    x_ft = ifftshift(fft2(fftshift(x)))/length(x(:));
   
%    
    [Ny, Nx] = size(x);
%     if(mod(Nx,2)==0)
%         % even number
%         x_ft = ifftshift(fft2(fftshift(x)));
%     else    
%         x_ft = fftshift(fft2(x)); 
%     end

    x_ft = ifftshift(fft2(fftshift(x)));   % shift 2 pixels
    
end