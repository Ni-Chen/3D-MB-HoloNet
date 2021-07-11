function x_ift = iFT2(x)
    [Ny, Nx] = size(x);

%    x_ift = ifftshift(ifft2(fftshift(x))).*length(x(:));

%    x_ift = ifftshift(ifft2(fftshift(x)));   % will shift 2 pixels

%     x_ift = fftshift(ifft2(fftshift(x)));   % will not pixels
    
%     x_ift = fftshift(ifft2(ifftshift(x)));
        
%     if(mod(Nx,2)==0)
%         % even number
%         x_ift = fftshift(ifft2(ifftshift(x))); 
%     else    
%         x_ift = ifft2(ifftshift(x));
%     end

    x_ift = fftshift(ifft2(ifftshift(x)));   
    
end