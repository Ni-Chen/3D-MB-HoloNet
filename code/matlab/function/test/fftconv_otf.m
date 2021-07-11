
function conv_out = fftconv_otf(x, H)
% 2D linear convolution. out = ifft2(fft2(x).*H)
    [Nx1, Nx2]= size(x);
    [NH1, NH2] = size(H);
    
    N1 = Nx1 + NH1 -1;
    N2 = Nx2 + NH2 -1;
    
    x_pad = [x zeros(Nx1, NH2-1); zeros(NH1-1, Nx2 + NH2-1)]; 
    H_pad = [H zeros(NH1, Nx2-1); zeros(Nx1-1, NH2 + Nx2-1)]; 
    
%     conv_out = ifft2(fft2(x, N1, N2).*padarray(H, [floor((N1-NH1)/2), floor((N1-NH1)/2)], 0, 'both'));
    
    conv_out = ifft2(fft2(x_pad).*H_pad);
    
    % padding constants (for output of size == size(A))
    pad1 = ceil((NH1-1)./2);
    pad2 = ceil((NH2-1)./2);
    
    % frequency-domain convolution result
    conv_out = conv_out(pad1+1:Nx1+pad1, pad2+1:Nx2+pad2); 
end