function [otf] = ProjKernel(params)
    pph = params.pps;
    lambda = params.lambda;
    z_list = params.z;
    Ny = params.Ny;
    Nx = params.Nx;


    % Constant frequencies
    % write it this way to avoid fftshifts
    [X, Y] = meshgrid(0:(Nx - 1), 0:(Ny - 1));
    fx = (mod(X + Nx / 2, Nx) - floor(Nx / 2)) / Nx;
    fy = (mod(Y + Ny / 2, Ny) - floor(Ny / 2)) / Ny;
    
    term = (fx.^2 + fy.^2) * (lambda/pph)^2;

%     % rigorous
%     sqrt_input = 1 - term;
%     sqrt_input(sqrt_input < 0) = 0;
%     final_term = sqrt(sqrt_input);

    % Fresnel expansion: (1+x)^(1/2) = 1 + a * x + a*(a-1)/2! * x^2
    final_term = - 1/2*term - 1/8*term.^2 - 1/16*term.^3;

%     planewave_phase = ones(Ny, Nx) .* permute(exp(-1i * 2 * pi .* z_list / lambda), [3 1 2]);

    % Make sure the sign is correct
    otf = arrayfun(@(idx) exp(1i*2*pi/lambda*z_list(idx) * final_term), 1:length(z_list), 'un', 0);   
    otf = cat(3, otf{:});

end