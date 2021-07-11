function show3dcmp(obj_vol1, obj_vol2,  alpham)
%     obj_vol1 = permute(obj_vol1, [2 3 1]);  % Complex images
    [Ny, Nx, Nz] = size(obj_vol1);
    
    axes('fontsize', 14, 'xtick', 0:floor(Nx/4):Nx, 'ytick', 0:floor(Ny/4):Ny, 'ztick', 0:floor(Nz/4):Nz);
    vol3d('CData', obj_vol1, 'texture', '3D');
    view(-25, 20);

    colormap('hot');
    colormap(1 - colormap);
    colorbar;
    
    hold on;
    vol3d('CData', obj_vol2, 'texture', '3D');
    view(-25, 20);
    
    colormap('summer');
    colormap(1 - colormap);
    colorbar;
    
    
    xlabel('x', 'fontsize', 14);
    ylabel('y', 'fontsize', 14);
    zlabel('z', 'fontsize', 14);
    
    axis([0 Nx 0 Ny 0 Nz]);
    axis tight;
   
    box on,
    ax = gca;
    ax.BoxStyle = 'full';
    
    grid on; 
    grid minor;
    % zoom(0.7)
    alphamap('decrease', alpham); 
end