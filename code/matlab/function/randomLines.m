function obj3d = randomLines(Nxy, Nz)
%{
------------------------------------------------
Generates circluar helix object

Inputs: 
    sr -> radius of one single scatter

Example: 
    im = randomScatter(128,128, 20, 1);

Copyright (C) 2019, Ni Chen, nichen@snu.ac.kr
------------------------------------------------
%}
%     rng(0);
%     obj3d = zeros(Nxy, Nxy, Nz);
%     for ip = 1:Np
%         ix = ceil(rand(1)*Nxy);
%         iy = ceil(rand(1)*Nxy);
%         iz = ceil(rand(1)*Nz);
% 
%         nd = randi([1,2], 1, 1);
%         line_len = randi([2,15], 1, 1);
%         if nd == 1
%             iy_range = iy:iy+line_len;
%             ix_range = ix:ix+sr;
%         else
%             ix_range = ix:ix+line_len;
%             iy_range = iy:iy+sr;
%         end
%         
%         iz_range = iz:iz+sr;
% %         iz_range = 2;
% 
%         ix_range(ix_range>Nxy) = Nxy;
%         iy_range(iy_range>Nxy) = Nxy;
%         iz_range(iz_range>Nz) = Nz;
% 
%         obj3d(iy_range, ix_range, iz_range) = 1;     % rand(1)
%     end
    
    obj3d = helix(Nxy,Nxy,Nz);
    
end

function obj = helix(Nx,Ny,Nz)

a = 0.8;
c = 0.6;
t = 0:0.005:1*pi;
% x = (a*t/2*pi*c).*sin(t);
% y = (a*t/2*pi*c).*cos(t);
% z = t/(2*pi*c);

% figure;
% plot3(x, y, z); 
% xlabel('x'); ylabel('y'); title('Circula helix');

%%
Nx = 128;
Ny = Nx;
Nz = 31;
obj = zeros(Ny,Nx,Nz);

ix = 0.025*1;
iy = ix;
iz = 0.035*1;

xx = round(a*t.*sin(t)/(2*pi*c)/ix) + Nx/2;
yy = round(a*t.*cos(t)/(2*pi*c)/iy) + Ny/2;
zz = round(t/(2*pi*c)/iz) + 1;

xyz = sub2ind(size(obj),xx,yy,zz);

obj(xyz) = 1;

% figure;
% temp = abs(permute((obj), [2 3 1])); 
% show3d(temp, 0.01, 'hot');



end

