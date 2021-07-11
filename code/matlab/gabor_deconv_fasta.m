% Gabor hologram reconstruction with optimization

clc; clear; 
% close all;


paper_dir = '../../papers/OE/figure/';
%% Initialization
% addpath(genpath('./function/'));  % Add funtion path with sub-folders
addpath(genpath('./function/'));  % Add funtion path with sub-folders
% indir = '../data/test_data_particle/';  mu = 0.4;   tv_mu = 0.001;    % old data
% indir = '../data/nanowire/';  mu = 0.5;   tv_mu = 0.00001;
indir = '../data/20200806/';  mu = 3.8;   tv_mu = 0.000001;  obj_name = 'exp3_'
% indir = '../data/x_shift/';  mu = 0.2;   tv_mu = 0.00001;
% indir = '../data/particle/';   mu = 0.5;
% indir = '../data/particle_flow/';   mu = 0.6;
outdir = indir;  % Output files

%% Hologram Generation / loading (experimental) 
% run([indir, 'param.m']);  % Parameters of the object and hologram 
load([indir, 'holo_data.mat']);

% mu = 0.8;   tv_mu = 0.00001;    
% mu = 0.05;   tv_mu = 0.001;    % old data
x = ((1:params.Nx) - round(params.Nx/2))*params.pps; 
y = ((1:params.Ny) - round(params.Ny/2))*params.pps; 

iter = 1000;    % iterations

plot_ratio = (params.Nz*params.dz)/(params.Nx*params.pps);

      
[otf] = ProjKernel(params);   %
A = @(volume) real(ForwardProjection(volume, otf));
AT = @(holo) (real(BackwardProjection(holo, otf)));


%% FASTA solution
opts = [];
opts.tol = 1e-6;              % Use super strict tolerance
opts.recordObjective = true;  %  Record the objective function so we can plot it
opts.verbose = 2;
opts.stringHeader=' ';        % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer. 
opts.adaptive = true;
opts.stopRule = 'iterations';
opts.maxIters = iter;
opts.accelerate = true;       % Turn on FISTA-type acceleration
opts.plot_steps = false;

method = 'FUSED_LASSO2D';

deconv_recon = zeros(size(data,1), params.Ny,params.Nx,params.Nz);
for idata = 1:size(data,1)
%     holo = squeeze(data(:,:,idata));
    holo = squeeze(data(idata,:,:));
    holo = gpuArray(holo);
   
    switch method
        case 'L1'
            tic
            [reobj_fasta, outs_accel] = fasta_sparseLeastSquares(A, AT, holo, mu, (AT(holo)), opts);
            
            out_file_name = ['L1_'];
        case 'TV'
            opts.TV_subproblem_its = 5;

            [reobj_fasta, outs_accel] = fasta_fusedLasso(A, AT, holo, 0, tv_mu, AT(holo), opts);
            out_file_name = [ 'TV_'];
        case 'TV2D'
            opts.plot_steps = true;
            opts.TV_subproblem_its = 5;

            [reobj_fasta, outs_accel] = fasta_fusedLasso2D(A, AT, holo, 0, tv_mu, AT(holo), opts);

            out_file_name = [ 'TV2D_'];
        case 'FUSED_LASSO'
            opts.TV_subproblem_its = 5;

            [reobj_fasta, outs_accel] = fasta_fusedLasso(A, AT, holo, mu, tv_mu, AT(holo), opts);
            out_file_name = [ 'FUSEDLASSO_NonNeg'];

        case 'FUSED_LASSO2D'
            opts.TV_subproblem_its = 5;
            [reobj_fasta, outs_accel] = fasta_fusedLasso2D(A, AT, holo, mu, tv_mu, AT(holo), opts);
            out_file_name = [ 'FUSEDLASSO2D_'];
    end

%     figure; semilogy(outs_accel.objective); title('Objective');
%     export_fig([outdir, out_file_name, 'loss_iter', num2str(iter), '_obj', num2str(outs_accel.objective(end)), '.png']);

    reobj_fasta = gather(reobj_fasta);
    
    tmp = abs(reobj_fasta);  
    reobj_fasta= (tmp-min(tmp(:)))./(max(tmp(:))-min(tmp(:))); 
    deconv_recon(idata, :,:,:) = reobj_fasta;
    
%     figure; imagesc(plotdatacube(reobj_fasta)); axis image; drawnow; colormap(hot); colorbar; axis off;
    
    figure; show3d(reobj_fasta, 0.0); view(-90, 0); colorbar off;  axis image;
    set_ticks(reobj_fasta, ((1:params.Nx) - round(params.Nx/2))*params.pps*1e3, ((1:params.Ny) - round(params.Ny/2))*params.pps*1e3, params.z*1e3, 'mm');
    set(gca,'DataAspectRatio',[plot_ratio/3 plot_ratio/3 1]);
    
    export_fig([paper_dir, obj_name, num2str(idata) ,  '_fasta.png'], '-transparent');

end

 save([paper_dir, obj_name 'fasta_recons.mat'], 'deconv_recon');
    
 
 
 
function set_ticks(vol, x_tick, y_tick, z_tick, tick_unit)

    xticks([1 size(vol,2)/2 size(vol,2)]);
    yticks([1 size(vol,1)/2 size(vol,1)]);
    zticks([1 size(vol,3)/2 size(vol,3)]);
    
%     x_ticks = round(min(x_tick),1):(max(x_tick)-min(x_tick))/5:round(max(x_tick),1);
%     set(gca,'xticklabels', round(x_ticks,1));
%     y_ticks = round(min(y_tick),1):(max(y_tick)-min(y_tick))/5:round(max(y_tick),1);
%     set(gca,'yticklabels', round(y_ticks,1));
%     z_ticks = round(min(z_tick),1):(max(z_tick)-min(z_tick))/8:round(max(z_tick),1);
%     set(gca,'zticklabels',round(z_ticks,1));
    
    x_ticks = {floor(min(x_tick)*10)/10, 0, floor(max(x_tick)*10)/10};
    set(gca,'xticklabels', (x_ticks));
    y_ticks = {floor(min(y_tick)*10)/10, 0, floor(max(y_tick)*10)/10};
    set(gca,'yticklabels', (y_ticks));
    z_ticks = {floor(min(z_tick)),  floor((min(z_tick) + (max(z_tick)-min(z_tick))/2) *10)/10, floor(max(z_tick)*10)/10};
    set(gca,'zticklabels',(z_ticks));
    
    xlabel(['x (', tick_unit, ')'], 'fontsize', 14);
    ylabel(['y (', tick_unit, ')'], 'fontsize', 14);
    zlabel(['z (', tick_unit, ')'], 'fontsize', 14);
    
    xlim([1 size(vol,2)])
    ylim([1 size(vol,1)])
    zlim([1 size(vol,3)])
end
