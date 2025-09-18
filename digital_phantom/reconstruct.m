clear;
%% Magnetic Resonance Fingerprinting reconstruction code

%% Initialize paths, check for bart toolbox and load data 

% Required: BART (installed in WSL)
bart_loc = '/mnt/c/Users/lucya/MSC_PROJECT/bart/';

% matlab folder of BART contains a script to call BART in WSL
addpath(genpath('C:/Users/lucya/MSC_PROJECT/bart/matlab/'));
setenv('TOOLBOX_PATH',bart_loc);

% check if bart toolbox is installed
if ~isempty(which('bart')); else;disp('>> BART toolbox not found in path. Please check BART path.');end

% add the "utils" subfolder to the PATH:
addpath(genpath('C:/Users/lucya/MSC_PROJECT/MatlabCode_Phantom/CodeMATLAB/utils'));

% Load data  
folder = 'C:/Users/lucya/MSC_PROJECT/DigitalPhantom';
load(fullfile(folder, 'komamri_signal.mat'));

Nkx = 248;
Nky = 1000;
Nc = 1;
dyn = 1;
neco = 1;
slices = 1;

raw = reshape(signal, [Nkx, Nc, Nky, dyn]);

% Data preprocessing
U.RawKspaceData=squeeze(raw(:,:,:,1:dyn));
kdim = c12d(size(U.RawKspaceData));

% Load trajectory
load(fullfile(folder, 'komamri_traj.mat'));  

trajGIRF = zeros(3, 248, 1000);

trajGIRF(1,:,:) = traj(3,:,:);  % kz
trajGIRF(2,:,:) = traj(2,:,:);  % ky
trajGIRF(3,:,:) = 0;

trajGIRF        = repmat(trajGIRF,[1 1 1 1 kdim(6)]);
dcf = radial_density(trajGIRF);
N = 124;
comb_raw = zeros(kdim(1),Nc,kdim(2)*dyn);
location=0;
step=1000;

U.RawKspaceData= permute(U.RawKspaceData,[1 4 2 3]); 
for d=1:dyn
    comb_raw(:,:,location+1:step)= squeeze(U.RawKspaceData(:,:,:,d));
    location=location+1000;
    step=step+1000;
end

U.RawKspaceData= permute(comb_raw,[1 3 4 2]);

% think these do same thing
csm = ones(N, N, 1, 1);  % ideal single coil

% nufft = bart(['nufft -a -d ',num2str(N),':',num2str(N),':1'],trajGIRF,permute(bsxfun(@times,dcf,U.RawKspaceData),[3 1 2 4])); %Phantom 0.25
% imgForCSM = bart('fft 3',nufft);
% csm   = bart('ecalib -m1',imgForCSM); 


%% Dictionary 

% loading dictionary from MATLAB file (5865 entries):
dict_folder = 'C:/Users/lucya/MSC_PROJECT/Dictionary/dict';
load(fullfile(dict_folder, 'blochdict_7mm_71_short.mat'));

dict0 = dict0(1:1000*dyn,:);    % ... First dyn*1000 time points
[~,s,~]=svd((dict0),'econ');    % ... SVD: captures dominant signal patterns
result_s    = diag(s);
NRJ= zeros(100,1);
for R = 1:100    % ... Energy retained in first R modes
    NRJ(R,1) = sum(result_s(1:R))/sum(result_s);
end

R = find(NRJ>=0.999,1);    % Pick # of modes to keep 99.9% of signal energy
disp(R)

% dict must be under the form timeFrame * parameters combinations
[D] = compression_mrf_dictionary(dict0,idx, R);   

% First write all the cfl/hdr files in the correct dimensions for BART
writecfl('csm',single(csm));
writecfl('u',permute(D.u,[3 4 5 6 7 1 2]));
writecfl('traj',permute(0.12 *trajGIRF,[1 2 5 4 6 3]))
writecfl('dcf',permute(sqrt(dcf),[5 1 3 4 6 2]));
writecfl('kdata',permute(U.RawKspaceData,[3 1 5 4 6 2]));

% LR inversion parameters
lambda = 0.05;   
n_iter = 50; 

% Reconstruct singular value images with BART
tic 
cmd = sprintf('pics -d1 -e -l1 -r %f -i %d -p dcf -t traj -B u kdata csm', lambda, n_iter);
recon = bart(cmd);   
writecfl('recon',single(recon));
toc 
svd_images= reshape(readcfl('recon'),N,N,R);
match_images = reshape(svd_images,N^2,R);
c   = zeros([size(match_images,1) 1],'single');
idx2 = zeros([size(match_images,1) 1],'single');

% Dictionary match on singular value images
for n = 1 : N^2
    [c(n,1),idx2(n,1)] = max(match_images(n,:) * conj(D.magnetization), [], 2);
end


Qmaps = reshape(D.lookup_table(idx2,:),[[N N 1], size(D.lookup_table,2)]);
Qmaps = cat(numel(size(Qmaps)),Qmaps,reshape(c ./ D.normalization(idx2).',[N N]));
% PD    = c ./ D.normalization(idx2).';
% PD    = reshape(PD, [N N]);

% Do some background masking
% mask = sum(abs(reshape(svd_images,[[N N] R])),3);
% mask(mask < .4* max(mask(:))) = 0;
% mask (mask ~= 0) = 1;
% Qmaps = bsxfun(@times,Qmaps,mask);

Qmaps = flip (Qmaps);
% figure; imshow3(fliplr(Qmaps(:,:,1,1)), [0 max(max(Qmaps(:,:,1,1)))])
% figure; imshow3(fliplr(Qmaps(:,:,1,2)), [0 max(max(Qmaps(:,:,1,2)))])
% max(max(Qmaps(:,:,1,1)))
% max(max(Qmaps(:,:,1,2)))

figure; imshow3(fliplr(Qmaps(:,:,1,1)), []);
figure; imshow3(fliplr(Qmaps(:,:,1,2)), []); 
% figure; imshow3(fliplr(Qmaps(:,:,1,1)), [0 3000]); colormap hot;
% figure; imshow3(fliplr(Qmaps(:,:,1,2)), [0 1500]); colormap turbo;


recon_folder = 'C:/Users/lucya/MSC_PROJECT/MatlabCode_Phantom/recon_results';
save(fullfile(recon_folder, 'digital_phantom.mat'), 'svd_images', 'D', 'dict0', 'idx', 'R', 'match_images', 'idx2', 'c', 'N', 'Qmaps');

