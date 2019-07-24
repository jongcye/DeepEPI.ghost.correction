%%%%% inference

clear; close all;
reset(gpuDevice(1));
gpus        = 1;

run('C:\Users\user\Documents\MATLAB\matconvnet-1.0-beta24\matconvnet-1.0-beta24\matlab\vl_setupnn.m');

%%

dataDir         = './db/';
netDir        	= './network/';

load([dataDir 'imdb_raw_test1.mat']); % Test image set: imdb_raw_test1/test2

%% Param

imageRange      = [0, 256];
imageSize       = [64 128 16];
inputSize       = [64 128 16];

wgt             = 5;

meanNorm        = false;
varNorm         = false;

gpus            = 1;
train           = struct('gpus', gpus);

param.isreal = 0;

%%
% NETWORK SETTING
modelPath	= @(ep) fullfile(netDir, sprintf('net-epoch-%d.mat', ep));

epoch = 100;

net         = loadState(modelPath(epoch)) ;

net = dagnn.DagNN.loadobj(net) ;

rmlyr_set = {'loss', 'psnr', 'mse'};

for ilyr = 1:length(rmlyr_set)
    lyr = rmlyr_set{ilyr};
    net.removeLayer(lyr);
end

net.mode                = 'test';

v                    	= net.getVarIndex('regr') ;
net.vars(v).precious	= true ;

% GPU
if ~isempty(gpus)
    net.move('gpu');
end

%%
% DATA LOADING

for ival	= 1

    data_mc        = ghost_;
    labels_mc      = ssos(label_,3);

    if ~isempty(gpus)
        data_mc     = gpuArray(data_mc);
    end

    
    data = squeeze(data_mc);
    
    images_cat  = gpuArray(zeros(inputSize(1), inputSize(2), inputSize(3)*2));
    images_cat  = cat(3, real(data), imag(data));
     
    images_cat    = wgt.*images_cat ;
    
    net.eval({'input',single(images_cat)}) ;
    
    rec_batch  = gather(net.vars(v).value);
    
    res_tmp = rec_batch;
    res_tmp_unw = complex(res_tmp(:,:,1:end/2,:), res_tmp(:,:,end/2+1:end,:));
    res_plot = res_tmp_unw;
    rec_plot = ssos(res_plot,3)./wgt;
    
    figure;
    subplot(1,2,1), imagesc(abs(rec_plot)), title('result image'), colormap gray, axis equal tight off
    subplot(1,2,2), imagesc(abs(labels_mc)), title('ALOHA image'), colormap gray, axis equal tight off
end
