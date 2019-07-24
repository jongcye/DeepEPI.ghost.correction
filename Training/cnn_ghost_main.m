clear ;
reset(gpuDevice(1));

% restoredefaultpath();
run('C:\Users\user\Documents\MATLAB\matconvnet-1.0-beta24\matconvnet-1.0-beta24\matlab\vl_setupnn.m');


%%
netDir        	= '.\network\';
expDir          = [netDir 'test'];

network         = 'cnn_ghost_init';  % network structure: cnn_ghost_init.m

networkType     = 'dagnn';

method          = 'image';

% solver_handle   = @solver.sgd;
% solver_handle	= @solver.adam;
solver_handle	= [];

imageRange      = [0, 256];

imageSize       = [64, 128, 16];   % Image size (width,height,depth)
inputSize       = [64, 128, 16];   % Input size

wgt             = 5;  % k-space scaling factor

numEpochs       = 300;  % # of epochs

batchSize       = 1;    % Batch size
subbatchSize    = 1;    % Subbatch size
numSubBatches   = ceil(batchSize/subbatchSize);
batchSample     = 1;

lrnrate         = [-4, -6];     % learning rate
wgtdecay        = 1e-4;         % weight decay

meanNorm        = false;
varNorm         = false;

gpus            = 1;
train           = struct('gpus', gpus);

param.isreal = 0;

%% load data

load('data_list.mat')
imdb.data_list  = data_list; % sample brain data

imdb.train_set  = [1:4];     % training data index
imdb.val_set    = [5:6];       % validataion data index


%% TRAIN
[net_train, info_train] = cnn_ghost( 'param', param,	...
 	'wgt',          wgt,        'meanNorm',     meanNorm,   'varNorm',          varNorm,        ...
    'network',      network,	'networkType',  networkType,'imdb',             imdb,           ...
    'expDir',       expDir,     'method',       method,     'solver',           solver_handle,  ...
    'imageRange',	imageRange,	'imageSize',    imageSize,	'batchSample',      batchSample,    ...
    'inputSize',    inputSize,	...
    'numEpochs',    numEpochs,  'batchSize',    batchSize,	'numSubBatches',    numSubBatches,	...
    'lrnrate',      lrnrate,    'wgtdecay',     wgtdecay,   'train',            train);

return ;
