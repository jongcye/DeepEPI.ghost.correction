function [net, info] = cnn_ghost(varargin)

% run(fullfile(fileparts(mfilename('fullpath')),...
%     '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.meanNorm           = true ;
opts.varNorm            = true ;
opts.network            = [] ;
opts.networkType        = 'simplenn' ;
opts.method             = 'image';

[opts, varargin]        = vl_argparse(opts, varargin) ;

opts.expDir             = '';
opts.train              = struct() ;
[opts, varargin]        = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

opts.imdb               = [];

opts.imageRange         = [0, 1];
opts.imageSize          = [512, 512, 1];
opts.inputSize          = [40, 40, 1];
opts.wrapSize           = [0, 0, 1];

opts.wgt                = 1;

opts.numEpochs          = 1e2;
opts.batchSize          = 16;
opts.numSubBatches      = 1;
opts.batchSample        = 1;

opts.param              = [];

opts.lrnrate            = [-3, -5];
opts.wgtdecay           = 1e-4;

opts.solver             = [];

[opts, ~]               = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

network = str2func(opts.network);
net    	= network( 'batchSample', opts.batchSample, 'param', opts.param, ...
    'wgt',          opts.wgt,           'meanNorm',     opts.meanNorm,      'varNorm',          opts.varNorm,     	...
    'networkType',  opts.networkType,   'method',       opts.method,        'imageRange',       opts.imageRange,	...
    'imaegSize',    opts.imageSize,     'inputSize',	opts.inputSize, 	'wrapSize',         opts.wrapSize,      ...
    'numEpochs',    opts.numEpochs,   	'batchSize',    opts.batchSize,     'numSubBatches',    opts.numSubBatches, ...
    'lrnrate',      opts.lrnrate,       'wgtdecay',     opts.wgtdecay,      'solver',           opts.solver) ;

imdb    = opts.imdb;

% net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
    case 'simplenn',    trainfn = @cnn_train ;
    case 'dagnn',       trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', imdb.val_set) ;

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
    case 'simplenn'
        fn = @(x,y) getSimpleNNBatch(x,y) ;
    case 'dagnn'
        bopts = struct( 'numGpus', numel(opts.train.gpus), 'method', opts.method, ...
                        'wgt', opts.wgt, 'meanNorm', opts.meanNorm, 'varNorm', opts.varNorm, ...
                        'imageSize', opts.imageSize, 'inputSize', opts.inputSize) ;
        fn = @(x,y,z) getDagNNBatch(bopts,x,y,z) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch, mode)
% --------------------------------------------------------------------
meanNorm    = opts.meanNorm;
varNorm     = opts.varNorm;
nsz         = opts.imageSize;
patch       = opts.inputSize;
method      = opts.method;

% if ~strcmp(mode, 'train')
%     patch = nsz;
% end
max_intensity = 128;

wgt         = opts.wgt;
offset      = 0;

nbatch  = length(batch);

images  = zeros(patch(1), patch(2), patch(3), nbatch, 'single');
labels	= zeros(patch(1), patch(2), nsz(3), nbatch, 'single');

by      = 1:patch(1);
bx      = 1:patch(2);


for ibatch = 1:nbatch
    
    batch_data              = batch(ibatch);
    
    load(imdb.data_list{batch_data})
    
    labels(:,:,:,ibatch)	= label_;
    images(:,:,:,ibatch) 	= ifftshift(ifft2(fftshift(ghost_,2)),2);


end

if meanNorm
    means   = mean(mean(mean(images, 1), 2), 3);
else
    means   = 0;
end

if varNorm
    vars   = max(max(max(abs(images), [], 1), [], 2), [], 3);
else
    vars    = 1;
end

if strcmp(method, 'residual')
    tmp     = ifftshift(ifftshift(ifft2(fftshift(fftshift(images,1),2)),1),2);
    labels  = tmp - labels;
end

images_cat  = cat(3, real(images), imag(images));
labels_cat  = cat(3, real(labels), imag(labels));

images_cat  = wgt.*images_cat + offset;
labels_cat  = wgt.*labels_cat + offset;

if opts.numGpus > 0
    images = gpuArray(single(images_cat)) ;
    labels = gpuArray(single(labels_cat));
end

inputs	= {'input', images, 'label', labels, 'means', means} ;


