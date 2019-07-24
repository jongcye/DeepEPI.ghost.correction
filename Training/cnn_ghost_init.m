function net = cnn_ghost_init(varargin)

opts.meanNorm = true;
opts.varNorm = true;
opts.networkType = 'dagnn';
opts.method = 'residual';

opts.param              = [];

opts.imageRange = [0, 1];
opts.imageSize = [256, 256, 1];

opts.inputSize = [40, 40, 1];
opts.windowSize = [80, 80, 1];
opts.wrapSize = [0, 0, 1];

opts.wgt = 1;

opts.numEpochs = 1e2; 
opts.batchSize          = 16;
opts.numSubBatches      = 1;
opts.batchSample        = 1;

opts.lrnrate            = [-3, -5];
opts.wgtdecay           = 1e-4;

opts.solver             = [];

[opts, ~]               = vl_argparse(opts, varargin) ;

%% edit filter size and channel size
conv_f_size     = 3;            % convolution filter size
pad_filter_size = [1,1,1,1];    % padding size

% Channel size
cn_input = 32; 
cn_b0   = 64;
cn_b1   = 128;
cn_b2   = 256;

%%
opts.bBias      = true;
opts.bBnorm     = true;
opts.bReLU      = true;

bBias       = opts.bBias;
bBnorm      = opts.bBnorm;
bReLU       = opts.bReLU;

%% Layer name

net             = dagnn.DagNN();
opts.input      = 'input';


%% layer stack
% Block0-1
net.addLayer('b0_conv1', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_input, cn_b0], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'input'}, {'b0_conv1'}, {['b0_c1f'],  ['b0_c1b']})
net.addLayer('b0_bnorm1', dagnn.BatchNorm('numChannels', cn_b0, 'epsilon', 1e-5), {'b0_conv1'}, {'b0_bnorm1'}, {['b0_bn1f'], ['b0_bn1b'], ['b0_bn1m']})
net.addLayer('b0_relu1', dagnn.ReLU(), {'b0_bnorm1'}, {'b0_relu1'})

net.addLayer('b0_conv2', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b0, cn_b0], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b0_relu1'}, {'b0_conv2'}, {['b0_c2f'],  ['b0_c2b']})
net.addLayer('b0_bnorm2', dagnn.BatchNorm('numChannels', cn_b0, 'epsilon', 1e-5), {'b0_conv2'}, {'b0_bnorm2'}, {['b0_bn2f'], ['b0_bn2b'], ['b0_bn2m']})
net.addLayer('b0_relu2', dagnn.ReLU(), {'b0_bnorm2'}, {'b0_relu2'})

net.addLayer('b0_pooling', dagnn.Pooling('method', 'avg', 'poolSize', [2, 2], 'pad', 0, 'stride', [2, 2]),{'b0_relu2'},{'b0_pooling'});


% Block1-1
net.addLayer('b1_conv1', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b0, cn_b1], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b0_pooling'}, {'b1_conv1'}, {['b1_c1f'],  ['b1_c1b']})
net.addLayer('b1_bnorm1', dagnn.BatchNorm('numChannels', cn_b1, 'epsilon', 1e-5), {'b1_conv1'}, {'b1_bnorm1'}, {['b1_bn1f'], ['b1_bn1b'], ['b1_bn1m']})
net.addLayer('b1_relu1', dagnn.ReLU(), {'b1_bnorm1'}, {'b1_relu1'})

net.addLayer('b1_conv2', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b1, cn_b1], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b1_relu1'}, {'b1_conv2'}, {['b1_c2f'],  ['b1_c2b']})
net.addLayer('b1_bnorm2', dagnn.BatchNorm('numChannels', cn_b1, 'epsilon', 1e-5), {'b1_conv2'}, {'b1_bnorm2'}, {['b1_bn2f'], ['b1_bn2b'], ['b1_bn2m']})
net.addLayer('b1_relu2', dagnn.ReLU(), {'b1_bnorm2'}, {'b1_relu2'})

net.addLayer('b1_pooling', dagnn.Pooling('method', 'avg', 'poolSize', [2, 2], 'pad', 0, 'stride', [2, 2]),{'b1_relu2'},{'b1_pooling'});



% Block2-1
% net.addLayer('b2_conv1', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b1, cn_b2], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b1_pooling'}, {'b2_conv1'}, {['b2_c1f'],  ['b2_c1b']})
% net.addLayer('b2_bnorm1', dagnn.BatchNorm('numChannels', cn_b2, 'epsilon', 1e-5), {'b2_conv1'}, {'b2_bnorm1'}, {['b2_bn1f'], ['b2_bn1b'], ['b2_bn1m']})
% net.addLayer('b2_relu1', dagnn.ReLU(), {'b2_bnorm1'}, {'b2_relu1'})
% 
% net.addLayer('b2_conv2', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b2, cn_b2], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b2_relu1'}, {'b2_conv2'}, {['b2_c2f'],  ['b2_c2b']})
% net.addLayer('b2_bnorm2', dagnn.BatchNorm('numChannels', cn_b2, 'epsilon', 1e-5), {'b2_conv2'}, {'b2_bnorm2'}, {['b2_bn2f'], ['b2_bn2b'], ['b2_bn2m']})
% net.addLayer('b2_relu2', dagnn.ReLU(), {'b2_bnorm2'}, {'b2_relu2'})
% 
% net.addLayer('b2_pooling', dagnn.Pooling('method', 'avg', 'poolSize', [2, 2], 'pad', 0, 'stride', [2, 2]),{'b2_relu2'},{'b2_pooling'});



% Block3-1
% net.addLayer('b3_conv1', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b2, cn_b3], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b2_pooling'}, {'b3_conv1'}, {['b3_c1f'],  ['b3_c1b']})
% net.addLayer('b3_bnorm1', dagnn.BatchNorm('numChannels', cn_b3, 'epsilon', 1e-5), {'b3_conv1'}, {'b3_bnorm1'}, {['b3_bn1f'], ['b3_bn1b'], ['b3_bn1m']})
% net.addLayer('b3_relu1', dagnn.ReLU(), {'b3_bnorm1'}, {'b3_relu1'})
% 
% net.addLayer('b3_conv2', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b3, cn_b3], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b3_relu1'}, {'b3_conv2'}, {['b3_c2f'],  ['b3_c2b']})
% net.addLayer('b3_bnorm2', dagnn.BatchNorm('numChannels', cn_b3, 'epsilon', 1e-5), {'b3_conv2'}, {'b3_bnorm2'}, {['b3_bn2f'], ['b3_bn2b'], ['b3_bn2m']})
% net.addLayer('b3_relu2', dagnn.ReLU(), {'b3_bnorm2'}, {'b3_relu2'})
% 
% net.addLayer('b3_pooling', dagnn.Pooling('method', 'avg', 'poolSize', [2, 2], 'pad', 0, 'stride', [2, 2]),{'b3_relu2'},{'b3_pooling'});



% Block4
net.addLayer('b2_conv1', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b1, cn_b2], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b1_pooling'}, {'b2_conv1'}, {['b2_c1f'],  ['b2_c1b']})
net.addLayer('b2_bnorm1', dagnn.BatchNorm('numChannels', cn_b2, 'epsilon', 1e-5), {'b2_conv1'}, {'b2_bnorm1'}, {['b2_bn1f'], ['b2_bn1b'], ['b2_bn1m']})
net.addLayer('b2_relu1', dagnn.ReLU(), {'b2_bnorm1'}, {'b2_relu1'})

net.addLayer('b2_conv2', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b2, cn_b1], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b2_relu1'}, {'b2_conv2'}, {['b2_c2f'],  ['b2_c2b']})
net.addLayer('b2_bnorm2', dagnn.BatchNorm('numChannels', cn_b1, 'epsilon', 1e-5), {'b2_conv2'}, {'b2_bnorm2'}, {['b2_bn2f'], ['b2_bn2b'], ['b2_bn2m']})
net.addLayer('b2_relu2', dagnn.ReLU(), {'b2_bnorm2'}, {'b2_relu2'})

net.addLayer('b1_unpooling', dagnn.UnPooling('method', 'avg', 'poolSize', [2, 2], 'pad', 0, 'stride', [2, 2]),{'b2_relu2'},{'b1_unpooling'});


% Block3-2
% net.addLayer('b3_concat', dagnn.Concat(),{'b3_relu2','b3_unpooling'}, {'b3_concat'});
% 
% net.addLayer('b3_conv3', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b3*2, cn_b3], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b3_concat'}, {'b3_conv3'}, {['b3_c3f'],  ['b3_c3b']})
% net.addLayer('b3_bnorm3', dagnn.BatchNorm('numChannels', cn_b3, 'epsilon', 1e-5), {'b3_conv3'}, {'b3_bnorm3'}, {['b3_bn3f'], ['b3_bn3b'], ['b3_bn3m']})
% net.addLayer('b3_relu3', dagnn.ReLU(), {'b3_bnorm3'}, {'b3_relu3'})
% 
% net.addLayer('b3_conv4', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b3, cn_b2], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b3_relu3'}, {'b3_conv4'}, {['b3_c4f'],  ['b3_c4b']})
% net.addLayer('b3_bnorm4', dagnn.BatchNorm('numChannels', cn_b2, 'epsilon', 1e-5), {'b3_conv4'}, {'b3_bnorm4'}, {['b3_bn4f'], ['b3_bn4b'], ['b3_bn4m']})
% net.addLayer('b3_relu4', dagnn.ReLU(), {'b3_bnorm4'}, {'b3_relu4'})
% 
% net.addLayer('b2_unpooling', dagnn.UnPooling('method', 'avg', 'poolSize', [2, 2], 'pad', 0, 'stride', [2, 2]),{'b3_relu4'},{'b2_unpooling'});


% Block2-2
% net.addLayer('b2_concat', dagnn.Concat(),{'b2_relu2','b2_unpooling'}, {'b2_concat'});
% 
% net.addLayer('b2_conv3', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b2*2, cn_b2], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b2_concat'}, {'b2_conv3'}, {['b2_c3f'],  ['b2_c3b']})
% net.addLayer('b2_bnorm3', dagnn.BatchNorm('numChannels', cn_b2, 'epsilon', 1e-5), {'b2_conv3'}, {'b2_bnorm3'}, {['b2_bn3f'], ['b2_bn3b'], ['b2_bn3m']})
% net.addLayer('b2_relu3', dagnn.ReLU(), {'b2_bnorm3'}, {'b2_relu3'})
% 
% net.addLayer('b2_conv4', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b2, cn_b1], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b2_relu3'}, {'b2_conv4'}, {['b2_c4f'],  ['b2_c4b']})
% net.addLayer('b2_bnorm4', dagnn.BatchNorm('numChannels', cn_b1, 'epsilon', 1e-5), {'b2_conv4'}, {'b2_bnorm4'}, {['b2_bn4f'], ['b2_bn4b'], ['b2_bn4m']})
% net.addLayer('b2_relu4', dagnn.ReLU(), {'b2_bnorm4'}, {'b2_relu4'})
% 
% net.addLayer('b1_unpooling', dagnn.UnPooling('method', 'avg', 'poolSize', [2, 2], 'pad', 0, 'stride', [2, 2]),{'b2_relu4'},{'b1_unpooling'});


% Block1-2
net.addLayer('b1_concat', dagnn.Concat(),{'b1_relu2','b1_unpooling'}, {'b1_concat'});

net.addLayer('b1_conv3', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b1*2, cn_b1], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b1_concat'}, {'b1_conv3'}, {['b1_c3f'],  ['b1_c3b']})
net.addLayer('b1_bnorm3', dagnn.BatchNorm('numChannels', cn_b1, 'epsilon', 1e-5), {'b1_conv3'}, {'b1_bnorm3'}, {['b1_bn3f'], ['b1_bn3b'], ['b1_bn3m']})
net.addLayer('b1_relu3', dagnn.ReLU(), {'b1_bnorm3'}, {'b1_relu3'})

net.addLayer('b1_conv4', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b1, cn_b0], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b1_relu3'}, {'b1_conv4'}, {['b1_c4f'],  ['b1_c4b']})
net.addLayer('b1_bnorm4', dagnn.BatchNorm('numChannels', cn_b0, 'epsilon', 1e-5), {'b1_conv4'}, {'b1_bnorm4'}, {['b1_bn4f'], ['b1_bn4b'], ['b1_bn4m']})
net.addLayer('b1_relu4', dagnn.ReLU(), {'b1_bnorm4'}, {'b1_relu4'})

net.addLayer('b0_unpooling', dagnn.UnPooling('method', 'avg', 'poolSize', [2, 2], 'pad', 0, 'stride', [2, 2]),{'b1_relu4'},{'b0_unpooling'});


% Block0-2
net.addLayer('b0_concat', dagnn.Concat(),{'b0_relu2','b0_unpooling'}, {'b0_concat'});

net.addLayer('b0_conv3', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b0*2, cn_b0], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b0_concat'}, {'b0_conv3'}, {['b0_c3f'],  ['b0_c3b']})
net.addLayer('b0_bnorm3', dagnn.BatchNorm('numChannels', cn_b0, 'epsilon', 1e-5), {'b0_conv3'}, {'b0_bnorm3'}, {['b0_bn3f'], ['b0_bn3b'], ['b0_bn3m']})
net.addLayer('b0_relu3', dagnn.ReLU(), {'b0_bnorm3'}, {'b0_relu3'})

net.addLayer('b0_conv4', dagnn.Conv('size',[conv_f_size, conv_f_size, cn_b0, cn_input], 'pad', pad_filter_size, 'stride', 1, 'hasBias', bBias), {'b0_relu3'}, {'b0_conv4'}, {['b0_c4f'],  ['b0_c4b']})
net.addLayer('b0_bnorm4', dagnn.BatchNorm('numChannels', cn_input, 'epsilon', 1e-5), {'b0_conv4'}, {'b0_bnorm4'}, {['b0_bn4f'], ['b0_bn4b'], ['b0_bn4m']})
net.addLayer('b0_relu4', dagnn.ReLU(), {'b0_bnorm4'}, {'b0_relu4'})


% Fully connected layer
conv_fully          	= dagnn.Conv('size', [1, 1, cn_input, cn_input], 'pad', [0 0 0 0], 'stride', 1, 'hasBias', true);
net.addLayer('conv_fully',  conv_fully, {'b0_relu4'}, {'res_unwgt'}, {'fc_cf', 'fc_cb'});

net.addLayer('l_fft', dagnn.IFFT('param', opts.param), {'res_unwgt'}, {'regr'});


%%
l_loss      = dagnn.EuclideanLoss('p', 2);
l_err2psnr  = dagnn.Error('loss', 'psnr'); 
l_err2mse 	= dagnn.Error('loss', 'mse'); 

net.addLayer('loss',    l_loss,     {'regr', 'label'}, {'objective'});

% if strcmp(opts.method, 'image')
    net.addLayer('psnr',    l_err2psnr,	{'regr', 'label', 'means'}, {'psnr'});
    net.addLayer('mse',     l_err2mse,  {'regr', 'label', 'means'}, {'mse'});
% else
%     net.addLayer('psnr',    l_err2psnr,	{'regr', 'label', 'means', 'input'}, {'psnr'});
%     net.addLayer('mse',     l_err2mse,  {'regr', 'label', 'means', 'input'}, {'mse'});
% end




net.initParams();

%% Meta parameters
net.meta.inputSize                  = opts.inputSize ;

net.meta.trainOpts.method           = opts.method;

if length(opts.lrnrate) == 2
    net.meta.trainOpts.learningRate     = logspace(opts.lrnrate(1), opts.lrnrate(2), opts.numEpochs) ;
else
    net.meta.trainOpts.learningRate     = opts.lrnrate;
end

net.meta.trainOpts.errorFunction	= 'euclidean';

net.meta.trainOpts.numEpochs        = opts.numEpochs ;
net.meta.trainOpts.batchSize        = opts.batchSize ;
net.meta.trainOpts.numSubBatches    = opts.numSubBatches ;
net.meta.trainOpts.batchSample      = opts.batchSample ;

net.meta.trainOpts.weightDecay      = opts.wgtdecay;
net.meta.trainOpts.momentum         = 9e-1;

net.meta.trainOpts.imageRange    	= opts.imageRange;

net.meta.trainOpts.solver           = opts.solver ;
% net.meta.trainOpts.solver           = @solver.adam ;


end