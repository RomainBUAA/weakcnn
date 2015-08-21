function [net, info] = cnn_weakly_label(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.expDir = fullfile('data','voc2012') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','voc2012') ;
opts.imdbPath = fullfile(opts.expDir, 'train.mat');
opts.useBnorm = false ;
opts.train.batchSize = 10 ;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
%opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  load(opts.imdbPath) ;
else
  %imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% reset imdb.images.set
sizeSet=size(imdb.images.set);
imdb.images.set(1,sizeSet(2)-10:sizeSet(2))=2;

net = cnn_weakly_label_init('useBnorm', opts.useBnorm) ;
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_weakly_label_train(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 2)) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(:,batch) ;

% --------------------------------------------------------------------
