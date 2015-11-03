function [net, info] = cnn_weakly_label(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.expDir = fullfile('/home/zhouhy/Downloads/matconvnet/examples/data','voc2012') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','voc2012') ;
opts.imdbPath = fullfile(opts.expDir, 'train.mat');
opts.useBnorm = false ;
opts.train.batchSize = 16 ;
opts.train.numEpochs = 200 ;
opts.train.continue = true ;
opts.train.gpus = 1 ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

% if exist(opts.imdbPath, 'file')
%   load(opts.imdbPath) ;
% else
%   imdb = getMnistImdb(opts) ;
%   mkdir(opts.expDir) ;
%   save(opts.imdbPath, '-struct', 'imdb') ;
% end

% reset imdb.images.set
% sizeSet=size(imdb.images.set);
% imdb.images.set(1,sizeSet(2)-100:sizeSet(2))=2;

%net = cnn_weakly_label_init('useBnorm', opts.useBnorm) ;
f=1/100;
net=load('hybridCNN.mat');
%reorganise
net.layers{1,1}.pad=[2,1,2,1];
net.layers{1,3}.param=[5,1,1e-4,0.75];
net.layers{1,5}.filters=net.layers{1,5}.weights{1,1};
net.layers{1,7}.param=[5,1,1e-4,0.75];
net.layers{1,20}=struct('type','conv',...
    'weights',{{f*randn(1,1,4096,2048,'single'),zeros(1,2048,'single')}},...
    'stride',1,...
    'pad',0);
net.layers{1,21}=struct('type','relu');
net.layers{end+1}=struct('type','conv',...
    'weights',{{f*randn(1,1,2048,20,'single'),zeros(1,20,'single')}},...
    'stride',1,....
    'pad',0);
net.layers{end+1}=struct('type','globalmaxpool','method','max','stride',1,'pad',0);

%net.layers{end+1}=struct('type','softmax');
net.layers{end+1}=struct('type','softmaxforweaklabel');
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

imdb={};
[net, info] = cnn_weakly_label_train(net,imdb,@getBatch,opts.train) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(Set, batch, mode, scale)
% --------------------------------------------------------------------
if strcmp(mode,'train')
    szBatch=size(batch,2);
    labels=zeros(20,szBatch);
    for k=1:szBatch
        n=char(Set(batch(k)));
        im(:,:,:,k)=imresize(imread(n),scale*[500 500]);
        txtname=strcat('/home/zhouhy/VOCdevkit/VOC2012/labels/',n(43:53),'.txt');
        [classes,x,y,width,height]=textread(txtname,'%d %f %f %f %f');
        for j=1:size(classes,1)
            labels(classes(j)+1,k)=1;
        end
    end
else
    batch=batch-5717; %train number
    szBatch=size(batch,2);
    labels=zeros(20,szBatch);
    for k=1:szBatch
        n=char(Set(batch(k)));
        im(:,:,:,k)=imresize(imread(n),[500 500]);
        txtname=strcat('/home/zhouhy/VOCdevkit/VOC2012/labels/',n(43:53),'.txt');
        [classes,x,y,width,height]=textread(txtname,'%d %f %f %f %f');
        for j=1:size(classes,1)
            labels(classes(j)+1,k)=1;
        end
    end
end
% im = imdb.images.data(:,:,:,batch) ;
% labels = imdb.images.labels(:,batch) ;

% --------------------------------------------------------------------
