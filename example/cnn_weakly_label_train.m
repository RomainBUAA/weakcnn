function [net, info] = cnn_weakly_label_train(net, imdb ,getBatch, opts, varargin)
% CNN_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic
%    gradient descent with momentum to train a CNN. It can be used
%    with different datasets and tasks by providing a suitable
%    getBatch function.
%
%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option). Multi-GPU
%    support is relatively primitive but sufficient to obtain a
%    noticable speedup.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.batchSize = 16 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 200 ;
opts.gpus = 1 ; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
opts.continue = true ;
opts.expDir = fullfile('data','exp_101') ;
opts.conserveMemory = false ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false ;
opts.cudnn = true ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
% opts = vl_argparse(opts, varargin) ;

%----------------------------------------------------------------------------------
% read data in limited space
di=dir('/home/zhouhy/VOCdevkit/VOC2012/JPEGImages/*.jpg');
train_set=textread('/home/zhouhy/VOCdevkit/VOC2012/2012_train.txt','%s');
val_set=textread('/home/zhouhy/VOCdevkit/VOC2012/2012_val.txt','%s');
bS=opts.batchSize;
trainSize=size(train_set);
valSize=size(val_set);

dbSize=trainSize(1)+valSize(1);
%imdb.images.data=zeros(500,500,3,dbSize);
tx=dir('/home/zhouhy/VOCdevkit/VOC2012/labels/*.txt');
txSize=size(tx);
% imdb.images.labels=zeros(20,dbSize);
% imdb.images.set=zeros(1,dbSize);
imdb.images.data_mean={};

for k=1:txSize
    if k<=trainSize(1)
%         n=char(train_set(k));
%         imdb.images.data(:,:,:,k)=imresize(imread(n),[500 500]);
%         txtname=strcat('/home/zhouhy/VOCdevkit/VOC2012/labels/',n(43:53),'.txt');
%         [classes,x,y,width,height]=textread(txtname,'%d %f %f %f %f');
%         numClasses=size(classes);
%         for j=1:numClasses(1)
%             if classes(j)==20
%                 printf('cao');
%             end
%             imdb.images.labels(classes(j)+1,k)=1;
%         end
        imdb.images.set(k)=1;
    else
%         n=char(val_set(k-5717));
%         imdb.images.data(:,:,:,k)=imresize(imread(n),[500 500]);
%         txtname=strcat('/home/zhouhy/VOCdevkit/VOC2012/labels/',n(43:53),'.txt');
%         [classes,x,y,width,height]=textread(txtname,'%d %f %f %f %f');
%         numClasses=size(classes);
%         for j=1:numClasses(1)
%             if classes(j)==20
%                 printf('cao');
%             end
%             imdb.images.labels(classes(j)+1,k)=1;
%         end
        imdb.images.set(k)=3;
    end
    
    
end

% imdb.images.data_mean=mean(imdb.images.data,4);
%----------------------------------------------------------------------------------------

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==3) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;

if ~evaluateMode
  for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
      J = numel(net.layers{i}.weights) ;
      for j=1:J
        net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
      end
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, J, 'single') ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = ones(1, J, 'single') ;
      end
    end
    % Legacy code: will be removed
%     if isfield(net.layers{i}, 'filters')
%         net.layers{i}.momentum{1} = zeros(size(net.layers{i}.filters), 'single') ;
%         net.layers{i}.momentum{2} = zeros(size(net.layers{i}.biases), 'single') ;
%         if ~isfield(net.layers{i}, 'learningRate')
%             net.layers{i}.learningRate = ones(1, 2, 'single') ;
%         end
%         if ~isfield(net.layers{i}, 'weightDecay')
%             net.layers{i}.weightDecay = single([1 0]) ;
%         end
%     end
  end
end

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

% setup error calculation function
if isstr(opts.errorFunction)
  switch opts.errorFunction
    case 'none'
      opts.errorFunction = @error_none ;
    case 'multiclass'
      opts.errorFunction = @error_multiclass ;
      if isempty(opts.errorLabels), opts.errorLabels = {'top1e', 'top5e'} ; end
    case 'binary'
      opts.errorFunction = @error_binary ;
      if isempty(opts.errorLabels), opts.errorLabels = {'bine'} ; end
    otherwise
      error('Uknown error function ''%s''', opts.errorFunction) ;
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

for epoch=1:opts.numEpochs
  learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;

  % fast-forward to last checkpoint
  modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
  if opts.continue
    if exist(modelPath(epoch),'file')
      if epoch == opts.numEpochs
        load(modelPath(epoch), 'net', 'info') ;
      end
      continue ;
    end
    if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1) ;
      load(modelPath(epoch-1), 'net', 'info') ;
    end
  end

  % train one epoch and validate
  train = opts.train(randperm(numel(opts.train))) ; % shuffle
  val = opts.val;
  if numGpus <= 1
      status='train';
      if mod(epoch,10)==0
          learningRate=learningRate*0.8;
      end
      [net,stats.train] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net, train_set, status) ;
      status='validation';
      [~,stats.val] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net, val_set, status) ;
  else
    spmd(numGpus)
      [net_, stats_train_] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net,train_set) ;
      [~, stats_val_] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net_, val_set) ;
    end
    net = net_{1} ;
    stats.train = sum([stats_train_{:}],2) ;
    stats.val = sum([stats_val_{:}],2) ;
  end

  % save
  %if evaluateMode, sets = {'val'} ; else sets = {'train', 'val'} ; end
  sets={'train'};
  for f = sets
    f = char(f) ;
    n = numel(eval(f)) ;
    info.(f).speed(epoch) = n / stats.(f)(1) * max(1, numGpus) ;
    info.(f).objective(epoch) = stats.(f)(2) / n ;
    %info.(f).error(:,epoch) = stats.(f)(3:end) / n ;
  end
  if ~evaluateMode, save(modelPath(epoch), 'net', 'info') ; end

%   figure(1) ; clf ;
%   hasError = isa(opts.errorFunction, 'function_handle') ;
%   subplot(1,1+hasError,1) ;
%   if ~evaluateMode
%     semilogy(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
%     hold on ;
%   end
%   semilogy(1:epoch, info.val.objective, '.--') ;
%   xlabel('training epoch') ; ylabel('energy') ;
%   grid on ;
%   h=legend(sets) ;
%   set(h,'color','none');
%   title('objective') ;
%   if hasError
%     subplot(1,2,2) ; leg = {} ;
%     if ~evaluateMode
%       plot(1:epoch, info.train.error', '.-', 'linewidth', 2) ;
%       hold on ;
%       leg = horzcat(leg, strcat('train ', opts.errorLabels)) ;
%     end
%     plot(1:epoch, info.val.error', '.--') ;
%     leg = horzcat(leg, strcat('val ', opts.errorLabels)) ;
%     set(legend(leg{:}),'color','none') ;
%     grid on ;
%     xlabel('training epoch') ; ylabel('error') ;
%     title('error') ;
%   end
%   drawnow ;
%   print(1, modelFigPath, '-dpdf') ;
end

% -------------------------------------------------------------------------
function mAP = error_multiclass(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;
if (size(predictions,1)*size(predictions,2)*size(predictions,3)*size(predictions,4))~=(size(labels,1)*size(labels,2))
    disp 'cao';
end
predictions=reshape(predictions,size(labels));
numRank=sum(labels,1);
sizeLabel=size(labels);
numImgs=sizeLabel(2);
[~,groundtruth]=sort(labels,1,'descend');
%error = ~bsxfun(@eq, predictions, groundtruth) ;
mAP=0;
for i=1:numImgs
    for j=1:numRank(i)
        for k=1:numRank(i)
            if predictions(j,i)==groundtruth(k,i)
                mAP=mAP+1;
                groundtruth(k,i)=0;
            end
        end
    end
end
mAP=mAP/sum(numRank);
%err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
%err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:5,:),[],3)))) ;

% -------------------------------------------------------------------------
function err = error_binaryclass(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
error = bsxfun(@times, predictions, labels) < 0 ;
err = sum(error(:)) ;

% -------------------------------------------------------------------------
function err = error_none(opts, labels, res)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function  [net_cpu,stats,prof] = process_epoch(opts, getBatch, epoch, subset, learningRate, imdb, net_cpu, Set, status)
% -------------------------------------------------------------------------

% move CNN to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net = vl_simplenn_move(net_cpu, 'gpu') ;
else
  net = net_cpu ;
  net_cpu = [] ;
end

% validation mode if learning rate is zero
training = learningRate > 0 ;
if training, mode = 'training' ; else mode = 'validation' ; end
if nargout > 2, mpiprofile on ; end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end
res = [] ;
mmap = [] ;
stats = [] ;
totalBatch=0;
sz=size(Set,1);
top5=0;
for t=1:opts.batchSize:numel(subset)
  fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
          fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
      if strcmp(mode,'validation')
          disp 'cao';
      end
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
  batchTime = tic ;
  numDone = 0 ;
  error = [] ;
  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    totalBatch=totalBatch+size(batch,2);
    %randomized rescale
    scale=rand()*0.7+0.7;
    [im, labels]=getBatch(Set, batch, status, scale) ;

    if opts.prefetch
      if s==opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      getBatch(Set, nextBatch, status) ;
    end

    if numGpus >= 1
      im = gpuArray(im) ;
    end

    % evaluate CNN
    net.layers{end}.class = labels ;
    if training, dzdy = one; else dzdy = [] ; end
    res = vl_simplenn(net, im, dzdy, res, ...
                      'accumulate', s ~= 1, ...
                      'disableDropout', ~training, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
                      'sync', opts.sync, ...
                      'cudnn', opts.cudnn) ;

    % accumulate training errors
%     error = sum([error, [...
%       sum(double(gather(res(end).x))) ;
%       reshape(opts.errorFunction(opts, labels, res),[],1) ; ]],2) ;
    error=opts.errorFunction(opts, labels, res); %output error information
    numDone = numDone + numel(batch) ;
  end

  % gather and accumulate gradients across labs
  if training
    if numGpus <= 1
      [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res) ;
    else
      if isempty(mmap)
        mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_gradients(mmap, net, res) ;
      labBarrier() ;
      [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res, mmap) ;
    end
  end

  % print learning statistics
  batchTime = toc(batchTime) ;
  stats = sum([stats,[batchTime ; error]],2); % works even when stats=[]
  speed = batchSize/batchTime ;

  fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
  n = (t + batchSize - 1) / max(1,numlabs) ;
  fprintf(' obj:%.3g', stats(2)/n) ;
%   for i=1:numel(opts.errorLabels)
%     fprintf(' %s:%.3g', opts.errorLabels{i}, stats(i+2)/n) ;
%   end
  %mAP
  error=(error*size(batch,2)+top5)/totalBatch;
  top5=top5+error*size(batch,2);
  fprintf(' mAP:%.5f',error);
  fprintf(' [%d/%d]', numDone, batchSize);
  fprintf('\n') ;

  % debug info
  if opts.plotDiagnostics && numGpus <= 1
    figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
  end
end

if nargout > 2
  prof = mpiprofile('info');
  mpiprofile off ;
end

if numGpus >= 1
  net_cpu = vl_simplenn_move(net, 'cpu') ;
else
  net_cpu = net ;
end

% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
for l=numel(net.layers):-1:1
  for j=1:numel(res(l).dzdw)
    thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
    thisLR = lr * net.layers{l}.learningRate(j) ;

    % accumualte from multiple labs (GPUs) if needed
    if nargin >= 6
      tag = sprintf('l%d_%d',l,j) ;
      tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
      for g = setdiff(1:numel(mmap.Data), labindex)
        tmp = tmp + mmap.Data(g).(tag) ;
      end
      res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
    end

    if isfield(net.layers{l}, 'weights') && l>=18
      net.layers{l}.momentum{j} = ...
        opts.momentum * net.layers{l}.momentum{j} ...
        - thisDecay * net.layers{l}.weights{j} ...
        - (1 / batchSize) * res(l).dzdw{j} ;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j} ;
%     else
%       % Legacy code: to be removed
%       if j == 1
%         net.layers{l}.momentum{j} = ...
%           opts.momentum * net.layers{l}.momentum{j} ...
%           - thisDecay * net.layers{l}.filters ...
%           - (1 / batchSize) * res(l).dzdw{j} ;
%         net.layers{l}.filters = net.layers{l}.filters + thisLR * net.layers{l}.momentum{j} ;
%       else
%         net.layers{l}.momentum{j} = ...
%           opts.momentum * net.layers{l}.momentum{j} ...
%           - thisDecay * net.layers{l}.biases ...
%           - (1 / batchSize) * res(l).dzdw{j} ;
%         net.layers{l}.biases = net.layers{l}.biases + thisLR * net.layers{l}.momentum{j} ;
%       end
    end
  end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)} ;
  end
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net, res)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j}) ;
  end
end
