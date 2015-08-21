function [result,maxpooling] = vl_nnglobalmaxpool(x,layers,pad,stride,method,cudnn,dzdy)
%VL_NNGLOBALMAXPOOL 此处显示有关此函数的摘要
%   此处显示详细说明
if nargin<=6
    szData=size(x);
    poolH=szData(1);
    poolW=szData(2);
    %layers.pool=[poolH poolW];
    %global maxpooling
    maxpooling=[poolH poolW];
    result=vl_nnpool(x,maxpooling,...
        'pad',pad,'stride',stride,...
        'method',method,...
        cudnn);
else
    result=vl_nnpool(x,layers,single(dzdy),...
        'pad',pad,'stride',stride,...
        'method',method,...
        cudnn);
    maxpooling=layers;
end

end

