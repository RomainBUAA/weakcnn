#Version 0.1
This is a matlab implementation of weakly supervised ConvNet, which has its origin from
'Is object localization for free? – Weakly-supervised learning with convolutional neural networks' in CVPR 2015.

##How to implement the code
1.I used matconvnet in this project, which means you should have installed Matconvnet in your computer.
Here is the link (http://www.vlfeat.org/matconvnet/) Thanks to A. Vedaldi and K. Lenc for providing such a good tool.
You need to download and install Matconvnet.

2.Put the function file in folders 'example' and 'matlab' to corresponding location.(PS: you should have the same two folders
  in the root path of Matconvnet)

3.run example/cnn_weakly_label.m.
##Some Notes
1.I didn't use GPU here which is easily supported in Matconvnet. So, be free to use it in the code.
2.Global max pooling is implemented in vl_nnglobalmaxpool.m using normal vl_nnpool.m plus some tricks.
3.I added a softmax layer  to the original paper. Please read cnn_weakly_label_init.m carefully.
4.Pay attention to the output error.  mAP(mean average precision) instead of top error is applied to the output.
I didn't modify the output message carefully cause I don't have enough time. This situatio will be improved in the latter version.
5.We only used a small set from PASCAL VOC 2012 which is not enough for complete training process.
Feel free to contact me via zhouhy at lamda.nju.edu.cn.
