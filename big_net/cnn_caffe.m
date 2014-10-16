load('imagenet-caffe-ref');

im = imread('cat.jpg');
im = imresize(im,[227,227]);

x = double(im);

x = x - normalization.averageImage;

x = conv_layer(x,layers{1}.filters,layers{1}.biases,layers{1}.stride(1),layers{1}.pad(1),layers{2}.type);

x = max_pooler(x,layers{3}.stride(1),layers{3}.pool(1));
x3_new = x;
%error()
%size(x)
%x(1:10,1:10,1)

x = norm(x,layers{4}.param);

x = conv_layer(x,layers{5}.filters,layers{5}.biases,layers{5}.stride(1),layers{5}.pad(1),layers{6}.type);

x = max_pooler(x,layers{7}.stride(1),layers{7}.pool(1));

x = norm(x,layers{8}.param);

x = conv_layer(x,layers{9}.filters,layers{9}.biases,layers{9}.stride(1),layers{9}.pad(1),layers{10}.type);

x = conv_layer(x,layers{11}.filters,layers{11}.biases,layers{11}.stride(1),layers{11}.pad(1),layers{12}.type);

x = conv_layer(x,layers{13}.filters,layers{13}.biases,layers{13}.stride(1),layers{13}.pad(1),layers{14}.type);

x = max_pooler(x,layers{15}.stride(1),layers{15}.pool(1));

x = conv_layer(x,layers{16}.filters,layers{16}.biases,layers{16}.stride(1),layers{16}.pad(1),layers{17}.type);

x = conv_layer(x,layers{18}.filters,layers{18}.biases,layers{18}.stride(1),layers{18}.pad(1),layers{19}.type);

x = conv_layer(x,layers{20}.filters,layers{20}.biases,layers{20}.stride(1),layers{20}.pad(1),layers{21}.type);

x = softmax(x);

[val,ind] = max(x)
classes.description{ind}
