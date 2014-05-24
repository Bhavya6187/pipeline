

net.imageDim = 32;
net.numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
net.filterDim = 5;    % Filter size for conv layer
net.padding = 2;

%Initialise paramenters%
net.param(1) = .0001*randn(5,5,64,3);
net.param(2) = .0001*randn(5,5,64,64);
net.b(1) = zeros(64, 3);
net.b(1) = zeros(64, 64);

a = load('data/data_batch_1.mat');
b = load('data/data_batch_2.mat');
c = load('data/data_batch_3.mat');
d = load('data/data_batch_4.mat');
e = load('data/data_batch_5.mat');
images = [a.data(); b.data(); c.data(); d.data()];
labels = [a.labels(); b.labels(); c.labels(); d.labels()];

net.rL = [];

opts.numImages = size(images,1);
opts.batchsize = 64;
opts.numbatches = numImages/batchsize;
opts.numepochs = 100;

for i = 1:opts.numepochs
    k = randperm(opts.numImages);
    for j=1:opts.numbatches
        batch_x = images(k((j-1)*opts.batchsize+1:j*opts.batchsize),:);
        batch_y = labels(k((j-1)*opts.batchsize+1:j*opts.batchsize));
        net = cnnff(net, batch_x);
        net = cnnbp(net, batch_y);
        net = cnnapplygrads(net, opts);
    end
end
