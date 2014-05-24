clear;
load 'data\input.mat';

%{
train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');
%}
train_y = train_y';
test_y = test_y';
opts.alpha = .01;
opts.batchsize = 50;
opts.numepochs = 10;

net.param1 = .01*(randn(5,5,3,32)-0.5);
net.param2 = .01*(randn(5,5,32,32)-0.5);
net.param3 = .01*(randn(5,5,32,64)-0.5);
net.b1=zeros(32);
net.b2=zeros(32);
net.b3=zeros(64);

net.change1 = .01*(zeros(5,5,3,32)-0.5);
net.change2 = .01*(zeros(5,5,32,32)-0.5);
net.change3 = .01*(zeros(5,5,32,64)-0.5);

net.mom = 0.9;
opts.L2conv = .004;
opts.L2fc = .03;
fvnum = 1024;
onum = 64;
net.ffW1 = (rand(onum, fvnum) - 0.5) * 0.01;
net.change4 = (zeros(onum, fvnum) - 0.5) * 0.01;
net.ffb1 = zeros(onum, 1);

fvnum = 64;
onum = 10;
net.ffW2 = (rand(onum, fvnum) - 0.5) * 0.01;
net.change5 = (zeros(onum, fvnum) - 0.5) * 0.01;
net.ffb2 = zeros(onum, 1);

m = size(train_x, 4);
numbatches = m / opts.batchsize;
net.rL = zeros(1,numbatches );
net.fv = [];

epoch_error= ones(1,opts.numepochs);

for i = 1 : opts.numepochs
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    tic;
    net.rL = zeros(1,numbatches );
    kk = randperm(m);
    for l = 1 : numbatches
        
        batch_x = train_x(:, :,:, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
        batch_y = train_y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
        
        net = cnnff(net, batch_x,batch_y);
        net = cnnbp(net, batch_y,batch_x);
        net = cnnapplygrads(net, opts);
        net.rL(l) = net.errors;
        if(mod(l,100)==0)
            disp(l);
        end
    end
    toc;
    sum(net.rL)
end

%figure; plot(epoch_error);

net = cnnff(net, test_x,test_y);

result = double(bsxfun(@eq, net.o, max(net.o, [], 1)));
errors = 0;
for i = 1:size(test_y,2)
    er = ~all(test_y(:,i)==result(:,i));
    errors = errors+er;
end
disp(errors)
%er = numel(bad) / size(test_y, 2);