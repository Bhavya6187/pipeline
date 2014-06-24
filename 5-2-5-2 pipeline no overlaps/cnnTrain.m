clear;
load '../data/input.mat';

train_y = train_y';
test_y = test_y';
opts.alpha = .001;
opts.batchsize = 50;
opts.numepochs = 100;

net.param1 = .001*(randn(5,5,3,16)-0.5);
net.b1=zeros(16);

net.param2 = .001*(randn(5,5,16,16)-0.5);
net.b2=zeros(16);

fvnum = 25*16;
onum = 10;
net.ffW = (rand(onum, fvnum) - 0.5) * 0.001;
net.ffb = zeros(onum, 1);

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
