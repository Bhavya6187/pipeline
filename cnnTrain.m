load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 10;


net.param1 = .0001*randn(5,5,1,6);
net.param2 = .0001*randn(5,5,6,12);
net.b1=zeros(6);
net.b2=zeros(12);

fvnum = 192;
onum = 10;
net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));

net.ffb = zeros(onum, 1);

m = size(train_x, 3);
numbatches = m / opts.batchsize;
net.rL = [];
net.fv = [];

for i = 1 : opts.numepochs
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    tic;
    kk = randperm(m);
    for l = 1 : numbatches
        
        batch_x = train_x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
        batch_y = train_y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
        
        net = cnnff(net, batch_x);
        net = cnnbp(net, batch_y,batch_x);
        net = cnnapplygrads(net, opts);
        if isempty(net.rL)
            net.rL(1) = net.L;
        end
        net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
    end
    toc;
end

net = cnnff(net, test_x);
[~, h] = max(net.o);
[~, a] = max(test_y);
bad = find(h ~= a);

er = numel(bad) / size(test_y, 2);

figure; plot(net.rL);
