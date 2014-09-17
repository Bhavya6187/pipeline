clear;
load '../data/input.mat';

train_y = train_y';
test_y = test_y';
opts.alpha = .001;

opts.alpha = .001;
opts.batchsize = 50;
opts.numepochs = 50;

for j = 1 : 16 %  output map
    for i = 1 : 3 %  input map
        net.param1{i}{j} = (rand(5) - 0.5) * .001;
    end
    net.b1{j} = 0;
end

for j = 1 : 16 %  output map
    for i = 1 : 16 %  input map
        net.param2{i}{j} = (rand(5) - 0.5) * .001;
    end
    net.b2{j} = 0;
end

fvnum = 25*16;
onum = 10;
net.ffW = (rand(onum, fvnum) - 0.5) * 0.001;
net.ffb = zeros(onum, 1);

m = size(train_x, 4);
numbatches = m / opts.batchsize;
net.fv = [];

for i = 1 : opts.numepochs
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    tic;
    net.rL = [];
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(:, :,:, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
        batch_y = train_y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
        net = cnnff(net, batch_x);
        net = cnnbp(net, batch_y,batch_x);
        net = cnnapplygrads(net, opts);
        if isempty(net.rL)
            net.rL(1) = net.L;
        end
        
        if isempty(net.rL)
            net.rL(1) = net.L;
        end
        %net.rL(end+1) = net.L;
        net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
    end
    disp(net.L)
    toc;
end

%figure; plot(net.rL);
%figure; plot(epoch_error);

net = cnnff(net, test_x);

result = double(bsxfun(@eq, net.o, max(net.o, [], 1)));
errors = 0;
for i = 1:size(test_y,2)
    er = ~all(test_y(:,i)==result(:,i));
    errors = errors+er;
end
disp(errors)
%er = numel(bad) / size(test_y, 2);
