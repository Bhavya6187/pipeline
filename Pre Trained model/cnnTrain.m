clear;

load '../data/input.mat';

train_y = train_y';
test_y = test_y';

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 2;

M = csvread('conv1.csv');
Mb = csvread('conv1_biases.csv');

for j = 1 : 32 %  output map
    net.param1{1}{j} = reshape(M(1:25,j),5,[])';
    net.param1{2}{j} = reshape(M(26:50,j),5,[])';
    net.param1{3}{j} = reshape(M(51:75,j),5,[])';
    net.b1{j} = Mb(j);
end

N = csvread('conv2.csv');
Nb = csvread('conv2_biases.csv');

for j = 1 : 32 %  output map
    for i = 1 : 32 %  input map
        net.param2{i}{j} = reshape(N(25*(i-1)+1:25*i,j),5,[])';
    end
    net.b2{j} = Nb(j);
end

net.ffW = csvread('fc10.csv')';
net.ffb = csvread('fc10_biases.csv')';

net = cnnff(net, test_x);

result = double(bsxfun(@eq, net.o, max(net.o, [], 1)));
errors = 0;
for i = 1:size(test_y,2)
    er = ~all(test_y(:,i)==result(:,i));
    errors = errors+er;
end
disp(errors)
%er = numel(bad) / size(test_y, 2);
