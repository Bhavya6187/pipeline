clear;

load '../data/input.mat';

train_y = train_y';
test_y = test_y';
train_x = train_x;
test_x = test_x;

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

net = cnnff(net, train_x,train_y);

%result = double(bsxfun(@eq, net.o, max(net.o, [], 1)));
net.errors = 0;
result = tiedrank(net.o);
for i = 1:size(y,2)
    A = train_y(:,i);
    index = find(A==max(A));
    B = result(:,i);
    if (B(index) < 6)
        net.errors = net.errors+1;
    end
end
net.errors
