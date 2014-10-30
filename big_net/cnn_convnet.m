load '../data/input.mat';

addpath('../util/');
addpath('../util/convnet_avg_pool');

train_y = train_y';
test_y = test_y';
train_x = bsxfun(@minus, train_x, mean(train_x,4)) ;
M = csvread('conv1.csv');
Mb = csvread('conv1_biases.csv');

net.param1 = zeros(5,5,3,32);
net.b1 = zeros(32,1);
for j = 1 : 32 %  output map
    net.param1(:,:,1,j) = reshape(M(1:25,j),5,[])';
    net.param1(:,:,2,j) = reshape(M(26:50,j),5,[])';
    net.param1(:,:,3,j) = reshape(M(51:75,j),5,[])';
    net.b1(j) = Mb(j);
end

N = csvread('conv2.csv');
Nb = csvread('conv2_biases.csv');

net.param2 = zeros(5,5,32,32);
net.b2 = zeros(32,1);
for j = 1 : 32 %  output map
    for i = 1 : 32 %  input map
        net.param2(:,:,i,j) = reshape(N(25*(i-1)+1:25*i,j),5,[])';
    end
    net.b2(j) = Nb(j);
end

net.ffW = csvread('fc10.csv')';
net.ffb = csvread('fc10_biases.csv')';

in = train_x(:,:,:,1:10);
out = train_y(:,1:10);
error = 0;

for i = 1:size(in,4)
    x = in(:,:,:,i);
    x = conv_layer(x,net.param1,net.b1,1,0,'sigm');
    x = max_pooler(x,2,2);
    x = conv_layer(x,net.param2,net.b2,1,0,'sigm');
    x = max_pooler(x,2,2);
    x = reshaper_row(x);
    size(net.ffW )
    size(x)
    x = sigm(net.ffW * x + net.ffb);
    result = tiedrank(x);
    A = out(:,i);
    index = find(A==max(A));
    if(result(index) < 10)
        error = error + 1;
    end
    result(index)
end
error

%{
error5 = 0;
error3 = 0;
error = 0;
result = tiedrank(net.o);
for i = 1:size(out,2)
    A = out(:,i);
    index = find(A==max(A));
    B = result(:,i);
    if (B(index) < 6)
        error5 = error5+1;
    end
    if (B(index) < 8)
        error3 = error3+1;
    end
    if (B(index) < 10)
        error = error+1;
    end
end
error5
error3
error
%}