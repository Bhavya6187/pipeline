function net = cnnff(net,x,y)

%x = padarray(x,[2 2],'both');
batch = 50;
for j = 1:32
    z = zeros(28,28,size(x,4));
    for k = 1:batch
        z(:,:,k) = conv2(x(:,:,1,k),net.param1{1}{j},'valid')+conv2(x(:,:,2,k),net.param1{2}{j},'valid')+conv2(x(:,:,3,k),net.param1{3}{j},'valid');
    end
    
    net.layers{1}.a{j} = sigm(z + net.b1{j});
    net.layers{1}.a{j} = net.layers{1}.a{j};
end

net.fv = [];

for j = 1:32
    z = convn(net.layers{1}.a{j}, ones(2) / (4), 'valid');   %  !! replace with variable
    net.layers{2}.a{j} = z(1 : 2 : end, 1 : 2 : end, :);
end

for j = 1:32
    z = zeros(10,10,size(x,4));
    for i = 1 : 32
        z = z + convn(net.layers{2}.a{i},net.param2{i}{j},'valid');
    end
    net.layers{3}.a{j} = sigm(z + net.b2{j});
end

net.fv = [];

for j = 1:32
    z = convn(net.layers{3}.a{j}, ones(2) / (4), 'valid');   %  !! replace with variable
    net.layers{4}.a{j} = z(1 : 2 : end, 1 : 2 : end, :);
end


for j = 1 : numel(net.layers{4}.a)
    sa = size(net.layers{4}.a{j});
    net.fv = [net.fv; reshape(net.layers{4}.a{j}, sa(1) * sa(2), sa(3))];
end
net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));

%{
result = double(bsxfun(@eq, net.o, max(net.o, [], 1)));
net.errors = 0;

for i = 1:size(y,2)
    er = ~all(y(:,i)==result(:,i));
    net.errors = net.errors+er;
end
net.errors
%}
end