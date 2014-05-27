function net = cnnff(net,x,y)

x = padarray(x,[2 2],'both');

%% check for neuron types

for j = 1:32
    z = zeros(32,32,1,size(x,4));
    for i = 1 : 3
        z = z + convn(x(:,:,i,:),net.param1(:,:,i,j),'valid');
    end
    net.layers{1}.a{j} = sigm(z + net.b1(j));
    net.layers{1}.a{j} = squeeze(net.layers{1}.a{j});
end

%size(net.layers{1}.a{j})

for j = 1:32
    [net.layers{2}.a{j},net.unpooler{1}]  = max_3d_pooler(net.layers{1}.a{j},2,3);
    temp{j} = padarray(net.layers{2}.a{j},[2 2],'both');
end

for j = 1:32
    z = zeros(size(temp{1}) - [4 4 0]);   
    for i = 1 : 32
        z = z + convn(temp{i},net.param2(:,:,i,j),'valid');
    end
    net.layers{3}.a{j} = sigm(z + net.b2(j));
end

for j = 1:32
    [net.layers{4}.a{j}]  = avg_3d_pooler(net.layers{3}.a{j},2,3);
    temp{j} = padarray(net.layers{4}.a{j},[2 2],'both');
end

for j = 1:64
    z = zeros(size(temp{1}) - [4 4 0]);
    for i = 1 : 32
        z = z + convn(temp{i},net.param3(:,:,i,j),'valid');
    end
    net.layers{5}.a{j} = sigm(z + net.b3(j));
end

for j = 1:64
    [net.layers{6}.a{j}]  = avg_3d_pooler(net.layers{5}.a{j},2,3);
end


net.fv = [];

for j = 1 : numel(net.layers{6}.a)
    sa = size(net.layers{6}.a{j});
    net.fv = [net.fv; reshape(net.layers{6}.a{j}, sa(1) * sa(2), sa(3))];
end

%size(net.fv)

net.fc1 = sigm(net.ffW1 * net.fv + repmat(net.ffb1, 1, size(net.fv, 2)));

net.o = sigm(net.ffW2 * net.fc1 + repmat(net.ffb2, 1, size(net.fc1, 2)));
%size(net.o)

result = double(bsxfun(@eq, net.o, max(net.o, [], 1)));
net.errors = 0;

for i = 1:size(y,2)
    er = ~all(y(:,i)==result(:,i));
    net.errors = net.errors+er;
end

end