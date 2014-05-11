function net = cnnff(net,x,y)

for j = 1:16
    %z = zeros(size(x(:,:,1,1)) - [4 4 3 0]);
    z = zeros(28,28,1,size(x,4));
    for i = 1 : 3
        z = z + convn(x(:,:,i,:),net.param1(:,:,i,j),'valid');
    end
    net.layers{1}.a{j} = sigm(z + net.b1(j));
    net.layers{1}.a{j} = squeeze(net.layers{1}.a{j});
end

%size(net.layers{1}.a{1}(:,:,1))
for j = 1:16
%    for k = 1:size(net.layers{1}.a{j},3)
%        [net.layers{2}.a{j}(:,:,k),net.unpooler{1}(:,:,k)]  = max_pooler(net.layers{1}.a{j}(:,:,k),2);
%    end
    [net.layers{2}.a{j},net.unpooler{1}]  = max_pooler(net.layers{1}.a{j},2);
end
%size(net.layers{1}.a{j})
for j = 1:16
    z = zeros(size(net.layers{2}.a{1}) - [4 4 0]);   
    for i = 1 : 16
        z = z + convn(net.layers{2}.a{i},net.param2(:,:,i,j),'valid');
    end
    net.layers{3}.a{j} = sigm(z + net.b2(j));
end

for j = 1:16
%    for k = 1:size(net.layers{3}.a{j},3)
%        [net.layers{4}.a{j}(:,:,k),net.unpooler{2}(:,:,k)]  = max_pooler(net.layers{3}.a{j}(:,:,k),2);
%    end
    %[net.layers{4}.a{j},net.unpooler{2}]  = max_pooler(net.layers{3}.a{j},2);
    [net.layers{4}.a{j},net.unpooler{2}]  = max_pooler(net.layers{3}.a{j},2);
end
net.fv = [];

for j = 1 : numel(net.layers{4}.a)
    sa = size(net.layers{4}.a{j});
    net.fv = [net.fv; reshape(net.layers{4}.a{j}, sa(1) * sa(2), sa(3))];
end

net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));

result = double(bsxfun(@eq, net.o, max(net.o, [], 1)));
net.errors = 0;

for i = 1:size(y,2)
    er = ~all(y(:,i)==result(:,i));
    net.errors = net.errors+er;
end

end