function net = cnnff(net,x,y)

%x = padarray(x,[2 2],'both');

for j = 1:32
    %z = zeros(32,32,1,size(x,4));
    z = zeros(28,28,1,size(x,4));
    for i = 1 : 3
        z = z + convn(x(:,:,i,:),net.param1(:,:,i,j),'valid');
    end
    net.layers{1}.a{j} = sigm(z + net.b1(j));
    net.layers{1}.a{j} = squeeze(net.layers{1}.a{j});
end

net.fv = [];

for j = 1 : numel(net.layers{1}.a)
    sa = size(net.layers{1}.a{j});
    net.fv = [net.fv; reshape(net.layers{1}.a{j}, sa(1) * sa(2), sa(3))];
end

net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));

result = double(bsxfun(@eq, net.o, max(net.o, [], 1)));
net.errors = 0;

for i = 1:size(y,2)
    er = ~all(y(:,i)==result(:,i));
    net.errors = net.errors+er;
end

end