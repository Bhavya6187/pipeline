function net = cnnff(net,x,y)

%x = padarray(x,[2 2],'both');

for j = 1:16
    z = zeros(32,32,1,size(x,4));
    for i = 1 : 3
        z = z + convn(x(:,:,i,:),net.param1(:,:,i,j),'valid');
    end
    net.layers{1}.a{j} = sigm(z + net.b1(j));
    net.layers{1}.a{j} = squeeze(net.layers{1}.a{j});
end

net.fv = [];

for j = 1:16
    z = convn(net.layers{1}.a{j}, ones(2) / (4), 'valid');   %  !! replace with variable
    net.layers{2}.a{j} = z(1 : 2 : end, 1 : 2 : end, :);
end


for j = 1 : numel(net.layers{2}.a)
    sa = size(net.layers{2}.a{j});
    net.fv = [net.fv; reshape(net.layers{2}.a{j}, sa(1) * sa(2), sa(3))];
end

net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));

result = double(bsxfun(@eq, net.o, max(net.o, [], 1)));
net.errors = 0;

for i = 1:size(y,2)
    er = ~all(y(:,i)==result(:,i));
    net.errors = net.errors+er;
end

end