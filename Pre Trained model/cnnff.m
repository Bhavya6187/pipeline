function net = cnnff(net,x)

for j = 1:32
    z = zeros(28,28,size(x,4));
    for i = 1 : 3
        temp = rot90(net.param1{i}{j},2);
        channel = squeeze(x(:,:,i,:));
        z = z + convn(channel,temp,'valid');
    end
    net.layers{1}.a{j} = sigm(z + net.b1{j});
    net.layers{1}.a{j} = net.layers{1}.a{j};
end

net.fv = [];
size(net.layers{1}.a{j})
for j = 1:32
    z = convn(net.layers{1}.a{j}, ones(2) / (4), 'valid');   %  !! replace with variable
    net.layers{2}.a{j} = z(1 : 2 : end, 1 : 2 : end, :);
end

size(net.layers{2}.a{j})
for j = 1:32
    z = zeros(10,10,size(x,4));
    for i = 1 : 32
        temp = rot90(net.param2{i}{j},2);
        z = z + convn(net.layers{2}.a{i},temp,'valid');
    end
    net.layers{3}.a{j} = sigm(z + net.b2{j});
end

net.fv = [];

for j = 1:32
    z = convn(net.layers{3}.a{j}, ones(2) / (4), 'valid');   %  !! replace with variable
    net.layers{4}.a{j} = z(1 : 2 : end, 1 : 2 : end, :);
end

size(net.layers{4}.a{1})
for j = 1 : numel(net.layers{4}.a)
    sa = size(net.layers{4}.a{j});
    %net.fv = [net.fv; reshape(net.layers{4}.a{j}, sa(1) * sa(2), sa(3))];
    net.fv = [net.fv; reshaper_row(net.layers{4}.a{j})];
end
size(net.fv)
net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));

end