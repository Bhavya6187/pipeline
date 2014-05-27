function net = cnnbp(net,y,x)
net.e = net.o - y;

net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);

net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
net.fvd2 = (net.ffW2' * net.od);              %  feature vector delta
net.fvd = (net.ffW1' * net.fvd2);              %  feature vector delta

sa = size(net.layers{6}.a{1});
fvnum = sa(1) * sa(2);

for j = 1 : numel(net.layers{6}.a)
    net.layers{6}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
end

for j = 1 : numel(net.layers{5}.a)
    [net.layers{5}.d{j}]  = net.layers{5}.a{j} .* (1 - net.layers{5}.a{j}) .* avg_3d_unpooler(net.layers{6}.d{j},2,3);
end

for i = 1 : numel(net.layers{4}.a)
    %    z = zeros(size(net.layers{4}.a{1}));
    z = zeros(12,12,50);
    for j = 1 : numel(net.layers{5}.a)
        z = z + convn(net.layers{5}.d{j}, rot180(net.param3(:,:,i,j)), 'full');
    end
    net.layers{4}.d{i} = z(3:10,3:10,:);
end

for j = 1 : numel(net.layers{3}.a)
    [net.layers{3}.d{j}]  = net.layers{3}.a{j} .* (1 - net.layers{3}.a{j}) .* avg_3d_unpooler(net.layers{4}.d{j},2,3);
end

for i = 1 : numel(net.layers{2}.a)
    %z = zeros(size(net.layers{2}.a{1}));
    z = zeros(20,20,50);
    for j = 1 : numel(net.layers{3}.a)
        z = z + convn(net.layers{3}.d{j}, rot180(net.param2(:,:,i,j)), 'full');
    end
    net.layers{2}.d{i} = z(3:18,3:18,:);
end

for j = 1 : numel(net.layers{1}.a)
    [net.layers{1}.d{j}]  = net.layers{1}.a{j} .* (1 - net.layers{1}.a{j}) .* max_3d_unpooler(net.layers{2}.d{j},net.unpooler{1},2,3);
end


for j = 1 : numel(net.layers{1}.a)
    for i = 1 : 3
        net.layers{1}.dk{i}{j} = convn(squeeze(flipall(x(:,:,i,:))), net.layers{1}.d{j}, 'valid') / size(net.layers{1}.d{j}, 3);
    end
    net.layers{1}.db{j} = sum(net.layers{1}.d{j}(:)) / size(net.layers{1}.d{j}, 3);
end

for j = 1 : numel(net.layers{3}.a)
    for i = 1 : numel(net.layers{2}.a)
        net.layers{3}.dk{i}{j} = convn(flipall(net.layers{2}.a{i}), net.layers{3}.d{j}, 'valid') / size(net.layers{3}.d{j}, 3);
    end
    net.layers{3}.db{j} = sum(net.layers{3}.d{j}(:)) / size(net.layers{3}.d{j}, 3);
end

for j = 1 : numel(net.layers{5}.a)
    for i = 1 : numel(net.layers{4}.a)
        net.layers{5}.dk{i}{j} = convn(flipall(net.layers{4}.a{i}), net.layers{5}.d{j}, 'valid') / size(net.layers{5}.d{j}, 3);
    end
    net.layers{5}.db{j} = sum(net.layers{5}.d{j}(:)) / size(net.layers{5}.d{j}, 3);
end

net.dffW1 = net.fvd * (net.fv)' / size(net.fv, 2);
net.dffb1 = mean(net.fvd2, 2);

net.dffW2 = net.od * (net.fc1)' / size(net.od, 2);
net.dffb2 = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end