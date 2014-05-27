function net = cnnbp(net,y,x)
net.e = net.o - y;

net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);

net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
net.fvd = (net.ffW' * net.od);              %  feature vector delta

sa = size(net.layers{1}.a{1});
fvnum = sa(1) * sa(2);

for j = 1 : numel(net.layers{1}.a)
    net.layers{1}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
end

for j = 1 : numel(net.layers{1}.a)
    for i = 1 : 3
        net.layers{1}.dk{i}{j} = convn(squeeze(flipall(x(:,:,i,:))), net.layers{1}.d{j}, 'valid') / size(net.layers{1}.d{j}, 3);
    end
    net.layers{1}.db{j} = sum(net.layers{1}.d{j}(:)) / size(net.layers{1}.d{j}, 3);
end


net.dffW = net.od * (net.fv)' / size(net.od, 2);
net.dffb = mean(net.od, 2);
    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end