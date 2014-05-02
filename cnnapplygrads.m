function net = cnnapplygrads(net, opts)

for j = 1 : 6
    for ii = 1 : 3     
        net.param1(:,:,ii,j) = net.param1(:,:,ii,j) - opts.alpha * net.layers{1}.dk{ii}{j};
    end
    net.b1(j) = net.b1(j) - opts.alpha * net.layers{1}.db{j};
end

for j = 1 : numel(net.layers{3}.a)
    for ii = 1 : numel(net.layers{2}.a)
        net.param2(:,:,ii,j) = net.param2(:,:,ii,j) - opts.alpha * net.layers{3}.dk{ii}{j};
    end
    net.b2(j) = net.b2(j) - opts.alpha * net.layers{3}.db{j};
end

net.ffW = net.ffW - opts.alpha * net.dffW;
net.ffb = net.ffb - opts.alpha * net.dffb;
end
