function net = cnnapplygrads(net, opts)

for j = 1 : 16
    for i = 1 : 3   
        net.param1{i}{j} = net.param1{i}{j} - opts.alpha*net.layers{1}.dk{i}{j};
    end
    net.b1{j} = net.b1{j} - opts.alpha * net.layers{1}.db{j};
end

for j = 1 : 16
    for i = 1 : 16
        net.param2{i}{j} = net.param2{i}{j} - opts.alpha*net.layers{3}.dk{i}{j};
    end
    net.b2{j} = net.b2{j} - opts.alpha * net.layers{3}.db{j};
end

net.ffW = net.ffW - opts.alpha * net.dffW;
net.ffb = net.ffb - opts.alpha * net.dffb;

end