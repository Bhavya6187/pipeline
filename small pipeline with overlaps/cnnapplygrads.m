function net = cnnapplygrads(net, opts)

for j = 1 : 32
    for ii = 1 : 3     
        net.change1(:,:,ii,j) = (net.change1(:,:,ii,j)*net.mom)-(opts.L2conv*net.param1(:,:,ii,j)*opts.alpha)+(opts.alpha*net.layers{1}.dk{ii}{j});
        net.param1(:,:,ii,j) = net.param1(:,:,ii,j) + net.change1(:,:,ii,j);
    end
    net.b1(j) = net.b1(j) - 2*opts.alpha * net.layers{1}.db{j};
end

net.change2 = (net.change2*net.mom)-(opts.L2fc*net.ffW*opts.alpha)+(opts.alpha*net.dffW);
net.ffW = net.ffW + net.change2;
net.ffb = net.ffb - 2*opts.alpha * net.dffb;

end