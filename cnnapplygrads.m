function net = cnnapplygrads(net, opts)

for j = 1 : 32
    for ii = 1 : 3     
        net.change1(:,:,ii,j) = (net.change1(:,:,ii,j)*net.mom)-(opts.L2conv*net.param1(:,:,ii,j)*opts.alpha)+(opts.alpha*net.layers{1}.dk{ii}{j});
        net.param1(:,:,ii,j) = net.param1(:,:,ii,j) + net.change1(:,:,ii,j);
    end
    net.b1(j) = net.b1(j) - 2*opts.alpha * net.layers{1}.db{j};
end

for j = 1 : 32
    for ii = 1 : 32
        net.change2(:,:,ii,j) = (net.change2(:,:,ii,j)*net.mom)-(opts.L2conv*net.param2(:,:,ii,j)*opts.alpha)+(opts.alpha*net.layers{3}.dk{ii}{j});
        net.param2(:,:,ii,j) = net.param2(:,:,ii,j) + net.change2(:,:,ii,j);
    end
    net.b2(j) = net.b2(j) - 2*opts.alpha * net.layers{3}.db{j};
end


for j = 1 : 64
    for ii = 1 : 32
        net.change3(:,:,ii,j) = (net.change3(:,:,ii,j)*net.mom)-(opts.L2conv*net.param3(:,:,ii,j)*opts.alpha)+(opts.alpha*net.layers{5}.dk{ii}{j});
        net.param3(:,:,ii,j) = net.param3(:,:,ii,j) + net.change3(:,:,ii,j);
    end
    net.b3(j) = net.b3(j) - 2*opts.alpha * net.layers{5}.db{j};
end

net.change4 = (net.change4*net.mom)-(opts.L2fc*net.ffW1*opts.alpha)+(opts.alpha*net.dffW1);
net.ffW1 = net.ffW1 + net.change4;
net.ffb1 = net.ffb1 - 2*opts.alpha * net.dffb1;

net.change5 = (net.change5*net.mom)-(opts.L2fc*net.ffW2*opts.alpha)+(opts.alpha*net.dffW2);
net.ffW2 = net.ffW2 + net.change5;
net.ffb2 = net.ffb2 + 2*opts.alpha * net.dffb2;
end
