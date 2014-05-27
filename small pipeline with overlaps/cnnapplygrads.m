function net = cnnapplygrads(net, opts)

net.change = (net.change*net.mom)-(opts.L2fc*net.ffW*opts.alpha)+(opts.alpha*net.dffW);
net.ffW = net.ffW + net.change;
net.ffb = net.ffb - 2*opts.alpha * net.dffb;

end
