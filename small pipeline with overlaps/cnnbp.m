function net = cnnbp(net,y,x)
net.e = net.o - y;

net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);

net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
net.fvd = (net.ffW' * net.od);              %  feature vector delta

net.dffW = net.od * (net.fv)' / size(net.od, 2);
net.dffb = mean(net.od, 2);
    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end