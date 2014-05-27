function net = cnnff(net,x,y)

sa = size(x);

net.fv = [];
net.fv = [net.fv; reshape(x, sa(1) * sa(2)*sa(3), sa(4))];

net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));

result = double(bsxfun(@eq, net.o, max(net.o, [], 1)));
net.errors = 0;

for i = 1:size(y,2)
    er = ~all(y(:,i)==result(:,i));
    net.errors = net.errors+er;
end

end