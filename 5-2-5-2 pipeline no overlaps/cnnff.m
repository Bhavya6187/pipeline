function net = cnnff(net,x,y)

%x = padarray(x,[2 2],'both');

for j = 1:32
    z = zeros(28,28,1,size(x,4));
    for i = 1 : 3
        channel = squeeze(x(:,:,i,:));
        filter = rot90(net.param1{i}{j},2);
        z = z + convn(channel,filter,'valid');
    end
    net.layers{1}.a{j} = sigm(z + net.b1{j});
    net.layers{1}.a{j} = squeeze(net.layers{1}.a{j});
end

net.fv = [];

for j = 1:32
    z = convn(net.layers{1}.a{j}, ones(2) / (4), 'valid');   %  !! replace with variable
    net.layers{2}.a{j} = z(1 : 2 : end, 1 : 2 : end, :);
end

for j = 1:32
    z = zeros(10,10,size(x,4));
    for i = 1 : 32
        z = z + convn(net.layers{2}.a{i},net.param2(:,:,i,j),'valid');
    end
    net.layers{3}.a{j} = sigm(z + net.b2(j));
end

net.fv = [];

for j = 1:32
    z = convn(net.layers{3}.a{j}, ones(2) / (4), 'valid');   %  !! replace with variable
    net.layers{4}.a{j} = z(1 : 2 : end, 1 : 2 : end, :);
end


for j = 1 : numel(net.layers{4}.a)
    sa = size(net.layers{4}.a{j});
    net.fv = [net.fv; reshaper_row(net.layers{4}.a{j}, sa(1) * sa(2), sa(3))];
end
net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));

size(repmat(net.ffb, 1, size(net.fv, 2)))
size(net.ffW * net.fv )

%result = double(bsxfun(@eq, net.o, max(net.o, [], 1)));
net.errors = 0;

for i = 1:size(y,2)
    result = tiedrank(net.o);
    A = y(:,i);
    index = find(A==max(A));
    if(result(index) < 10)
        net.errors = net.errors+1;
    end
    
%    er = ~all(y(:,i)==result(:,i));
%    net.errors = net.errors+er;
end

end