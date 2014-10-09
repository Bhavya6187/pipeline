function net = cnnff(net,x,y)


%x = padarray(x,[2 2],'both');

%{
for j = 1:32
    for i = 1:size(x,4)
        for l = 1:28
            for m = 1:28
                for k = 1:3
                    z = 0;
                    for n = 1:5
                        for o = 1:5
                            z = z + net.param1{k}{j}(n,o)*x(l+n-1,m+o-1,k,i);
                        end
                    end
                    net.layers{1}.a{j}(l,m,i) = sigm(z + net.b1{j});
                end
            end
        end
    end
end
%}

for j = 1:32
    z = zeros(28,28,size(x,4));
    for i = 1 : 3
        %temp = rot90(net.param1{i}{j},2);
        temp = net.param1{i}{j};
        channel = squeeze(x(:,:,i,:));
        %means = mean(mean(mean(channel,1)));
        %channel = channel - means;
        %channel = channel/255;
        z = z + convn(channel,temp,'valid');
    end
    net.layers{1}.a{j} = sigm(z + net.b1{j});
    net.layers{1}.a{j} = net.layers{1}.a{j};
end

net.fv = [];
size(net.layers{1}.a{j})
for j = 1:32
    z = convn(net.layers{1}.a{j}, ones(2) / (4), 'valid');   %  !! replace with variable
    net.layers{2}.a{j} = z(1 : 2 : end, 1 : 2 : end, :);
end

size(net.layers{2}.a{j})
%{
for j = 1:32
    for i = 1:size(x,4)
        for l = 1:10
            for m = 1:10
                for k = 1:32
                    z = 0;
                    for n = 1:5
                        for o = 1:5
                            z = z + net.param2{k}{j}(n,o)*net.layers{2}.a{k}(l+n-1,m+o-1,i);
                        end
                    end
                    net.layers{3}.a{j}(l,m,i) = sigm(z + net.b2{j});
                end
            end
        end
    end
end
%}

for j = 1:32
    z = zeros(10,10,size(x,4));
    for i = 1 : 32
        %temp = rot90(net.param2{i}{j},2);
        temp = net.param2{i}{j};
        z = z + convn(net.layers{2}.a{i},temp,'valid');
    end
    net.layers{3}.a{j} = sigm(z + net.b2{j});
end

net.fv = [];

for j = 1:32
    z = convn(net.layers{3}.a{j}, ones(2) / (4), 'valid');   %  !! replace with variable
    net.layers{4}.a{j} = z(1 : 2 : end, 1 : 2 : end, :);
end


for j = 1 : numel(net.layers{4}.a)
    sa = size(net.layers{4}.a{j});
    net.fv = [net.fv; reshape(net.layers{4}.a{j}, sa(1) * sa(2), sa(3))];
end
net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));

%{
result = double(bsxfun(@eq, net.o, max(net.o, [], 1)));
net.errors = 0;

for i = 1:size(y,2)
    er = ~all(y(:,i)==result(:,i));
    net.errors = net.errors+er;
end
net.errors
%}
end