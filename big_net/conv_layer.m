function ret = conv_layer(in,filter,bias,stride,pad,ntype)

ret = zeros( ((size(in,1) + 2*pad - size(filter,1))/stride)+1, ((size(in,2) + 2*pad - size(filter,2))/stride) + 1,size(filter,4));

group = size(in,3)/size(filter,3);

in_g = size(in,3)/group;
out_g = size(filter,4)/group;

for g = 1:group
    for i = out_g*(g-1)+1:out_g*g
        filter_index = 1;
        for j = in_g*(g-1)+1:in_g*g
            temp = conv(in(:,:,j),filter(:,:,filter_index,i),stride,pad);
            ret(:,:,i) = ret(:,:,i)+temp;
            filter_index = filter_index +1;
            
        end
        ret(:,:,i) = ret(:,:,i) + bias(i);
        if mod(i,100) == 0
            i
        end
    end
end

size(ret)
if strcmp(ntype,'relu')
    for i = 1:size(filter,4)
        ret(:,:,i) = max(ret(:,:,i),0);
    end
end

if strcmp(ntype,'sigm')
    ret = sigm(ret);
end
