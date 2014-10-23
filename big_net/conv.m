function ret = conv(in,filter,stride,pad)

ret = zeros( ((size(in,1) + 2*pad - size(filter,1))/stride)+1, ((size(in,2) + 2*pad - size(filter,2))/stride) + 1);

if pad>0
    in = padarray(in,[pad pad],'both');
end

filter = rot90(filter,2);
for i = 1:size(ret,1)
    for j = 1:size(ret,2)
        x_start= (stride*(i-1));
        y_start= (stride*(j-1));
        patch = in(x_start+1:x_start+size(filter,1),y_start+1:y_start+size(filter,2));
        ret(i,j) = conv2(patch,filter,'valid');
    end
end