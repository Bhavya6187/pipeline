function [a,b] = max_3d_pooler(A,stride,window)
A_pad = padarray(A,[window-stride window-stride],'post');
a= zeros(size(A,1)/stride,size(A,2)/stride,size(A,3));
b= zeros(size(A,1)/stride,size(A,2)/stride,size(A,3));
for i = 1:stride:size(A,1)
    for j = 1:stride:size(A,2)
        temp = A_pad(i:i+window-1,j:j+window-1,:);
        new_temp = reshape(temp,size(temp,1)*size(temp,2),size(temp,3));
        i_ind = ((i-1)/stride)+1;
        j_ind = ((j-1)/stride)+1;
        [a(i_ind,j_ind,:) b(i_ind,j_ind,:)] = max(new_temp);
    end
end
end