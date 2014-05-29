function [a] = max_3d_unpooler(A,max_tracker,stride,window)

count=1;
a = zeros(size(A,1)*stride+window-stride,size(A,2)*stride+window-stride,size(A,3));
for i = 1:size(A,1)
    for j = 1:size(A,2)
        temp = zeros(window*window,size(A,3));
        for k = 1:size(A,3)
            temp(max_tracker(i,j,k),k) = A(i,j,k);
        end
        temp = reshape(temp,window,window,size(A,3));
        i_ind = i*stride-1;
        j_ind = j*stride-1;
        a(i_ind:i_ind+window-1,j_ind:j_ind+window-1,:) = a(i_ind:i_ind+window-1,j_ind:j_ind+window-1,:)+temp;
        count = count+1;
    end
end
a = a(1:size(A,1)*stride,1:size(A,2)*stride,:);
end