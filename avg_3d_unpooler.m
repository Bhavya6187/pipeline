function [a] = avg_3d_unpooler(A,stride,window)

count=1;
a = zeros(size(A,1)*stride+window-stride,size(A,2)*stride+window-stride,size(A,3));
for i = 1:size(A,1)
    for j = 1:size(A,2)
        temp = ones(window,window);
        for k = 1:size(A,3)
            temp = ones(window,window)*A(i,j,k);
            i_ind = i*stride-1;
            j_ind = j*stride-1;
            a(i_ind:i_ind+window-1,j_ind:j_ind+window-1,k) = a(i_ind:i_ind+window-1,j_ind:j_ind+window-1,k)+temp;
            count = count+1;
        end
    end
end
a = a(1:size(A,1)*stride,1:size(A,2)*stride,:);
end