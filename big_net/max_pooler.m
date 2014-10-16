function ret = max_pooler(A,stride,window)
ret = zeros(size(A,1)/stride,size(A,2)/stride,size(A,3));

for i = 1:size(ret,1)
    for j = 1:size(ret,2)
        x = (stride*(i-1))+1;
        y = (stride*(j-1))+1;
        patch = A(x:x+window-1,y:y+window-1,:);
        ret(i,j,:) = max(max(patch));
    end
end
end