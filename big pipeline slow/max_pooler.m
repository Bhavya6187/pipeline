function [x,max_tracker] = max_pooler(A,pool)

B = tile(A,size(A,1)/pool,size(A,2)/pool,0);

x = ones(size(B));
max_tracker = ones(size(B));

for i = 1:size(B,1)
    for j = 1:size(B,2)
        temp = reshape(B{i,j},pool*pool,1);
        [x(i,j) max_tracker(i,j)]=max(temp);
    end
end