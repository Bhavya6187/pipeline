function [] = max_3d_pooler(A,pool)


for i = 1:size(A,1)/pool
    for j = 1:size(A,2)/pool
        temp = A(i*pool-(pool-1):i*pool,j*pool-(pool-1):j*pool,:)
        %[x(i,j) max_tracker(i,j)]=max(temp);
    end
end
end