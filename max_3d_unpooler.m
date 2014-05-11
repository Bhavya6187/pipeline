function [a] = max_3d_unpooler(A,max_tracker,pool)

count=1;
for i = 1:size(A,1)
    for j = 1:size(A,2)
        temp = zeros(pool*pool,size(A,3));
        for k = 1:size(A,3)
            temp(max_tracker(i,j,k),k) = A(i,j,k);
        end
        temp = reshape(temp,pool,pool,size(A,3));
        a(i*pool-(pool-1):i*pool,j*pool-(pool-1):j*pool,:) = temp;
        count = count+1;
    end
end
end