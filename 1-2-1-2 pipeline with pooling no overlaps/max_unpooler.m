function [x] = max_unpooler(A,max_tracker,pool)

count=1;
for i = 1:size(A,1)
    for j = 1:size(A,2)
        temp = zeros(pool,pool);
        temp(max_tracker(i,j)) = A(i,j);
        x(i*pool-(pool-1):i*pool,j*pool-(pool-1):j*pool) = temp;
        count = count+1;
    end
end

end
