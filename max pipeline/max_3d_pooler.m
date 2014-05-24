function [a,b] = max_3d_pooler(A,pool)

a= zeros(size(A,1)/pool,size(A,2)/pool,size(A,3));
b= zeros(size(A,1)/pool,size(A,2)/pool,size(A,3));
for i = 1:size(A,1)/pool
    for j = 1:size(A,2)/pool
        temp = A(i*pool-(pool-1):i*pool,j*pool-(pool-1):j*pool,:);
        new_temp = reshape(temp,size(temp,1)*size(temp,2),size(temp,3));
        [a(i,j,:) b(i,j,:)] = max(new_temp);
        %[x(i,j) max_tracker(i,j)]=max(temp);
    end
end
end