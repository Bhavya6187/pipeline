function ret = reshaper_row(in)
    ret = zeros(size(in,1)*size(in,2),size(in,3));
    for i = 1:size(in,3)
        temp = in(:,:,i)';
        s = size(temp,1)*size(temp,2);
        ret(:,i) = reshape(temp,s,1);
        %ret((i-1)*s+1:i*s) = reshape(temp,s,1);
    end
end