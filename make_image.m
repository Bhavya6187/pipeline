function image = make_image(im)

image = zeros(32,32,3);
for i=1:32
    for j = 1:32
        for k = 1:3
            image(i,j,k)=im(1024*(k-1)+(i-1)*32+j);
        end
    end
end