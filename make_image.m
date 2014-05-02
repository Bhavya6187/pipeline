a = load('data\data_batch_1.mat');
b = load('data\data_batch_2.mat');
c = load('data\data_batch_3.mat');
d = load('data\data_batch_4.mat');
e = load('data\data_batch_5.mat');

f = load('data\test_batch.mat');

total_x = [a.data; b.data; c.data; d.data; e.data];
train_x = zeros(32,32,3,50000);

for l=1:50000
    train_x(:,:,1,l)=reshape(total_x(l,1:1024),32,32)';
    train_x(:,:,2,l)=reshape(total_x(l,1025:2048),32,32)';
    train_x(:,:,3,l)=reshape(total_x(l,2049:3072),32,32)';
end

total_y = [a.labels; b.labels; c.labels; d.labels; e.labels];

train_y = zeros(50000,10);
for l=1:50000
    train_y(l,total_y(l)+1)=1;
end

total_x = [f.data];
test_x = zeros(32,32,3,10000);

for l=1:10000
    test_x(:,:,1,l)=reshape(total_x(l,1:1024),32,32)';
    test_x(:,:,2,l)=reshape(total_x(l,1025:2048),32,32)';
    test_x(:,:,3,l)=reshape(total_x(l,2049:3072),32,32)';
end

total_y = [f.labels];
test_y = zeros(10000,10);
for l=1:10000
    test_y(l,total_y(l)+1)=1;
end

clear('a','b','c','d','e','f','i','j','k','l');