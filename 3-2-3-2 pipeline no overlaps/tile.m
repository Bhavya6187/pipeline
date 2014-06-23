function A = tile(B,nx,ny,overlap)
%tile: split a 2D array into equally-sized subarrays.
%A = tile(B,nx,ny,overlap)
%input:
%B = 2D array
%nx = number of tiles in column direction
%ny = number of columns in row direction
%overlap = ammount of overlap of tiles in pixels
%OUTPUT
%A = cell array of size ny by nx with each cell containing a tile.

%Ian Howat, Ohio State University, ihowat@gmail.com

nl = size(B,1);
ns = size(B,2);

if ny > 1
   maxl = floor([(nl./ny).*(1:ny-1)+overlap, nl]);
   minl = [1,maxl(1:end-1)+1-overlap];
else
   maxl = nl;
   minl = 1;
end

if nx > 1
   maxs = floor([(ns/nx)*(1:nx-1)+overlap, ns]);
   mins = [1,maxs(1:end-1)+1-overlap];
else
   maxs = ns;
   mins = 0;
end



for i=1:length(minl)
   for j=1:length(mins)
   
    A{i,j} =  B(minl(i):maxl(i),mins(j):maxs(j));

   end
end  
