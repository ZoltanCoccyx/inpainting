list = dir('a*.bmp');

for i = 1:length(list),
   name = list(i).name;
   u = double(imread(name));
   % uR = squeeze(u(:,:,1));
   
   ua = rgb2ycbcr(u);
   ua = squeeze(ua(:,:,3));

   msk = double((ua > 112.5) & (ua < 112.6));
%   msk = imdilate(msk,[1 1 1; 1 1 1; 1 1 1]);
%    msk = imdilate(msk,[0 1 0; 1 1 1; 0 1 0]);

   msk_name = [name(1:4) '_msk.bmp'];
   imwrite(uint8(255*msk),msk_name);

   figure(1), imshow(uint8(u))
   figure(2), imshow(ua.*msk + (1 - msk)*60,[])
   figure(3), imshow(uint8(u.*repmat(1 - msk,[1,1,3])),[])
   pause
end
