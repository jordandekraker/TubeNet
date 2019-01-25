% Mfix = zeros([16,16,3]);
% Mfix(4,4,1) = 1;
% S_sz = [32 32];


function smiley =  fixeye(Mfix,S_sz,M_sz)
smiley = imread('SmileyFace8bitGray.png');
[x,y,z] = ind2sub(M_sz,find(Mfix==1));

% scale to appropriate sizes
fix = size(smiley).*[x y]./[M_sz(1) M_sz(2)];
z = 1/((z-0.5)^2);

smiley = lensdistort(smiley,z,fix,'bordertype','fit','padmethod', 'replicate');

smiley(:,all(diff(smiley)==0,1)) = [];
smiley(all(diff(smiley)==0,2),:) = [];

smiley = imresize(double(smiley),S_sz);
smiley = smiley(:);
smiley = zscore(smiley);

% figure; imagesc(reshape(smiley,S_sz))

