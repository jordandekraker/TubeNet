Mfix = zeros([16,16,3]);
Mfix(4,4,1) = 1;
S_sz = [32 32];
M_sz = [16,16,4];
i=0;

function out =  fixeye2(Mfix,S_sz,M_sz,i)

%% random head move
if rem(i,50)==0
    t = tcpip('192.168.1.1',2001);
    fopen(t);
    fwrite(t,[255,1,7,randi(160),255]);
    fwrite(t,[255,1,8,randi(90),255]);
    fclose(t);
end

%% get stream
img = rgb2gray(snapshot(ipcam('http://192.168.1.1:8080/?action=stream')));


%% scale to appropriate sizes
[x,y,z] = ind2sub(M_sz,find(Mfix==1));
fix = size(img).*[x y]./[M_sz(1) M_sz(2)];
z = 1/((z-0.5)^2);

%% fisheye and crop
out = lensdistort(img,z,fix,'bordertype','fit','padmethod', 'fill');
out(:,all(out==0,1)) = [];
out(all(out==0,2),:) = [];

%% normalize
out = imresize(double(img),S_sz);
out = out(:);
out = zscore(out);
