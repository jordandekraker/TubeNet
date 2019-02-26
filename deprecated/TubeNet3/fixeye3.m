function [out,Fstate] =  fixeye3(M,IO_sz,Fstate)

Mfix_sz = IO_sz{1};
Mhead_sz = IO_sz{2};
Mmot_sz = IO_sz{3};
S_sz = IO_sz{4};
img_sz = [480 640];

IO_sz = {Mfix_sz Mhead_sz Mmot_sz S_sz};
for n = 1:length(IO_sz)
    L_sz(n) = prod(IO_sz{n});
end
L_sz = cumsum(L_sz);

Mfix = reshape(M(1:L_sz(1)),Mfix_sz);
Mhead = reshape(M(L_sz(1)+1:L_sz(2)),Mhead_sz);
Mmot = reshape(M(L_sz(2)+1:L_sz(3)),Mmot_sz);

t = tcpip('192.168.1.1',2001);
t.Timeout = 5;

%% update fixation
min_Mfix = [1 1 1];
max_Mfix = [img_sz 10];

[x,y,z] = ind2sub(Mfix_sz,find(Mfix==1));

% absolute position (eg. given by proprioception)
f = [x,y,z]./Mfix_sz; 
f = round(f.*max_Mfix);
Fstate{1} = f;

% % relative postion (eg. inferred from memory)
% x = x - ceil(Mfix_sz(1)/2); 
% y = y - ceil(Mfix_sz(2)/2); 
% z = z - ceil(Mfix_sz(3)/3);
% Fstate{1} = Fstate{1}+[x y z];
% Fstate{1}(Fstate{1}<min_Mfix) = min_Mfix(Fstate{1}<min_Mfix);
% Fstate{1}(Fstate{1}>max_Mfix) = max_Mfix(Fstate{1}>max_Mfix);

%% update head
min_Mhead = [1 1];
max_Mhead = [90 160];

[x,y] = ind2sub(Mhead_sz,find(Mhead==1));

% absolute position (eg. given by proprioception)
f = [x,y]./Mhead_sz; 
f = round(f.*max_Mhead);
Fstate{2} = f;

% % relative postion (eg. inferred from memory)
% x = x - ceil(Mhead_sz(1)/2); 
% y = y - ceil(Mhead_sz(2)/2); 
% Fstate{2} = Fstate{2}+[x y];
% Fstate{2}(Fstate{2}<min_Mhead) = min_Mhead(Fstate{2}<min_Mhead);
% Fstate{2}(Fstate{2}>max_Mhead) = max_Mhead(Fstate{2}>max_Mhead);

fopen(t);
fwrite(t,[255,1,7,Fstate{2}(1),255]); %tilt
fwrite(t,[255,1,8,Fstate{2}(2),255]); %pan
fclose(t);

%% update mot (always relative; must be inferred from memory)
speed = 1/8;

[x,y] = ind2sub(Mmot_sz,find(Mmot==1));
x = x - ceil(Mmot_sz(1)/2); 
y = y - ceil(Mmot_sz(2)/2); 

if x<0
    fopen(t);
    fwrite(t,[255,0,3,0,255]); %left
%     fclose(t);
    pause(abs(x)*speed);
%     fopen(t);
    fwrite(t,[255,0,0,0,255]); %stop
    fclose(t);
elseif x>0
    fopen(t);
    fwrite(t,[255,0,4,0,255]); %right
%     fclose(t);
    pause(abs(x)*speed);
%     fopen(t);
    fwrite(t,[255,0,0,0,255]); %stop
    fclose(t);
% end
elseif y<0
    fopen(t);
    fwrite(t,[255,0,1,0,255]); %forward
%     fclose(t);
    pause(abs(y)*speed);
%     fopen(t);
    fwrite(t,[255,0,0,0,255]); %stop
    fclose(t);
elseif y>0
    fopen(t);
    fwrite(t,[255,0,2,0,255]); %backward
%     fclose(t);
    pause(abs(y)*speed);
%     fopen(t);
    fwrite(t,[255,0,0,0,255]); %stop
    fclose(t);
end

%% get stream
img = rgb2gray(snapshot(ipcam('http://192.168.1.1:8080/?action=stream')));

% fisheye and crop
out = lensdistort(img,Fstate{1}(3),Fstate{1}(1:2),'bordertype','fit','padmethod', 'fill');
out(:,all(out==0,1)) = [];
out(all(out==0,2),:) = [];

% normalize
out = imresize(double(img),S_sz);
out = out(:);
out = zscore(out);
