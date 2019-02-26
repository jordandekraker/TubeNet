t = tcpip('192.168.1.1',2001);
t.Timeout = 5;

fopen(t);
fwrite(t,[255,1,7,45,255]); %tilt
fwrite(t,[255,1,8,90,255]); %pan
fclose(t);

fopen(t);
fwrite(t,[255,0,4,0,255]); %left
pause(0.2);
fwrite(t,[255,0,0,0,255]); %stop
fclose(t);

% img = rgb2gray(snapshot(ipcam('http://192.168.1.1:8080/?action=stream')));
% imagesc(img); colormap('gray');