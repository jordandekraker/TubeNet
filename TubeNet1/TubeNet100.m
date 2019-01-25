clear; close all;

% network parameters
M_sz = [16,16,4];
S_sz = [32 32];
H_sz = [2000];


M = zeros(M_sz); M(4,4,4) = 1; M = reshape(M,[prod(M_sz),1]);
S = fixeye(M,S_sz,M_sz);
% imagesc(reshape(S,S_sz));
rollingR = 1;


Snet = feedforwardnet(H_sz);
Snet.trainParam.showWindow=0;
Snet.trainParam.epochs=1;
Snet.trainParam.min_grad = 0;
% Snet.performParam.regularization = 0.1;

Snet = adapt(Snet,rand([prod(M_sz)+prod(S_sz) 2]),rand([prod(S_sz) 2])); %first one has to contain 2+ training examples for some reason..

Mnet = Snet; %just duplicate


figure; hold on; colormap('gray')
for iters = 0:500
    %% first look at existing network state and choose Mnew
    Mnew = zeros([prod(M_sz),1]);
    if rand<1/(iters/50 +10) || iters==0
        Mnew(randi(prod(M_sz),1)) = 1;
    else
        [~,Mind] = max(Mnet([M;S]));
        Mnew(Mind) = 1;
    end
    
    %% train Snet based on old prediction and newM
    Snew = fixeye(M,S_sz,M_sz);
    [Snet,Spred,Sloss] = adapt(Snet,[Mnew;S],Snew);
    Sloss = Sloss.^2;
    
    %% train Mnet to find M that causes Snet to fail
    R = mean(Sloss);
    rollingR = 0.99*rollingR + 0.01*R;
    R = R - rollingR;
    Mtarget = Mnew + Mnew*R;
    [Mnet,Mpred,Mloss] = adapt(Mnet,[M;Spred],Mtarget);
    Mloss = Mloss.^2;
    
    %% update for next iteration
    S = Snew; M = Mnew;
    
    if rem(iters,1)==0
    subplot(1,2,1); imagesc(reshape(Snew,S_sz)); axis equal tight off; 
    subplot(1,2,2); imagesc(reshape(Spred,S_sz)); axis equal tight off; 
    title(sprintf('iteration: %d \nSloss: %f \nMloss: %f',iters,mean(Sloss),mean(Mloss)))
    drawnow;
    end
end