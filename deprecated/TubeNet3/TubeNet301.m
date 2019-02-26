clear; close all;

% network parameters
Mfix_sz = [5 5 5];
Mhead_sz = [5 5];
Mmot_sz = [5 5 5];
S_sz = [50 50];
H_sz = [2000];
C_sz = [20];

L_sz = prod(Mfix_sz)+prod(S_sz);

Slayers = [ ...
    sequenceInputLayer(L_sz)
    fullyConnectedLayer(L_sz)
    lstmLayer(prod(C_sz),'OutputMode','sequence')
    fullyConnectedLayer(round(L_sz/2))
    lstmLayer(prod(C_sz),'OutputMode','sequence')
    fullyConnectedLayer(round(L_sz/4))
    lstmLayer(prod(C_sz),'OutputMode','sequence')
    fullyConnectedLayer(round(L_sz/8))
    lstmLayer(prod(H_sz),'OutputMode','sequence')
    fullyConnectedLayer(round(L_sz/8))
    lstmLayer(prod(C_sz),'OutputMode','sequence')
    fullyConnectedLayer(round(L_sz/4))
    lstmLayer(prod(C_sz),'OutputMode','sequence')
    fullyConnectedLayer(round(L_sz/2))
    lstmLayer(prod(C_sz),'OutputMode','sequence')
    fullyConnectedLayer(L_sz)
    regressionLayer];

options = trainingOptions('adam',...
    'MaxEpochs',1,...
    'MiniBatchSize',1,...
    'Verbose',false);

Snet = trainNetwork(rand([L_sz 1]),rand([L_sz 1]),Slayers,options);

% initialize
M = zeros(Mfix_sz); M(1,1,1) = 1; M = reshape(M,[prod(Mfix_sz),1]);
Mfix_state = [100 100 5]; %max is [200 200 10]
[S,Mfix_state] = fixeye3_smiley(M,S_sz,Mfix_sz,Mfix_state);
% imagesc(reshape(S,S_sz));
rollingR = 1;

%%

figure; hold on; colormap('gray')
for iters = 0:50000

    %% train Snet based on old prediction and newM
    [Snew,Mfix_state] = fixeye3_smiley(M,S_sz,Mfix_sz,Mfix_state);
    [Snet,Spred] = predictAndUpdateState(Snet,[S;M]);
    
    %% reinforce choice of Mfix for bad Spred (relative to a rolling average)
    Mnew = zeros([prod(Mfix_sz),1]);
    if rand<1/(iters/50 +10) || iters==0
        Mnew(randi(prod(Mfix_sz),1)) = 1;
    else
        [~,Mind] = max(Spred(prod(S_sz)+1:end));
        Mnew(Mind) = 1;
    end
    R = mean([Snew-Spred(1:prod(S_sz))].^2);
    rollingR = 0.99*rollingR + 0.01*R;
    R = R - rollingR;
    Mtarget = Mnew + Mnew*R;
    
    [Snet,i] = trainNetwork([S;M],[Snew;Mtarget],Snet.Layers,options); %note this sets state to 0
    loss = i.TrainingLoss;
    
    %% update for next iteration
    S = Snew; M = Mnew;
    
    if rem(iters,1)==0
    subplot(1,2,1); imagesc(reshape(Snew,S_sz)); axis equal tight off; 
    subplot(1,2,2); imagesc(reshape(Spred(1:prod(S_sz)),S_sz)); axis equal tight off; 
    title(sprintf('iteration: %d, loss: %f',iters,loss))
    drawnow;
    end
end

%%
save('TubeNet3');