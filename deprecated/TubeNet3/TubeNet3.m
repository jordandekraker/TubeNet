clear; close all;

% network parameters
M_sz = [15 15 15];
S_sz = [32 32];
H_sz = [2000];

layers = [ ...
    sequenceInputLayer(prod(M_sz)+prod(S_sz))
    lstmLayer(prod(H_sz),'OutputMode','sequence')
    fullyConnectedLayer(prod(S_sz))
    regressionLayer];

options = trainingOptions('adam',...
    'MaxEpochs',1,...
    'MiniBatchSize',1,...
    'Verbose',false);

Snet = trainNetwork(rand([prod(M_sz)+prod(S_sz) 1]),rand([prod(S_sz) 1]),layers,options);


Mfix = zeros(M_sz); Mfix(1,1,1) = 1; Mfix = reshape(Mfix,[prod(M_sz),1]);
Mfix_state = [100 100 10];
[S,Mfix_state] = fixeye3_smiley(Mfix,S_sz,M_sz,Mfix_state);
% imagesc(reshape(S,S_sz));
rollingR = 1;

%%

figure; hold on; colormap('gray')
for iters = 0:50000
    %% choose random new Mfix change
    Mnew = zeros([prod(M_sz),1]);
    Mnew(randi(prod(M_sz),1)) = 1;
    
    %% train Snet based on old prediction and newM
    [Snew,Mfix_state] = fixeye3_smiley(Mfix,S_sz,M_sz,Mfix_state);
    [Snet,Spred] = predictAndUpdateState(Snet,[S;Mnew]);
    [Snet,i] = trainNetwork([S;Mnew],Snew,Snet.Layers,options); %note this sets state to 0
    Sloss = i.TrainingLoss;
    
    %% update for next iteration
    S = Snew; Mfix = Mnew;
    
    if rem(iters,1)==0
    subplot(1,2,1); imagesc(reshape(Snew,S_sz)); axis equal tight off; 
    subplot(1,2,2); imagesc(reshape(Spred,S_sz)); axis equal tight off; 
    title(sprintf('iteration: %d, Sloss: %f',iters,Sloss))
    drawnow;
    end
end

%%
save('TubeNet3');