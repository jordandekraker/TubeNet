clear; close all;


%% network parameters (only run when restarting training)
reload_last = false;

if ~reload_last
    Mfix_sz = [25 25 5];
    Mhead_sz = [25 25];
    Mmot_sz = [3 3];
    S_sz = [32 32];
    
    H_sz = [2000];
    C_sz = [20];
    
    % build the network
    IO_sz = {Mfix_sz Mhead_sz Mmot_sz S_sz};
    for n = 1:length(IO_sz)
        L_sz(n) = prod(IO_sz{n});
    end
    L_sz = cumsum(L_sz);
    
    Slayers = [ ...
        sequenceInputLayer(L_sz(end))
        fullyConnectedLayer(L_sz(end))
        lstmLayer(prod(C_sz),'OutputMode','sequence')
        fullyConnectedLayer(round(L_sz(end)/2))
        lstmLayer(prod(C_sz),'OutputMode','sequence')
        fullyConnectedLayer(round(L_sz(end)/4))
        lstmLayer(prod(C_sz),'OutputMode','sequence')
        fullyConnectedLayer(round(L_sz(end)/8))
        lstmLayer(prod(H_sz),'OutputMode','sequence')
        fullyConnectedLayer(round(L_sz(end)/8))
        lstmLayer(prod(C_sz),'OutputMode','sequence')
        fullyConnectedLayer(round(L_sz(end)/4))
        lstmLayer(prod(C_sz),'OutputMode','sequence')
        fullyConnectedLayer(round(L_sz(end)/2))
        lstmLayer(prod(C_sz),'OutputMode','sequence')
        fullyConnectedLayer(L_sz(end))
        regressionLayer];
    
    options = trainingOptions('adam',...
        'MaxEpochs',1,...
        'MiniBatchSize',1,...
        'Verbose',false);
    
    Snet = trainNetwork(rand([L_sz(end) 1]),rand([L_sz(end) 1]),Slayers,options);
    
    % initialize
    Mfix = zeros(Mfix_sz); Mfix(1,1,1) = 1; Mfix = reshape(Mfix,[prod(Mfix_sz),1]);
    Mhead = zeros(Mhead_sz); Mhead(1,1) = 1; Mhead = reshape(Mhead,[prod(Mhead_sz),1]);
    Mmot = zeros(Mmot_sz); Mmot(1,1) = 1; Mmot = reshape(Mmot,[prod(Mmot_sz),1]);
    M = [Mfix;Mhead;Mmot];
    
    Fstate = {[240 320 5]  [45 90]};
    [S,Fstate] = fixeye3(M,IO_sz,Fstate);
    % imagesc(reshape(S,S_sz));
    rollingR = 1;
    total_iters = 0;
    
elseif reload_last
    fn = strsplit(ls('TubeNet302_iteration*'));
    load(fn{end-1});
end

%% live
maxiters = 10000; maxWifiLoss = 50; wl = 0;
IsItBroken = false;
figure; hold on; colormap('gray')
while ~IsItBroken
for iters = 0:maxiters

    %% act, get Snew, get Spred
    try
        [Snew,Fstate] = fixeye3(M,IO_sz,Fstate);
    catch 
        wl = wl+1;
        if wl>maxWifiLoss
            IsItBroken = true;            
        end
        disp(['wifi signal lost ' datestr(clock)]);
        break
    end
    [Snet,Spred] = predictAndUpdateState(Snet,[S;M]);
    
    %% reinforce old M for bad Spred (relative to a rolling average)
    R = mean([Snew-Spred(L_sz(3)+1:L_sz(4))].^2);
    rollingR = 0.99*rollingR + 0.01*R;
    R = R - rollingR;
    Mtarget = M+M*R;
    
    %% update each motor component
    Mfix_new = zeros([prod(Mfix_sz),1]);
    if rand<1%/(iters/50 +10) || iters==0
        Mfix_new(randi(prod(Mfix_sz),1)) = 1;
    else
        [~,Mind] = max(Spred(1:L_sz(1)));
        Mfix_new(Mind) = 1;
    end
    
    Mhead_new = zeros([prod(Mhead_sz),1]);
    if rand<1%/(iters/50 +10) || iters==0
        Mhead_new(randi(prod(Mhead_sz),1)) = 1;
    else
        [~,Mind] = max(Spred(L_sz(1)+1:L_sz(2)));
        Mhead_new(Mind) = 1;
    end
    
    Mmot_new = zeros([prod(Mmot_sz),1]);
    if rand<1%/(iters/50 +10) || iters==0
        Mmot_new(randi(prod(Mmot_sz),1)) = 1;
    else
        [~,Mind] = max(Spred(L_sz(2)+1:L_sz(3)));
        Mmot_new(Mind) = 1;
    end
    Mnew = [Mfix_new;Mhead_new;Mmot_new];
    
    %% train
    if any(isnan([Mtarget; Snew]))
        disp(['training signal contained NaN ' datestr(clock)]);
        IsItBroken = true;
        break
    end
    [Snet,i] = trainNetwork([S;M],[Mtarget; Snew],Snet.Layers,options); %note this sets state to 0
    loss = i.TrainingLoss;
    
    %% update for next iteration
    S = Snew; M = Mnew;
    
    if rem(iters,1)==0
    subplot(1,2,1); imagesc(reshape(Snew,S_sz)); axis equal tight off; 
    subplot(1,2,2); imagesc(reshape(Spred(L_sz(3)+1:L_sz(4)),S_sz)); axis equal tight off; 
    title(sprintf('iteration: %d, loss: %f',iters,loss))
    drawnow;
    end
end
end

%% save
total_iters = total_iters+iters;
save(sprintf('TubeNet302_i%05d',total_iters));
if iters == maxiters
    disp(['finished all iters ' datestr(clock)]);
end
