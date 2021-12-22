function [raw_returns , standard_returns ,P_recovered]=remove_jumps(P,dt,net_jumps,fc_jumps) 


% This function removes the jumps from the returns


% P is a vector of ORIGINAL prices indexed by time

% Compute the Bipower Variation for the path
K_window = ceil(dt^(-0.5));
Bipower_StD=nan(length(K_window+1:length(P(:,1))),1);
i=0;
for ti=K_window+1:length(P) % Jump test time
    i=i+1;
    Bipower_StD(i) = BV(K_window,P,ti);
end

r_ln = log(P(2:end) ./ P(1:end-1));%Vector of log-returns for the unique path


   
X_jumps = ( (r_ln(K_window:end,:)./Bipower_StD) .*fc_jumps)'; % Using the RETURNS as test data
YPred_jumps = classify(net_jumps,X_jumps); % YPred is a vector containing the jumps DETECTED by the NN


% for the jumps predicted by the network
pos_jump_YPred = find(YPred_jumps=='jump')';

X_no_jumps=X_jumps;
%X_no_jumps(pos_jump_YPred)=0; % To set the returns containing a jump equal to 0.


X_no_jumps(pos_jump_YPred)=X_jumps(pos_jump_YPred-1);



standard_returns=X_no_jumps./fc_jumps;

raw_returns=(standard_returns.*Bipower_StD')';

% 
% figure
% plot(X_jumps,'o','Markersize',2)
% title('with jumps')
% ylim([-250 250])
% figure
% plot(X_no_jumps,'o','Markersize',2)
% title('no jumps')
% ylim([-250 250])

%% Recontruct the price path given the new time-series of returns without jumps

% P_recovered is the synthetic price (counter-factual price) path that is consistent with the
% returns without jumps.

P_recovered=nan(length(P),1);
P_recovered(1:K_window)=P(1:K_window); %Since we don't remove the jumps from the first K_window returns,
% we just keep the original prices and returns for the first K_window observations.

for t=K_window+1:length(P)

    % Recover the price path from (i) the first price P0 & (ii) the history of returns
    P_recovered(t) = exp( raw_returns(t-K_window) + log(P_recovered(t-1)) ); 
    
end

raw_returns = [r_ln(1:K_window-1) ; raw_returns];



end