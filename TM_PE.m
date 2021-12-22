function  [ind_TM_PE_smoothed] = TM_PE(P,window,oneday,dt)

method='parametric';

gamma_hat_0=0;
gamma_hat_1=0;
j=0;
for i=window:oneday:(length(P)) %Use a step of oneday because we re-estimate the parameters of the function every day (and not every 2-min)
j=j+1; % j counts the days
    
P_temp = P(i-window+1:i);
    
[param_hat] = sigma_estimator(window,dt*window,P_temp,method); %in this case n=window. And T changes as well.


gamma_hat_0( (j-1)*oneday+1 : j*oneday ) = param_hat(1);
gamma_hat_1( (j-1)*oneday+1 : j*oneday ) = param_hat(2);
% We have one different estimate of gamma every day.
% However we want to assign a gamma for every return at the 2-min
% frequency.
% So for every return within one day we assign the same value for gamma.

end

ind_TM_PE = nan(length(P)-1,1);
for k=1:length(gamma_hat_1)


    if gamma_hat_1(k) <=1 %attention: what do we do if gamma_hat_1 < 0.5 ??????
    
        ind_TM_PE(window-oneday+k-1) = 1; % PE: Parametric Estimator
    
    else
        
        ind_TM_PE(window-oneday+k-1) = 0;
    end

end
%figure
%plot(gamma_hat_1)

ind_TM_PE(1:window-oneday-1)=ind_TM_PE(window-oneday); % Because I have a rolling window so I assign the same value to the first observations


ind_TM_PE_daily = ind_TM_PE(1:oneday:end);



%%%%%%%%% Hidden Markov Model smoothing %%%%%%%%%
% You need to perform it on daily observations, since you estimate the states on daily basis and this method allows to
% smooth away only short-living deviations 
Trans = [0.995 0.005; 0.005 0.995]; % Probability of transitions of the true hidden state
Emis = [0.9 0.1; 0.01 0.99]; % Probability of emissions
seq_true_states = hmmviterbi(ind_TM_PE_daily,Trans,Emis, 'Statenames',[0,1], 'Symbols',[0,1]);




%%%%%%%%% Stretching %%%%%%%%%
ind_TM_PE_smoothed = nan(length(ind_TM_PE),1);
ind_TM_PE_smoothed(1:oneday-1) = seq_true_states(1);
j=1;
for i=oneday:oneday:length(ind_TM_PE)
    j=j+1; % j counts the days
    
ind_TM_PE_smoothed( (j-1)*oneday : j*oneday-1 ) = seq_true_states(j);
end



% figure
% plot(ind_TM_PE)
% title('Raw signal of the state of PE')
% ylim([-0.1 1.1])
% figure
% plot(ind_TM_PE_smoothed)
% title('Smoothed signal of the state of PE (after hmm)')
% ylim([-0.1 1.1])


end