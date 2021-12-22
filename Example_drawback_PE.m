

%%%%%% Simple example showing the failure of the Parametric Estimator
%%%%%% on an extremely simplified example (just for illustration purposes)



%%

clear all
close all
clc

%%


frequency = 2; %Frequency of the data (in minutes)
Years=1.5;
T = Years; % In years
NT = Years*248*6.5*60/frequency;% Number of time steps
dt = T/NT; %dt = time interval of the data (in minutes)


P0 = 100;    % Initial price

%% Estimate sigma(x) with the Parametric Estimator (PE)

close all
rng(2)
% rng(8) for 1Y is good
% rng(2) for 1.5Y is good as well
regime='switch';

NS=1;

gamma_normal =   [0.15   1.1;...
                  0.15   1.1;...
                  0.15   1.1;...
                  0.15   0.9;...
                  0.15   0.9;...
                  0.15   0.9];

        
gamma_crisis = gamma_normal;



Transition_matrix =    [0.996 0.002 0.002  0.0000 0.0000 0.00000;...
                            0.002 0.996 0.002  0.0000 0.0000 0.0000;...
                            0.0000 0.0039 0.996  0.0001 0.0000 0.0000;...
                            0.0000 0.0000 0.0001 0.996 0.0039 0.0000;...
                            0.0000 0.0000 0.0000 0.002 0.996 0.002;...
                            0.0000 0.0000 0.0000 0.002 0.002 0.996];

                        

[P,True_martingale_times, Path_mc]=Diffusion_sim(P0,NT,NS,dt,gamma_normal,gamma_crisis,regime,Transition_matrix);
r_ln = log(P(2:end,:) ./ P(1:end-1,:));%Vector of log-returns for all the paths



oneday=195;
window = oneday*30; % using xx days
[ind_TM_PE] = TM_PE(P,window,oneday,dt);    


%% Classify the data with the NN
fc = 10^4; % scaling constant

load net_1LSTM_80units


% Here I classify the data in either (i) True martingale (TM) or (ii) Strict local martingale (SLM)
XTemp_test = {0};
% YTemp_test={0};
for i=1:NS

   XTemp_test(i)= { [(r_ln(:,i).*fc)'; ((P(2:end,i)./P0)*100)']}; % Use both RETURNS and PRICES of the stochastic process as training data
    
end
XTest = XTemp_test'; 

YPred = classify(net,XTest{1});

% for the regimes predicted by the network
pos_TM_YPred = find(YPred=='TM')'; 
ind_TM_NN = zeros(length(P)-1,1);
ind_TM_NN(pos_TM_YPred)=1;



%% Plot to compare


xaxis=1:length(True_martingale_times)-1;

size_font=23;

figure
subplot(4,1,1);
plot(xaxis,r_ln,'o','Markersize',2)
title('Log-returns')
%xlabel('time')
ax = gca;
ax.FontSize = size_font;


subplot(4,1,2);
plot(xaxis,True_martingale_times(2:end),'LineWidth',5.5)
title("$\mathbf{ \xi = 1_{\{ 1/2 \leq  \gamma_1 \leq 1 \}} }$","Interpreter","latex")
%xlabel('time')
ylim([-0.1 1.1])
ax = gca;
ax.FontSize = size_font;

subplot(4,1,3);
plot(xaxis,ind_TM_PE,'LineWidth',5.5)
title("$\mathbf{ \hat{\xi}^{PE} }$","Interpreter","latex")
%xlabel('time')
ylim([-0.1 1.1])
ax = gca;
ax.FontSize = size_font;

subplot(4,1,4);
plot(xaxis,ind_TM_NN,'LineWidth',5.5)
title("$\mathbf{ \hat{\xi}^{NN} }$","Interpreter","latex")
%xlabel('time')
ylim([-0.1 1.1])
ax = gca;
ax.FontSize = size_font;



figure
plot(xaxis,Path_mc(2:end,1))
title('Path Markov Chain')
ylim([0.9 6.1])





