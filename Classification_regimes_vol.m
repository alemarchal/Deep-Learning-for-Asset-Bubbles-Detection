%%%%%%%%%% This code classifies the different regimes of the volatility s(x)
%%%%%%%%%% using a neural network.

% The goal is to be able to classify a process as being a true or a strict
% local martingale from discrete data points.

%%

clear all
close all
clc

%% Parameters Monte-Carlo simulation


frequency = 2; %Frequency of the data (in minutes)
Years=3;
T = Years; % In years
NT = Years*248*6.5*60/frequency;% Number of time steps
dt = T/NT; %dt = time interval of the data (in minutes)



P0 = 100;    % Initial firm value
fc = 10^4; % scaling constant


%% Simulate the price path where the functional form of the diffusion coefficient changes through time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Generate training data %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NS=30;
regime='switch';


gamma_normal=[0.25/1.3   1.12;...
                 0.2/1.3    1.09;...
                 0.19/1.3   1.07;...
                 0.35/1.2   0.95;...
                 0.3/1.2    0.92;...
                 0.25/1.2   0.88];

        
gamma_crisis=[0.25*1.8   1.12*1.125;...
                    0.2*1.8    1.09*1.1;...
                    0.19*1.7   1.07*1.1;...
                    0.35*2   0.95;...
                    0.3*2    0.85;...
                    0.25*2   0.8];



Transition_matrix_sim =    [0.996 0.002 0.002  0.0000 0.0000 0.00000;...
                            0.002 0.996 0.002  0.0000 0.0000 0.0000;...
                            0.0000 0.0039 0.996  0.0001 0.0000 0.0000;...
                            0.0000 0.0000 0.0001 0.996 0.0039 0.0000;...
                            0.0000 0.0000 0.0000 0.002 0.996 0.002;...
                            0.0000 0.0000 0.0000 0.002 0.002 0.996];



[P,True_martingale_times, Path_mc]=Diffusion_sim(P0,NT,NS,dt,gamma_normal,gamma_crisis,regime,Transition_matrix_sim);

%%%%%% True_martingale_times contains the time points when the process was a
%%%%%% true martingale (i.e.  =1 when it's a true martingale)


% Compute the log-returns
r_ln = log(P(2:end,:) ./ P(1:end-1,:));%Vector of log-returns for all the paths


% Here I classify the data in either (i) True martingale (TM) or (ii) Strict local martingale (SLM)


XTemp = {0};
for i=1:NS
    
  XTemp(i)= { [(r_ln(:,i).*fc)'; P(2:end,i)']}; % Use both RETURNS and PRICES of the stochastic process as training data
 
end

YTemp={0};
for j=1:NS
  YTemp{j} =categorical(True_martingale_times(2:end,j)',[0 1],{'SLM' 'TM'});  % Use the RETURNS of the stochastic process as training data 
  %(that's why I start from the 2 element in the array)
end

XTrain = XTemp'; % Should be NS*1 cell array, where each cell is 2*NT double, first line = returns, second line = prices
YTrain = YTemp';

% 
% close all
% figure
% plot(True_martingale_times(:,2))
% ylim([-0.1 1.1])
% figure
% plot(Path_mc(:,2))
% ylim([0.9 6.1])
%% Plot training data

which=1; %Which time-series I want to plot

X = XTrain{which}(1,:); % plotting: 1-returns, 2-prices 
classes = categories(YTrain{which});

figure
for j = 1:numel(classes)
    label = classes(j);
    idx = find(YTrain{which} == label);
    hold on
    plot(idx,X(idx),'o','MarkerSize',2)
    %plot(X)
end
hold off

xlabel("Time Step")
ylabel("returns")
legend(classes,'Location','northwest')

%% Define the network structure & Train it


numFeatures = 2; % 2 dimensions because we use both PRICE & RETURNS to train
numHiddenUnits = 70;
numClasses = 2; % 2 different categories (Strict-Local-Martingale or True-Martingale)

layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

whichAx = [false, true]; % [bottom, top]

options = trainingOptions('adam', ...
    'MaxEpochs',800, ...
    'GradientThreshold',2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%,...
%    'MiniBatchSize',128);
%, ...
%    'OutputFcn', @(x)makeLogVertAx(x,whichAx));


% Train the neural network    
net = trainNetwork(XTemp,YTrain,layers,options);

%% Test the network (validation)

% save('net_2LSTM_50units_perlayer','net')

% load net_2LSTM_50units_perlayer

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Generate test data %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamma_normal_test=[0.25/1.4   1.10;...
                 0.2/1.5    1.08;...
                 0.19/1.2   1.065;...
                 0.35/1.1   0.87;...
                 0.3/1.1    0.82;...
                 0.25/1.2   0.7];

        
gamma_crisis_test=[0.25*1.9   1.14*1.11;...
                    0.2*1.5    1.09*1.08;...
                    0.19*1.3   1.07*1.07;...
                    0.35*1.4   0.85;...
                    0.3*1.8    0.75;...
                    0.25*2   0.65];

Transition_matrix_test_1 =    [0.995 0.0025 0.0025  0.0000 0.0000 0.00000;...
                            0.0023 0.995 0.0027  0.0000 0.0000 0.0000;...
                            0.0000 0.0043 0.995  0.0007 0.0000 0.0000;...
                            0.0000 0.0000 0.0011 0.995 0.0039 0.0000;...
                            0.0000 0.0000 0.0000 0.002 0.996 0.002;...
                            0.0000 0.0000 0.0000 0.003 0.003 0.994];

Transition_matrix_test_2 =    [0.9945 0.003 0.0025  0.0000 0.0000 0.00000;...
                               0.002 0.995 0.003  0.0000 0.0000 0.0000;...
                               0.0000 0.0039 0.995  0.0011 0.0000 0.0000;...
                               0.0000 0.0000 0.0006 0.995 0.0044 0.0000;...
                               0.0000 0.0000 0.0000 0.002 0.996 0.002;...
                               0.0000 0.0000 0.0000 0.002 0.002 0.996];

                                               
                        
                        

NS=1;
regime_test='switch';
P0 = 20000;

cm_percentage_NN_sum = zeros(2,2);
cm_percentage_PE_sum = zeros(2,2);

trials=2; % On 150 paths of 3 years, the PE method takes 44 hours

B = binornd(1,1/2,trials,1); % Bernoulli r.v. (or Binomial with 1 trial) that takes the values {0,1} with equal probability 1/2.

method_to_test = 'both'; % 'NN' or 'both'



Ptg_detection_NN=nan(1,trials);
Ptg_spurious_NN=nan(1,trials);
Ptg_detection_PE=nan(1,trials);
Ptg_spurious_PE=nan(1,trials);
for t=1:trials


Transition_matrix_test = B(t) * Transition_matrix_test_1 + (1-B(t)) * Transition_matrix_test_2;
[P_test,True_martingale_times_test,~]=Diffusion_sim(P0,NT,NS,dt,gamma_normal_test,gamma_crisis_test,regime_test,Transition_matrix_test);



r_ln_test = log(P_test(2:end,:) ./ P_test(1:end-1,:));%Vector of log-returns for all the paths

% Here I classify the data in either (i) True martingale (TM) or (ii) Strict local martingale (SLM)
XTemp_test = {0};
% YTemp_test={0};
for i=1:NS

   XTemp_test(i)= { [(r_ln_test(:,i).*fc)'; ((P_test(2:end,i)./P0)*100)']}; % Use both RETURNS and PRICES of the stochastic process as training data
   % We divide price vector by the initial price level and then mult by
   % 100, because the network was trained on the price pathes started from
   % 100. 
end
XTest = XTemp_test'; % Should be NS*1 cell array, where each cell is 2*NT double, first line = returns, second line = prices

YPred = classify(net,XTest{1});

%%%%%%%%%%%%%% The part below is used to label the data in order to produce
%%%%%%%%%%%%%% a graph that says who detected what on SIMULATED DATA

% for the regimes predicted by the network
pos_TM_YPred = find(YPred=='TM')'; 
ind_TM_NN = zeros(length(P_test)-1,1);
ind_TM_NN(pos_TM_YPred)=1;



% figure
% plot(True_martingale_times_test)
% title('True martingale times')
% ylim([-0.1 1.1])
% figure
% plot(ind_TM_NN)
% title('NN martingale times')
% ylim([-0.1 1.1])



%%% Confusion charts %%%
% confusionchart(trueLabels,predictedLabels)
cm_NN = confusionmat(True_martingale_times_test(2:end),ind_TM_NN);
cm_percentage_NN_temp = (cm_NN/length(ind_TM_NN))*100;

Ptg_detection_NN(t) = sum(diag(cm_percentage_NN_temp));
Ptg_spurious_NN(t) = sum(diag(flip(cm_percentage_NN_temp)));

cm_percentage_NN_sum = cm_percentage_NN_sum + cm_percentage_NN_temp;

% classlabels = {'Strict Local Martingale','True Martingale'};
% figure
% cc_NN = confusionchart(cm_NN,classlabels);
% cc_NN.Title = 'Performance of the Neural Network';



if strcmp(method_to_test,'both')
        % Here we ALSO use the Parametric Estimator on the same price path
        %%% Estimate sigma(x) with a parametric method %%%
        oneday=195;
        window = oneday*30; % using xx days
        [ind_TM_PE] = TM_PE(P_test,window,oneday,dt);    


        cm_PE = confusionmat(True_martingale_times_test(2:end),ind_TM_PE);
        cm_percentage_PE_temp = (cm_PE/length(ind_TM_PE))*100;
        
        Ptg_detection_PE(t) = sum(diag(cm_percentage_PE_temp));
        Ptg_spurious_PE(t) = sum(diag(flip(cm_percentage_PE_temp)));
        
        cm_percentage_PE_sum = cm_percentage_PE_sum + cm_percentage_PE_temp;
        
        % figure
        % cc_PE = confusionchart(cm_PE,classlabels);
        % cc_PE.Title = 'Performance of the Parametric Estimator';

else 
   % nothing
    
end
   

end


    cm_percentage_NN = cm_percentage_NN_sum/trials;

    
    Avg_detection_NN=mean(Ptg_detection_NN)
    Avg_spurious_NN=mean(Ptg_spurious_NN)

    
    
    
    
if strcmp(method_to_test,'both')
    
    cm_percentage_PE = cm_percentage_PE_sum/trials;
    
    Avg_detection_PE=mean(Ptg_detection_PE)
    Avg_spurious_PE=mean(Ptg_spurious_PE)
else
     disp('Only the Neural Network was tested')
end


%% Histogram for Confidence Intervals (CI)
bins=20;


% Figure size parameters
x0=100;
y0=100;
width=1200;
height=800;



figure
histogram(Ptg_detection_NN - Ptg_detection_PE, bins)
xlabel("% detection NN - % detection PE")
ax = gca;
ax.FontSize = 18;
set(gcf,'position',[x0,y0,width,height])
saveas(gcf,'hist_bubbles_detection_NN_PE.png');


figure
histogram(Ptg_spurious_NN - Ptg_spurious_PE, bins)
xlabel("% spurious NN - % spurious PE")
ax = gca;
ax.FontSize = 18;
set(gcf,'position',[x0,y0,width,height])
saveas(gcf,'hist_bubbles_spurious_NN_PE.png');



%% Estimate sigma(x) using the non-parametric estimator of Florens-Zmirou


% 
% method='FZ';
% 
% i=0;
% sigma_square=0;
% 
% step=20.0; %a step too small makes the graph very unsmooth. I should choose a large step
% 
% % verify this code (and the function sigma_square) because the results are
% % very unsmooth (not like in the paper) but the values are plausible
% 
% 
% x_domain = min(P_test):step:max(P_test); % points that have been visited by the stock price
% 
% for x = x_domain
% i=i+1;
%     
% sigma_square(i) = sigma_estimator(n,T,P_test,method,x);
% 
% end
% 
% figure
% plot(x_domain,sqrt(sigma_square))


%% Plot the results on test data
% This plot the results on SIMULATED DATA


diff_ind_TM =  ind_TM_NN + 10*ind_TM_PE + 100*True_martingale_times_test(2:end) ;
YDiff=categorical(diff_ind_TM',[0 1 10 11 100 101 110 111],{...
    'SLM & correct both' ...
    'SLM & spurious NN & correct PE' ...
    'SLM & correct NN & spurious PE' ...
    'SLM & spurious both' ...
    'TM  & spurious both' ...
    'TM  & correct NN & spurious PE' ...
    'TM  & spurious NN & correct PE' ...
    'TM  & correct both'
      });




X = XTest{1}(1,:)./fc; % plotting: 1-returns, 2-prices 
classes = categories(YDiff);


exclude = ones(1,length(classes));
choice_color = {'#0072BD','r','#EDB120','#42B92F','k','r','#EDB120','#42B92F'};
choice_shape =  {'o','x'};
choice_size = [4 4];

figure
for j = 1:numel(classes)
    label = classes(j);

    
       idx = find(YDiff == label);
   if isempty(idx)
       exclude(j)=0;
   end
   
    hold on
   p= plot(idx,X(idx),choice_shape{1+double(j>4)},'MarkerSize',choice_size(1+double(j>1)),'color',choice_color{j});
   if j>1
   set(p, 'markerfacecolor', get(p, 'color')); % Use same color to fill in markers
   end
end
xlabel("Time")
ylabel("Returns")
ax = gca;
ax.FontSize = 16;

pos_temp =find(exclude==1);
legend(classes(pos_temp),'Location','northwest')
% I use classes(pos_temp) to avoid the problem of "Warning: Ignoring extra legend entries" which shifts the legend and so messes up the graph.

%% Plot returns for the image with 4 boxes summarizing our entire methodology


% frequency = 2; %Frequency of the data (in minutes)
% Years=1;
% T_plot = Years; % In years
% NT_plot = Years*248*6.5*60/frequency;% Number of time steps
% dt_plot = T_plot/NT_plot; %dt = time interval of the data (in minutes)
% 
% 
% 
% NS=1;
% regime='switch';
% P0 = 100;
% 
% 
% sz_font = 14;
% choice_size = 3.5;
% choice_shape =  {'o'};
%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      %%%% For the LABEL graph %%%%
%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  rng(15) %Freeze the random generator
% 
% % ATTENTION: I need to set the matrices for gamma & the Markov Chain before (I use the ones in the training data section) 
% [P,True_martingale_times,~]=Diffusion_sim(P0,NT_plot,NS,dt_plot,gamma_normal,gamma_crisis,regime,Transition_matrix_sim);
% r_ln = log(P(2:end,:) ./ P(1:end-1,:));
% 
% 
% diff_ind_TM = True_martingale_times(2:end) ;
% YDiff=categorical(diff_ind_TM',[0 1],{...
%     'Bubble' ...
%     'No Bubble' ...
%       });
% 
% 
% X = r_ln;
% classes = categories(YDiff);
% 
% 
% exclude = ones(1,length(classes));
% choice_color = {'r','#0072BD'};
% 
% 
% figure
% for j = 1:numel(classes)
%     label = classes(j);
% 
%     
%        idx = find(YDiff == label);
%    if isempty(idx)
%        exclude(j)=0;
%    end
%    
%     hold on
%    p= plot(idx,X(idx),choice_shape{1},'MarkerSize',choice_size,'color',choice_color{j});
%    if j>1
%    set(p, 'markerfacecolor', get(p, 'color')); % Use same color to fill in markers
%    end
% end
% xlabel("Time")
% ylabel("Log-Returns")
% ax = gca;
% ax.FontSize = sz_font;
% 
% pos_temp =find(exclude==1);
% legend(classes(pos_temp),'Location','northwest')
% % I use classes(pos_temp) to avoid the problem of "Warning: Ignoring extra legend entries" which shifts the legend and so messes up the graph.
% 
%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      %%%% For the CLASSIFY graph %%%%
%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
%  rng(47) %Freeze the random generator
% 
%  % ATTENTION: I need to set the matrices for gamma & the Markov Chain before (I use the ones in the training data section) 
% [P,True_martingale_times,~]=Diffusion_sim(P0,NT_plot,NS,dt_plot,gamma_normal,gamma_crisis,regime,Transition_matrix_sim);
% r_ln = log(P(2:end,:) ./ P(1:end-1,:));
% 
% figure
% plot(1:length(r_ln),r_ln,choice_shape{1},'Markersize',choice_size,'color',choice_color{2})
% xlabel("Time")
% ylabel("Log-Returns")
% ax = gca;
% ax.FontSize = sz_font;

%% Plot returns for the section with bubbles

% 
% frequency = 2; %Frequency of the data (in minutes)
% Years=1;
% T_plot = Years; % In years
% NT_plot = Years*248*6.5*60/frequency;% Number of time steps
% dt_plot = T_plot/NT_plot; %dt = time interval of the data (in minutes)
% 
% 
% 
% NS=1;
% regime='switch';
% P0 = 100;
% 
% 
% sz_font = 14;
% choice_size = 3.5;
% choice_shape =  {'o'};
%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      %%%% For the LABEL graph %%%%
%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  rng(15) %Freeze the random generator
% 
% % ATTENTION: I need to set the matrices for gamma & the Markov Chain before (I use the ones in the training data section) 
% [P,True_martingale_times,~]=Diffusion_sim(P0,NT_plot,NS,dt_plot,gamma_normal,gamma_crisis,regime,Transition_matrix_sim);
% r_ln = log(P(2:end,:) ./ P(1:end-1,:));
% 
% 
% diff_ind_TM = True_martingale_times(2:end) ;
% YDiff=categorical(diff_ind_TM',[0 1],{...
%     'Bubble' ...
%     'No Bubble' ...
%       });
% 
% 
% X = r_ln;
% classes = categories(YDiff);
% 
% 
% exclude = ones(1,length(classes));
% choice_color = {'r','#0072BD'};
% 
% 
% figure
% for j = 1:numel(classes)
%     label = classes(j);
% 
%     
%        idx = find(YDiff == label);
%    if isempty(idx)
%        exclude(j)=0;
%    end
%    
%     hold on
%    p= plot(idx,X(idx),choice_shape{1},'MarkerSize',choice_size,'color',choice_color{j});
%    if j>1
%    set(p, 'markerfacecolor', get(p, 'color')); % Use same color to fill in markers
%    end
% end
% xlabel("Time")
% ylabel("Log-Returns")
% ax = gca;
% ax.FontSize = sz_font;
% 
% pos_temp =find(exclude==1);
% legend(classes(pos_temp),'Location','northwest')
% % I use classes(pos_temp) to avoid the problem of "Warning: Ignoring extra legend entries" which shifts the legend and so messes up the graph.



%% Application on REAL DATA

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Using the real data %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load net_2LSTM_50units_perlayer ; load Network_jumps ; load Stocks_paths


% This loads a table called 'S' that contains all the stock paths
% that all have the same trading dates and are clean of NaN and 0.




% Trading is from 9.30 until 16.00.
% So we have 193 observations per day (at 2 min frequency) (because we don't take the 1st and last ones).

frequency = 2; %Frequency of the data (in minutes)
Years = 3; % #of years
T = Years; % In years
NT = Years*248*6.5*60/frequency;% Number of time steps
dt = T/NT; %dt = time interval of the data (in minutes)


%ticker = "AA";


% Here I use ALL the time series data. This is the FULL list of tickers.
ticker = ["AA", "AIG", "AXP", "BA", "C", "CAT", "DD", "DIA", "DIS", "GE", ...
     "GM", "HD", "HON", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", ...
     "MMM", "MO", "MRK", "MSFT", "PFE", "PG", "PWI", "T", "UTX", "VZ", ...
     "WMT", "XOM"];

% If I have 1 ticker then I display the figure and don't save it.
% If I have >1 ticker I don't display the figures and save them in a folder.




% Parameters for the table of "ind_TM_xx"
sz = [length(S.date)-1 length(ticker)+1]; %+1 for the column with the dates
varTypes = cell(1,length(ticker)+1);
varTypes(1)={'cell'}; %For the dates
varTypes(2:end)={'double'}; %For the binary indicator
ind_TM_NN_Table = table('Size',sz,'VariableTypes',varTypes,'VariableNames',["date" ticker]);
ind_TM_NN_Table.date = S.date(2:end);


for i=1:length(ticker)

ticker(i)
    
eval( sprintf('P_real = S.%s;',ticker(i)) ); %Load the individual price path


[r_ln_real_no_jumps , ~ ,P_recovered] = remove_jumps(P_real,dt,net_jumps,fc_jumps);

P0=P_recovered(1);
XTest = { [(r_ln_real_no_jumps(:,1).*fc)'; ((P_recovered(2:end,1)./P0)*100)']};   
YPred = classify(net,XTest{1});

% for the regimes predicted by the network
pos_TM_YPred = find(YPred=='TM')'; 
ind_TM_NN = zeros(length(P_real)-1,1);
ind_TM_NN(pos_TM_YPred)=1;


eval( sprintf('ind_TM_NN_Table.%s = ind_TM_NN;',ticker(i)) );

r_ln_real_w_jumps = log(P_real(2:end,:) ./ P_real(1:end-1,:));%Vector of log-returns for all the paths
XTest_forplot = { [(r_ln_real_w_jumps(:,1).*fc)'; ((P_real(2:end,1)./P0)*100)']};
if length(ticker)==1
    % Visualize the results of the NEURAL NETWORK (NN) on REAL DATA for one single stock.
    method='Neural Network'; % Just for the title of the graph
    plot_real_data(method,ind_TM_NN,XTest_forplot,P0,ticker,'display');
    

else
    % Results of the NEURAL NETWORK (NN) on ALL THE STOCKS.
    plot_real_data('Neural Network',ind_TM_NN,XTest_forplot,P0,ticker(i),'no display');
    saveas(gcf, [pwd sprintf('/Figures/Price_%s_SLM.png',ticker(i))]) %Save the figure in a subfolder. "pwd" is the current folder.
end


end
%% Trading Strategy given the classification (of the NN)

% Set the initial capital
initial_capital=0; % =0 since I want to construct an arbitrage strategy (which requires 0 initial investment by definition.
% For the moment I assume 0 interest rate but I should change that. So that
% if I carry extra cash I earn something and if I borrow I should pay something.

position_size_in_dollars=100;

graph='no';


% load ind_TM_NN_Table_net_2LSTM_50units_perlayer


                        %%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%% SHORT LEG %%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%%

% Parameters for the table for the short of individual stocks
sz = [length(ind_TM_NN_Table.date) length(ticker)+1]; %+1 for the column with the dates
varTypes = cell(1,length(ticker)+1);
varTypes(1)={'cell'}; %For the dates
varTypes(2:end)={'double'}; %For the portfolio values
prtfl_Table = table('Size',sz,'VariableTypes',varTypes,'VariableNames',["date" ticker]);
prtfl_Table.date = ind_TM_NN_Table.date;

% TODO: in case we start we non-zero initial capital, remove DIA from the
% loop of tickers (remove DIA from the ticker list)
for i=1:length(ticker)

%filter=double(ind_TM_NN_stock==0 & ind_TM_NN_Table.DIA==1);
filter=double(ind_TM_NN_Table.(ticker(i))==0 & ind_TM_NN_Table.DIA==1);
% The goal of 'filter' is to trade ONLY WHEN the stock is in a bubble & the
% index is not.


[prtfl_performance_stock] = Trading_strategy('short SLM',ind_TM_NN_Table.(ticker(i)),S.(ticker(i))(2:end),initial_capital,position_size_in_dollars,graph,filter);

 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%
                        %%%%%% LONG LEG %%%%%%
                        %%%%%%%%%%%%%%%%%%%%%%
% DELETE BELOW
% nbr_stocks_shorted = (length(ticker) - sum(ind_TM_NN_Table{:,ticker},2) ).*ind_TM_NN_Table.DIA;
% 
% filter = double(  nbr_stocks_shorted >0);
% % This makes sure that we go long in the index ('DIA') ONLY when there is
% % at least one individual stock in a bubble state AND that stock is not the index
% % itself.
% DELETE ABOVE                        
                        
[prtfl_performance_DIA] = Trading_strategy('long TM',ind_TM_NN_Table.DIA,S.DIA(2:end),initial_capital,position_size_in_dollars,graph,filter);



prtfl_Table.(ticker(i)) = prtfl_performance_stock + prtfl_performance_DIA;

%figure
%plot(prtfl_performance_stock + prtfl_performance_DIA)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%%%%% TOTAL PERFORMANCE %%%%%%
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Here I aggregate the short & long legs in order to compute the total
% performance of the strategy.

prtfl_final = sum( prtfl_Table{:,ticker} ,2);







startDate = datenum(prtfl_Table.date{1});
endDate = datenum(prtfl_Table.date{end});
xaxis = (linspace(startDate,endDate,length(prtfl_Table.date)))'; 
figure
plot(xaxis,prtfl_final)
ylabel("$X_t$","Interpreter","latex")
ax = gca;
ax.FontSize = 15;
datetick(gca,'x','yyyy')

%% Parametric Estimator on REAL DATA

oneday=195;

window = oneday*30; % using xx days

[ind_TM_PE] = TM_PE(P_real,window,oneday,dt);

% Right now the problem is that the length of P_real must be a multiple of
% oneday, otherwise there are some NaN that appear.

% Visualize the results of the PARAMETRIC ESTIMATOR (PE)
% This plot shows the results on REAL DATA.

method='Parametric Estimator';
plot_real_data(method,ind_TM_PE,XTest,P0,ticker,'display');


%% Remove the jumps in the real data
% This section is useful to VISUALLY inspect if the SIMULATED returns
% resemble the REAL returns

% load Network_jumps


NS=1;
regime_test='switch';
P0=100;



[P_test,~,Path_mc_test]=Diffusion_sim(P0,NT,NS,dt,gamma_normal,gamma_crisis,regime_test,Transition_matrix_sim);

[raw_r_sim,st_r_sim]=remove_jumps(P_test(1:cutoff),dt,net_jumps,fc_jumps) ;

[raw_r_real,st_r_real]=remove_jumps(P_real,dt,net_jumps,fc_jumps) ;


figure
plot(raw_r_real,'o','Markersize',2)
title('raw returns from real data')
ylim([-11 11]*10^-2.5)


figure
plot(raw_r_sim,'o','Markersize',2)
title('raw returns from simulated data')
ylim([-11 11]*10^-2.5)



% 
figure
plot(st_r_real,'o','Markersize',2)
title('standardized returns from real data')
ylim([-15 15])

figure
plot(st_r_sim,'o','Markersize',2)
title('standardized returns from simulated data')
ylim([-15 15])


figure
plot(Path_mc_test(1:cutoff))
ylim([0.9 6.1])



% figure
% hist(st_r_real,100)
% xlim([-10 10])
% figure
% hist(st_r_sim,100)
% xlim([-10 10])



