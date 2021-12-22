function [prtfl_total] = Trading_strategy(strategy,ind_TM_xx_temp,P_temp,initial_capital,position_size_in_dollars,graph,filter_temp)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Trading Strategy %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% P: price of the asset.

% ind_TM_xx: time vector indicating when the asset is a True Martingale and when it's not.
% ind_TM_xx =1 when the price is a True Martingale (TM).




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Toy example %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This example is just to see how the strategy is working
 
% ind_TM_xx_temp=ones(100,1);
% rng(1) %Freeze the random generator (to repeat the same simulation)
% P_temp=nan(100,1);
% 
% P_temp(1)=10;
% for i=2:length(P_temp)
% 
%   P_temp(i)=P_temp(i-1)+randn;
% end
% 
% ind_TM_xx_temp(45:55)=0; % I arbitrarily decide when the price is a TM & a SLM
% ind_TM_xx_temp(63:69)=0; % just for the sake of the toy example.
% 
% figure
% plot(ind_TM_xx_temp)
% legend('ind TM xx')
% ylim([-0.1 1.1])
% 
% % Set the initial capital
% initial_capital=100;
% position_size_in_dollars=20;
% 
% graph='yes';
% 
% filter_temp=ones(length(P_temp),1); %Here the Filter does nothing
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% SIGNALS %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%


burnout = 0; %Skip the first few observations.
ind_TM_xx = ind_TM_xx_temp(1+burnout:end);
P = P_temp(1+burnout:end);
filter=filter_temp(1+burnout:end);


% Basically the signal to short is given by the True Martingale times (i.e. by ind_TM_xx)
% detected in the data. So there is not much work here.


if strcmp(strategy,'short SLM')

% The idea behind this strategy is to short the stock if it's in a bubble state.
% This means short the stock when it is a strict local martingale.
    
signal = (ind_TM_xx-1).*filter;
% signal = -1 the WHOLE period when I'm short.
% signal = 0 the WHOLE period when I'm neutral.
% signal = +1 the WHOLE period when I'm long.


direction_strategy = -1;


elseif strcmp(strategy,'long TM')

signal = ind_TM_xx.*filter;    

direction_strategy = 1;
else
    error 
end

signal(end)=0; % This will force to close any open position at the end of the time.

open_position_temp=diff(signal); %Here I loose one observation because I take a return
open_position=[signal(1) open_position_temp']'; %Shift the signals to go back to the price index
%open_position = -1 at the moment you start to go short
%              = +1 at the moment you start to go long (or when you close the short)


signals_dataframe = table(ind_TM_xx,signal,open_position);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% BACKTESTING %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

position_size_in_shares = position_size_in_dollars./P;


open_time = find(open_position==direction_strategy); %time when I START a position (regardless or whether it's a long or short).

close_time = find(open_position==(-1)*direction_strategy); %time when I CLOSE a position (regardless or whether it's a long or short).


open_qty = position_size_in_shares(open_time); % Number of shares I STARTED a trade with


position_size_in_shares = position_size_in_shares.*open_position;

position_size_in_shares(close_time)=open_qty;


for t=1:length(open_time)

  position_size_in_shares(open_time(t):close_time(t)) = open_qty(t);
    
end

what_to_trade = open_position.*position_size_in_shares;
% 'what_to_trade' gives the number of shares to buy/sell at the specific
% moment when you need to trade.

nbr_shares=position_size_in_shares.*signals_dataframe.signal;
% 'nbr_shares' gives the number of shares I hold AT ALL TIMES.

position_value=nbr_shares.*P;
% 'position_value' gives the $ value of the position I have in the stock.

cash=initial_capital - cumsum(what_to_trade.*P);


prtfl_total=position_value + cash;


portfolio_dataframe = table(what_to_trade,nbr_shares,position_value,cash,prtfl_total);
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Plot %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(graph,'yes')

ind_xaxis_short=find(open_position==-1);
ind_xaxis_buy=find(open_position==1);

figure
plot(P,'p','MarkerIndices',ind_xaxis_short,...
'MarkerFaceColor','red',...
'MarkerSize',10)
    hold on
plot(P,'d','MarkerIndices',ind_xaxis_buy,...
    'MarkerFaceColor','green',...
    'MarkerSize',10)
hold on
plot(P,'Color','#0072BD')
legend('Start Shorting','Start Longing','Stock Price')



figure
plot(prtfl_total,'p','MarkerIndices',ind_xaxis_short,...
'MarkerFaceColor','red',...
'MarkerSize',10)
    hold on
plot(prtfl_total,'d','MarkerIndices',ind_xaxis_buy,...
    'MarkerFaceColor','green',...
    'MarkerSize',10)
hold on
plot(prtfl_total,'Color','#0072BD')
legend('Start Shorting','Start Longing','Value of my portfolio')
ylim([min(prtfl_total)-2 max(prtfl_total)+2])

end

%%
end