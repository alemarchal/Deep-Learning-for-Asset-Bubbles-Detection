% PREPARE THE DATA FOR THE TRADING STRATEGY

% This file is just used to clean real data (get rid of the 0 and NaN in prices)
% and to make sure all of the time series have the SAME LENGTH (later for
% the trading strategy).
% Then it saves all the prices in a table.


% Here I use ALL the time series data. This is the FULL list of tickers.
 ticker = ["AA", "AIG", "AXP", "BA", "C", "CAT", "DD", "DIA", "DIS", "GE", ...
      "GM", "HD", "HON", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", ...
      "MMM", "MO", "MRK", "MSFT", "PFE", "PG", "PWI", "T", "UTX", "VZ", ...
      "WMT", "XOM"];
  
%ticker = "MSFT";
%% Determining which time-series is the SHORTEST (not necessary)

lengths_stocks=cell(length(ticker),2);
for i=1:length(ticker) 

    
%%%%%%%% To load only the prices (NO DATES) %%%%%%%%   
stock_path=sprintf('../HF data/%s_2min.mat',ticker(i));
stock_name=sprintf('%s_2min',ticker(i));
structured_file=load (stock_path);
P_real=structured_file.(stock_name);    
    
    
lengths_stocks{i,1} = sprintf('The length of %s is',ticker(i));
lengths_stocks{i,2}=length(P_real);
end

% From that we learn that:
% AA (with others) is the longest time-series.
%
% MSFT (with others) is the shortest time-series (it is 2 days shorter).
% That information is not very useful for the rest of the code.

%% This section creates the variable 'M_dates_ref'.

% I need to remove all the dates in the longer time-series (so that we have observations on the
% SAME DATES for all the time-series). 
% This is important for time consistency of the trading strategy).


[~,M_dates_ref] = extract_trading_days("MSFT");   
% I take the trading days of MSFT as reference
% since it is (one of) the stock with the smallest number of trading days.
% But it doesn't really matter which reference I take.

% M_dates_ref = "reference dates"

for i=1:length(ticker) 

    
[~,M_dates] = extract_trading_days(ticker(i));     
    
    
dates_member = ismember(M_dates_ref,M_dates);
% Here I check if the "reference dates" belong to each individual stocks dates.

ind_row_non_common_dates = find(dates_member(:,1)==0); 
                                               
M_dates_ref(ind_row_non_common_dates,:) = [];    
    
end

% What I do here is check if the "reference dates" has dates that are
% not present in individual stocks (I check against each individual stock).
% If it's the case, then I remove these dates from "reference dates".

% I do that so that "reference dates" will be the LARGEST SUBSET of dates
% such that these dates are present in each individual price time-series.
% --> LARGEST SUBSET of common dates.

%%
% Now I have the "reference dates" to which I'm gonna compare all my stocks.
% If the dates of my stock are not present in 'M_dates_ref', I remove them.
% So 'M_dates_ref' is like a benchmark.

% Parameters for the final table
sz = [length(reshape(M_dates_ref',[],1)) length(ticker)+1]; %+1 for the column with the dates
varTypes = cell(1,length(ticker)+1);
varTypes(1)={'cell'}; %For the dates
varTypes(2:end)={'double'}; %For the stock prices
S = table('Size',sz,'VariableTypes',varTypes,'VariableNames',["date" ticker]);
S.date = reshape(M_dates_ref',[],1);


check_mat = ones(size(M_dates_ref)); %To check later if the code works.

for i=1:length(ticker) 


[M_prices,M_dates] = extract_trading_days(ticker(i));   
    
% Here I check if each individual stocks dates are in the "reference dates".    
dates_member = ismember(M_dates,M_dates_ref);
% =1 if the trading date is in both stocks.
% =0 for the dates that are not common.

ind_row_non_common_dates = find(dates_member(:,1)==0); 
                                               
M_dates(ind_row_non_common_dates,:) = [];
% Remove the non-common trading dates.

M_prices(ind_row_non_common_dates,:) = []; 
% Remove the PRICES which are associated to the non-common trading dates.


%%%%%%% Now I store the paths in a table %%%%%%%
price_vector_form = reshape(M_prices',[],1); % Put the price into a column vector (instead of a matrix)
eval( sprintf('S.%s = price_vector_form;',ticker(i)) );


%%%%%%% Verification that the code works %%%%%%%
% Now that I have removed the non-common dates in 'M_dates',
% this 'check_mat' matrix should remain a matrix full of 1, iteration after
% iteration if the code works correctly.

% At this point, M_dates & M_dates_ref should be equal elementwise.
% That's why I check by using a stronger condition (elementwise equality)
% rather than simply ismember(.,.)
check_mat = check_mat*isequal(M_dates,M_dates_ref);
end

if all(all(check_mat))
% The function "all" verifies if ALL the elements are non-zero
% =1 is all the elements are non-zero. =0 otherwise.
disp('The code works properly.')
else
warning('The code did not work! Not all the dates match. Check at which iteration the matrix <check_mat> stops to be filled with ones only')    
end
    

%% Now that I observe the prices on the exact same dates for all the time-series,
% we can take care of removing the NaN and 0 in the price.


method='replace';

for i=1:length(ticker) 

%Load the individual price path
eval( sprintf('P_temp = S.%s;',ticker(i)) );   

    
  if      strcmp(method,'delete')
% The code below just deletes the observation if the price is 0 or a NaN.
% The problem is that if different time-series have a different number of
% 0/NaN then they will have different lengths after cleaning.

            P_temp=P_temp(P_temp~=0); %Get rid of the 0 to compute the log-returns
            P_temp(isnan(P_temp)) = [] ; % Remove the NaN
            P_clean=P_temp;

  elseif strcmp(method,'replace')
% The code below replaces the price with the previous observation in case
% of 0 or NaN.


            P_clean=P_temp;
            index_nan = find(isnan(P_clean)==1);
            index_zero = find(P_clean==0);
           

            while ~isempty(index_nan) %To tackle CONSECUTIVE NaN
                   
            index_nan = find(isnan(P_clean)==1);
            P_clean(index_nan)=P_clean(index_nan-1);
            end

            while ~isempty(index_zero) %To tackle CONSECUTIVE zeros
               
            index_zero = find(P_clean==0); 
            P_clean(index_zero)=P_clean(index_zero-1);                
            end
  %N.B: since I replace with the previous observation, I will have a bug if
  %I have a NaN/0 at the beginning of the series.
  %If this happens I need to adapt the code.
            
  elseif strcmp(method,'nothing')          
            disp('No data cleaning')
            
  else
      error
  end
  
% Now that the individual price path is clean,
% I put it back into the table S.
eval( sprintf('S.%s = P_clean;',ticker(i)) );

end


save('Stocks_paths','S')

