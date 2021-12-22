function [M_prices,M_dates] = extract_trading_days(ticker)

% For instance: ticker = "MSFT";


%%%%%%%% To load the ORIGINAL data with the Trading DATES %%%%%%%%
stock_path=sprintf('../TAQmat_65_kofd935/%s_Y2006_2008M1_12_2min.mat',ticker);
structured_file=load (stock_path);
data=structured_file.('CLEANDATA_2min');

M_dates_matlab_format = data(:,:,1); %Matrix contening the TRADING DATES
M_prices = data(:,:,2); %Matrix with the PRICES

M_dates_temp=datetime(M_dates_matlab_format,'ConvertFrom','datenum');

% The code below replaces the date with the NEXT/PREVIOUS observation in case of NaT.
            M_dates_clean=M_dates_temp;
            
            for r=1:length(M_dates_temp) %I do the replacement row-by-row
            
            index_NaT = find(isnat(M_dates_clean(r,:))==1);
  
                if isempty(index_NaT)
                % Do nothing
                else
            
                      if index_NaT(1) < floor(length(M_dates_clean(r,:))/2)
                      %If the NaT are at the beginning of the row then I replace with the NEXT observation    
                                while ~isempty(index_NaT) %To tackle CONSECUTIVE NaT                  
                                index_NaT = find(isnat(M_dates_clean(r,:))==1);
                                M_dates_clean(r,index_NaT)= M_dates_clean(r,index_NaT+1);
                                end
                      else
                      %If the NaT are at the end of the row then I replace with the PREVIOUS observation    
                                while ~isempty(index_NaT)                   
                                index_NaT = find(isnat(M_dates_clean(r,:))==1);
                                M_dates_clean(r,index_NaT)= M_dates_clean(r,index_NaT-1);
                                end
% ATTENTION: this code will have a bug IF I have a NaT simultaneously at
% the first and last position in a row. Hopefully it doesn't happen with
% this data set. If it happens then I have to re-writte the code.
                      end
                  
                end
            end    

M_dates_clean.Format = 'dd-MMM-yyyy'; %To get rid of the "hour:minute:second" information (only keep the specific DAY)
% That's because the stocks don't trade on the exact same second. So we
% can't compare with the time included.

M_dates = cellstr(M_dates_clean); %Trading dates for the specific stock corresponding to the ticker.


end