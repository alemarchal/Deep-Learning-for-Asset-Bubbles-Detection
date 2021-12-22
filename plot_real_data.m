function  [p] = plot_real_data(method,ind_TM_xx,XTest,P0,ticker,visibility)

% ind_TM_xx: index of True Martingale times (=1 when true martingale) of
% the method xx (either NN or PE).


YDiff=categorical(ind_TM_xx',[0 1],{...
    'SLM' ...
    'TM' ...
      });

  

% startDate = datenum('03-Jan-2006 9:35:00');
% endDate = datenum('31-Dec-2008 16:00:00');
% xaxis = linspace(startDate,endDate,length(P_real));  
  

%X = XTest{1}(1,:)./fc; % plotting: 1-returns
X = (XTest{1}(2,:).*P0)./100; % plotting: 2-prices 


classes = categories(YDiff);

exclude = ones(1,length(classes));
choice_color = {'r','#0072BD'};
choice_shape =  {'o','d'};
choice_size = [2 2];

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

    if      strcmp(visibility,'display')
        set(gcf,'Visible','on');    
    elseif  strcmp(visibility,'no display')    
        set(gcf,'Visible','off'); %In order NOT TO display the figure
    else
        error
    end
    
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]); % Set the figure SIZE to full screen
xlabel("Time")
ylabel("$P_t$","Interpreter","latex")
title(sprintf('%s Classification of %s',method,ticker))
ax = gca;
ax.FontSize = 20;

%datetick(gca,'x','yyyy')

pos_temp =find(exclude==1);
legend(classes(pos_temp),'Location','northwest')
% I use classes(pos_temp) to avoid the problem of "Warning: Ignoring extra legend entries" which shifts the legend and so messes up the graph.




end