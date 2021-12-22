function [output] = sigma_estimator(n,T,P,method,x)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%% Estimator of Florens-Zmirou %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(method,'FZ')

    % sympref('HeavisideAtOrigin',0);
    
%     h = n^(-1/4);
% 
%     
%     ind=  @(y) heaviside(h-abs(y)); % indicator function
%     
%     
%     terms_L=nan(1,length(P));
%     terms_l=nan(1,length(P)-1);
%     
%     for i=1:length(P)-1
%     
%     terms_L(i) = ind(P(i)-x);
%     
%     terms_l(i) = ind(P(i)-x) * n * (P(i+1)-P(i))^2;
%     
%     end
% 
%     terms_L(end)=ind(P(end)-x);
%     
%     
%     L = T/(2*n*h)*sum(terms_L);
%     
%     l = T/(2*n*h)*sum(terms_l);
%     
%     sigma_square = l/L;
%     
%    output = sigma_square;
    
    
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%%%% Estimator of JKP %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(method,'JKP')
% Implements the Jarrow-Kchia-Protter to estimate sigma(x)
% non-parametrically. Based on the paper "How to detect an asset bubble"
% (2011) SIAM.
    
    
    
    
% 
% h = n^(-1/4);
% 
% phi = @(y) 1/sqrt(2*pi)* exp(-0.5 * y.^2); % Gaussian Kernel --> kernel
% is not correct !!!
% 
% 
% terms_V = nan(1,length(P)-1);
% terms_L = nan(1,length(P)-1);
% 
% for i=1:length(P)-1
% 
% terms_V(i) =  phi((P(i)-x)/h) *n *(P(i+1)-P(i))^2;
%  
% 
% terms_L(i) =   phi((P(i)-x)/h);
% end
% 
% 
% % From equation (6) & (7) of the paper by JKP (2011)
% V =  1/(n*h)*sum(terms_V);
% L =  1/(n*h)*sum(terms_L);
% 
% sigma_square = V/L; % estimate of sigma^2(x)  (at the point x)

% output = sigma_square;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%% Parametric Estimator of Genon-Catalot Jacod %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

elseif strcmp(method,'parametric')

    %%
    
objective = @(param)  1/n * sum( (param(1)^2 * P(1:end-1).^(2*param(2)) - n/T * (P(2:end)-P(1:end-1)).^2).^2 ); % Using the power function family

starting_points = [1,1];


% optimset('fminsearch')
options = optimset('MaxIter',5000,'MaxFunEvals',10000);
param_star = fminsearch(objective,starting_points,options);
    

output = param_star;
     
 
else
    
    error
    
end


end