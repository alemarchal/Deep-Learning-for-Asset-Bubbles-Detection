function [P,True_martingale_times, Path_mc] = Diffusion_sim(P0,NT,NS,dt,gamma_normal,gamma_crisis,regime,Transition_matrix_sim)


% P0 = initial spot price



% sigma = volatillity
% T = maturity

% NT = number of time steps
% NS = number of stock price paths



                               
                
True_martingale_times = zeros(NT,NS);
Path_mc = zeros(NT,NS);
gamma_path_mc_crisis = zeros(NT,NS);

% Initialize the stock price paths
P = zeros(NT,NS);
P(1,:) = P0;

Z = normrnd(0,1,NT,NS);
W = sqrt(dt)*Z; %Wiener Process
mc=dtmc(Transition_matrix_sim, 'StateNames', ["SLM1", "SLM2", "SLM3", "TM1", "TM2", "TM3"]);



Transition_matrix_gamma=[0.999998 0.000002   0         0;...
                            0       0.9999    0        0.0001;...
                            0.0001   0        0.9999     0;...
                            0       0        0.000035  0.999965];
mc_gamma=dtmc(Transition_matrix_gamma, 'StateNames', ["normal", "TrfNormal", "TrfCrisis" "crisis"]);

%% Simulation of the PRICE process
drift = 0;

diffusion = @(gamma, x) gamma(1) * x^gamma(2);

for s=1:NS  % For each path
    
    
                        gamma_path_mc_crisis(:,s) = simulate(mc_gamma,NT-1)-1;
%                         plot(gamma_path_mc_crisis)
%                         ylim([-0.1 4.1])
       
       
       
                        if strcmp(regime,'switch')
                        %%%This is different for each path%%%


                       
                        Path_mc(:,s) = simulate(mc,NT-1);
                        True_martingale_times(:,s) = double(Path_mc(:,s)>3);
                        





                        elseif strcmp(regime,'tm_all_times')                
                             True_martingale_times(:,s)=1;   
                             Path_mc(:,s) = length(gamma_normal(:,1));


                        elseif strcmp(regime,'slm_all_times')
                             True_martingale_times(:,s)=0; 
                             Path_mc(:,s) = 1;

                        else 
                            error
                        end
                        
                        
    for t=2:NT % For each time step
        
              gamma = gamma_normal* (1-(gamma_path_mc_crisis(t,s)/3)) + gamma_crisis*(gamma_path_mc_crisis(t,s)/3);
        
        
              P(t,s) = P(t-1,s) +  drift*dt + diffusion(gamma(Path_mc(t,s),:) , P(t-1,s)) *W(t-1,s);     
        

    end 
end









end