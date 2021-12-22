function [Bipower_StD]=BV(K_window,V_filtre,ti) 
 % Compute the Bipower variation

    t=ti-K_window+2;
    ending = ti-1;

    %Computation Bipower quadratic variation
    A = abs(log(V_filtre(t:ending)./V_filtre(t-1:ending-1)));
    B = abs(log(V_filtre(t-1:ending-1)./V_filtre(t-2:ending-2)));
    bivar = A'*B;
    Bipower_StD = sqrt(1/(K_window-2)*sum(bivar));
    
 
 end