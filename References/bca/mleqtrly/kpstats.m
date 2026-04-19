%KPSTATS   This code records correlations and standard
%          deviations of the wedges.  Run after running
%          bca.m

%          Ellen McGrattan, 2-7-05


%
% Wedges:
%

Ewedge  = (Zt/Zt(Y0)).^(1-theta);
Lwedge  = (1-Tault)/(1-Tault(Y0));
Xwedge  = (1+tauxt(Y0))*ones(T,1)./(1+tauxt);
Gwedge  = exp(lgt-lgt(Y0));
W       = [Ewedge,Lwedge,Xwedge,Gwedge];
labels  = ['Efficiency ';
           'Labor      ';
           'Investment ';
           'Govt. Cons.'];

%
% Output Components:
%

OC      = [fig2mz(:,2),fig2ml(:,2),fig3mx(:,2),fig3mg(:,2)]/100;

%
% Plot results:
%
clc
disp('         PROPERTIES OF THE WEDGES, 1959:1-2004:3')
tab1    = stats(W,exp(lyt-lyt(Y0)),0,labels);
disp(' ')
disp(' ')
disp('   PROPERTIES OF THE OUTPUT COMPONENTS, 1959:1-2004:3')
tab2    = stats(OC,exp(lyt-lyt(Y0)),0,labels);

