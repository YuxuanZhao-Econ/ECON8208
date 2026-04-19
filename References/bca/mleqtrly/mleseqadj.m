function [L,Lt,innov,X0,Cbar,A,K,D,gz] = mleseq(param,adja);
%MLE     Log-likelihood function for the standard growth model with
%        fluctuations in four ``wedges'': z, taul, taux, g.
%        See appendix for details.
%

%        Ellen R. McGrattan, 5-14-04
%

global ZVAR    % = log([Output, Investment, Hours, Govt Spending ])
               %        (all in per-capita terms)
%---------------------------------------------------------------------
% 1. Default parameters for 
%    gn, gz, beta, delta, psi, sigma, theta, Sbar, P, Q, D, R
%
gn      = (1.015)^(1/4)-1;
gz      = (1.016)^(1/4)-1;
beta    = .9722^(1/4);
delta   = 1-(1-.0464)^(1/4);
psi     = 2.24;
sigma   = 1.000001;
theta   = .35;
adjb    = (1+gn)*(1+gz)-1+delta;
P       = .995*eye(4);
Sbar    = [log(1);.05;.0;log(.07)];
P0      = (eye(4)-P)*Sbar;
Q       = zeros(4);
Q(1,1)  = .01;
D       = zeros(4);
R       = zeros(4);
par     = [gn;gz;beta;delta;psi;sigma;theta;adja];

Sbar(1) = param(1);
Sbar(2) = param(2);
Sbar(3) = param(3);
Sbar(4) = param(4);
P(1,1)  = param(5);
P(2,1)  = param(6);
P(3,1)  = param(7);
P(4,1)  = param(8);
P(1,2)  = param(9);
P(2,2)  = param(10);
P(3,2)  = param(11);
P(4,2)  = param(12);
P(1,3)  = param(13);
P(2,3)  = param(14);
P(3,3)  = param(15);
P(4,3)  = param(16);
P(1,4)  = param(17);
P(2,4)  = param(18);
P(3,4)  = param(19);
P(4,4)  = param(20);
Q(1,1)  = param(21);
Q(2,1)  = param(22);
Q(3,1)  = param(23);
Q(4,1)  = param(24);
Q(2,2)  = param(25);
Q(3,2)  = param(26);
Q(4,2)  = param(27);
Q(3,3)  = param(28);
Q(4,3)  = param(29);
Q(4,4)  = param(30);
P0      = (eye(4)-P)*Sbar;
penalty = 500000*max(max(abs(eig(P)))-.995,0).^2;
                                                                                      
%---------------------------------------------------------------------
% 5.  Compute equilibrium
%---------------------------------------------------------------------

%
% 5a. Steady state:
%
tem        = (eye(4)-P)\P0;
zs         = exp(tem(1));
tauls      = tem(2);
tauxs      = tem(3);
gs         = exp(tem(4));
beth       = beta*(1+gz)^(-sigma);
kls        = ((1+tauxs)*(1-beth*(1-delta))/(beth*theta))^(1/(theta-1))*zs;
A          = (zs/kls)^(1-theta)-(1+gz)*(1+gn)+1-delta;
B          = (1-tauls)*(1-theta)*kls^theta*zs^(1-theta)/psi;
ks         = (B+gs)/(A+B/kls);
cs         = A*ks-gs;
ls         = ks/kls;
ys         = ks^theta*(zs*ls)^(1-theta);
xs         = ys-cs-gs;
X0         = [log(ks);log(zs);tauls;tauxs;log(gs);1];
Y0         = [log(ys);log(xs);log(ls);log(gs)];

%
% 5b. Call subroutine with residuals:
%

Z          = [log(ks);log(ks);log(ks);log(zs);log(zs);tauls;tauls;
              tauxs;tauxs;log(gs);log(gs)];
del        = max(abs(Z)*1e-5,1e-8);
for i=1:11;
  Zp       = Z;
  Zm       = Z;
  Zp(i)    = Z(i)+del(i);
  Zm(i)    = Z(i)-del(i);
  dR(i,1)  = (res_adjust(Zp,par)-res_adjust(Zm,par))/(2*del(i));
end;

%
% 5c. Solution:  log k[t+1] = gamma0 + gammak* log k[t] + gamma* S[t]
%

a0         = dR(1);
a1         = dR(2);
a2         = dR(3);
b0         = dR(4:2:11)';
b1         = dR(5:2:11)';
tem        = roots([a0,a1,a2]);
gammak     = tem(find(abs(tem)<1));
gamma      = -((a0*gammak+a1)*eye(4)+a0*P')\(b0*P+b1)';
gamma0     = (1-gammak)*log(ks)-gamma'*[log(zs);tauls;tauxs;log(gs)];
Gamma      = [gammak;gamma;gamma0];

%
% 5d. State-space system:   X[t+1] = A X[t] + B eps[t+1]
%                           Y[t]   = C X[t] + ome[t]
%                           ome[t] = D ome[t-1] + eta[t],  E eta eta' ~ N(0,R)
%
philh      =-(psi*ys*(1-theta)+(1-theta)*(1-tauls)*ys*(1-ls)/ls*theta+ ...
             (1-theta)*(1-tauls)*ys);
philk      = (psi*ys*theta+psi*(1-delta)*ks- ...
             (1-theta)*(1-tauls)*ys*(1-ls)/ls *theta)/philh;
philz      = (psi*ys*(1-theta)-(1-theta)^2*(1-tauls)*ys*(1-ls)/ls)/philh;
phill      = ((1-theta)*(1-tauls)*ys*(1-ls)/ls *(1/(1-tauls)))/philh;
philg      = (-psi*gs)/philh;
philkp     = (-psi*(1+gz)*(1+gn)*ks)/philh;
phiyk      = theta+(1-theta)*philk;
phiyz      = (1-theta)*(1+philz);
phiyl      = (1-theta)*phill;
phiyg      = (1-theta)*philg;
phiykp     = (1-theta)*philkp;
phixk      = -ks/xs*(1-delta);
phixkp     = ks/xs*(1+gz)*(1+gn);


A     = [ gammak,                         gamma', gamma0;
         [0;0;0;0],            P,                     P0;
               0,      0,      0,      0,      0,      1];
B     = [ 0,0,0,0;
            Q;
          0,0,0,0];
C     = [ [phiyk,phiyz,phiyl,    0,phiyg]+phiykp*Gamma(1:5)';
          [phixk,    0,    0,    0,    0]+phixkp*Gamma(1:5)';
          [philk,philz,phill,    0,philg]+philkp*Gamma(1:5)';
          0,0,0,0,1];
phi0  = Y0-C*X0(1:5);
C     = [C,phi0];
%---------------------------------------------------------------------
% 6. Specify observables (Z). 
%
T          = length(ZVAR);
Y          = log(ZVAR)-log([(1+gz).^[0:T-1]',(1+gz).^[0:T-1]', ...
                             ones(T,1),(1+gz).^[0:T-1]']);
Ybar       = Y(2:T,:)-Y(1:T-1,:)*D';
T          = T-1;
Cbar       = C*A-D*C;
Rbar       = R+C*B*B'*C';
[K,Sigma]  = kfilter(A,Cbar,B*B',Rbar,B*B'*C');
Omega      = Rbar+Cbar*Sigma*Cbar';
Omegai     = inv(Omega);
Xt(1,:)    = X0';
innov(1,:) = Ybar(1,:)-X0'*Cbar';
Lt         = zeros(T,1);
Lt(1)      = 0.5*(log(det(Omega))+innov(1,:)*Omegai*innov(1,:)');
for i=2:T;
  Xt(i,:)    = Xt(i-1,:)*A'+innov(i-1,:)*K';
  innov(i,:) = Ybar(i,:)-Xt(i,:)*Cbar';
  Lt(i)      = 0.5*(log(det(Omega))+innov(i,:)*Omegai*innov(i,:)');
end;
MY         = exp(Xt*Cbar');
DY         = exp(Ybar);
MX         = [exp(Xt(:,1:2)),Xt(:,3:4),exp(Xt(:,5))];
sum1       = innov(1:T,:)'*innov(1:T,:)/T;

L          = 0.5*(T*(log(det(Omega))+trace(Omegai*sum1))+penalty);
Lt         = Lt+0.5*penalty/T;
