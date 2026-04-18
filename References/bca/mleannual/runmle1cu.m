% RUNMLE1CU  Compute MLE estimates for the period 1901--1940 for
%            the model with variable capacity utilization.  Set 
%            ibind equal to 1 if bootstrapped standard errors are 
%            required (that is, if constraints bind on matrix P.

%

%            Ellen McGrattan, 2-16-02
%            Revised, ERM, 3-17-05

%------------------------------------------------------------------------------
load uszvar1.dat
global ZVAR
t    = uszvar1(1:40,1);
ZVAR = uszvar1(1:40,2:5);
x0   = [ ...
   0.74408282199772
   0.22927873983661
   0.28169549632633
  -2.78399474695614
   0.66596429818010
  -0.17807558542198
  -0.04015350515109
   0.17134693088127
   1.07982481158731
   0.03906495190886
  -0.19215019977223
   0.28516765729246
   0.10806601764070
   0.74441745479757
  -0.03250937759605
  -0.01093968220678
  -0.00970770536392
   0.03427070626369
  -0.00089506264134
   0.03750771903069
   0.22166031705720];

[x,f,g,code,status]  = uncmin(x0,'mle1cu',0);

ibind       = 1;
param       = zeros(30,1);
ind         = [1:7,9:11,13:15,20:23,25:26,28,30]';
param(ind)  = x;
if ibind==0;
  %
  % Standard errors for the case with nonbinding constraints
  %
  del         = diag(max(abs(param)*1e-4,1e-8));
  j=0;
  for i=1:length(param);
    if ~isempty(ind==i);
      j=j+1;
      [f1,f2] = mlese1cu(param+del(:,i),0);
      [m1,m2] = mlese1cu(param-del(:,i),0);
      dL(i,1)  = (f1-m1)/(2*del(i,i));
      dLt(i,:) = (f2-m2)'/(2*del(i,i));
    end;
  end;
  dLt =dLt(ind,:);
  sum1=0;
  [n,m]=size(dLt);
  for t=1:m;
    sum1=sum1+dLt(:,t)*dLt(:,t)';
  end;
  se=diag(sqrt(inv(sum1)));
else;
  %
  % Bootstrapped standard errors for the case with binding constraints
  %
  [L,Lt,ut,X0,Cbar,A,K,D,gz] = mlese1cu(param,0);
  B     = 500;    % number of bootstrap replications
  T     = length(ut);
  Ybar  = 0*ut;
  nx    = length(X0);
  Xt    = zeros(T+1,nx);
  Y     = zeros(T+1,4);
  Y(1,:)= log(ZVAR(1,:));
  Data  = zeros(4*(T+1),B);
  Theta = zeros(B,length(ind));
  rand('state',100462)
  thet0 = param(ind);
  for i=1:B;
    disp(i)
    %
    % Draw u's uniformly from sample {ut(1),ut(2)....ut(T)}
    %
    Xt(1,:)   = X0';
    for j=1:T;
      k             = ceil(rand*T);  % draw number between 1 and T
      Ybar(j,:)     = Xt(j,:)*Cbar'+ut(k,:);
      Xt(j+1,:)     = Xt(j,:)*A'+ut(k,:)*K';
    end;
    for j=2:T+1;
      Y(j,:) = Y(j-1,:)*D'+Ybar(j-1,:);
    end;
    ZVAR      = exp(Y+log([(1+gz).^[0:T]',(1+gz).^[0:T]', ...
                               ones(T+1,1),(1+gz).^[0:T]']));
    Data(:,i) = ZVAR(:);
    theta     = uncmin(thet0,'mle1cu',0);

    if theta(15)<0;
      theta(15:17) = -theta(15:17);
    end;
    if theta(18)<0;
      theta(18:19) = -theta(18:19);
    end;
    if theta(20)<0;
      theta(20)    = -theta(20);
    end;
    Theta(i,:) = theta';
  end;
  se   = std(Theta)';
end;

%
% Print results
%
                                                                                
disp('Results from Maximum Likelihood Estimation')
disp('------------------------------------------')
disp(' ')
disp('  [Theta, Standard Errors] ')
disp(sprintf(' %10.3e %10.3e\n', [param(ind),se]'))
disp(' ')
fprintf('  L(Theta) = %g ',L)
disp(' ')

save runmle1cu

