% RUNMLE1    Compute MLE estimates for the period 1901--1940 for
%            the benchmark model.  Set ibind equal to 1 if 
%            bootstrapped standard errors are required (that is,
%            if constraints bind on matrix P.
%

%            Ellen McGrattan, 2-16-02
%            Revised, ERM, 3-17-05

%------------------------------------------------------------------------------
load uszvar1.dat
global ZVAR
t    = uszvar1(1:40,1);
ZVAR = uszvar1(1:40,2:5);
x0=[ ...
   0.54082135494385
  -0.18961520211220
   0.28589191522520
  -2.79358307142896
   0.73194722119709
  -0.14950077057243
  -0.01135154662368
   0.05206442487424
   1.03867122470732
  -0.01965637369407
  -0.31698595387468
   0.39035390644605
   0.07313270008334
   0.74982873714821
  -0.05746385259960
   0.00561454650003
  -0.00298795420604
   0.05551125567149
  -0.00025361856807
  -0.03694910344636
   0.22143490701858];

[x,f,g,code,status]  = uncmin(x0,'mle1',0);
L           = f;
ibind       = 0;
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
      [f1,f2] = mlese1(param+del(:,i),0);
      [m1,m2] = mlese1(param-del(:,i),0);
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
  [L,Lt,ut,X0,Cbar,A,K,D,gz] = mlese1(param,0);
  B     = 500;    % number of bootstrap replications
  T     = length(ut);
  Ybar  = 0*ut;
  nx    = length(X0);
  Xt    = zeros(T+1,nx);
  Y     = zeros(T+1,4);
  Y(1,:)= log(ZVAR(1,:));
  Data  = zeros(4*(T+1),B);
  Theta = zeros(B,length(ind));
  thet0 = param(ind);
  rand('state',100462)
  for i=1:B;
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
    theta     = uncmin(thet0,'mle1',0);

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

%save runmle1

