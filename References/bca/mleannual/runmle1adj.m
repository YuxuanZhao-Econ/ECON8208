
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
   0.54082135494385   0.53688220711471   0.53632298081285
  -0.18961520211220  -0.19633927001699  -0.20039531347703
   0.28589191522520   0.29058999479661   0.29210706768606
  -2.79358307142896  -2.79623988122216  -2.80780348317646
   0.73194722119709   0.56390392825651   0.43182517280778
  -0.14950077057243   0.04695540205288   0.17815955068537
  -0.01135154662368  -0.37800322911024  -0.65916167358026
   0.05206442487424   0.08977302569669   0.12063917741983
   1.03867122470732   0.99533700152441   0.96304256251874
  -0.01965637369407   0.05448227333478   0.11170155154008
  -0.31698595387468  -0.19009624055368  -0.08657508959716
   0.39035390644605   0.21919090712206   0.09382053325445
   0.07313270008334   0.33666352061755   0.55959882698031
   0.74982873714821   0.76639431904874   0.78311669515450
  -0.05746385259960  -0.05675342255460   0.05614910291263
   0.00561454650003   0.00425205357843  -0.00326537888522
  -0.00298795420604   0.04319778764689  -0.17603674722045
   0.05551125567149   0.05542040154641   0.05526762622885
  -0.00025361856807   0.02058061530191   0.08107289332405
  -0.03694910344636  -0.07681637702580  -0.18875619010229
   0.22143490701858   0.22112658504395   0.22088105811702];
%  adja = 0                3.22               12.88

adja = 12.88;
x0   = x0(:,3);

[x,f,g,code,status]  = uncmin(x0,'mle1adj',adja);
L           = f;
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
      [f1,f2] = mlese1adj(param+del(:,i),adja);
      [m1,m2] = mlese1adj(param-del(:,i),adja);
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
  [L,Lt,ut,X0,Cbar,A,K,D,gz] = mlese1adj(param,adja);
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
    theta     = uncmin(thet0,'mle1adj',adja);

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

%save runmle1adj

