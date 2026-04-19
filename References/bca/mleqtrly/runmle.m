%RUNMLE     Compute MLE estimates for the benchmark model using the
%           sample period 1959:1--2004:3 and bootstrap standard errors 
%           if constraints bind on matrix P. The likelihood function
%           is in file mleq.m
%
                                                                                
%           Ellen McGrattan, 2-16-02
%           Revised, ERM, 3-18-05

load uszvarq.dat
global ZVAR
ZVAR  = uszvarq(:,2:5);

x0=[ ...
   0.56129229542991
  -0.22315683391720
   0.35940299493795
  -2.31053250090644
   0.97495145690175
  -0.00748563008432
  -0.00597701490688
                  0
   0.01728027189850
   1.01435681124639
  -0.00935002415423
                  0
   0.01392074289790
   0.04681660428893
   0.95238089627209
                  0
                  0
                  0
                  0
   0.96606868092918
   0.02379215892897
  -0.01064290730504
   0.01295512011754
                  0
   0.02736479689259
  -0.01874060582004
                  0
   0.02732929989630
                  0
   0.11138886148534];  % result from initpw with adja=0

X       = zeros(30,20);
[x1,f,g,code,status]   = uncmin(x0,'mleq',0);
X(:,1) = x1;
F(1,1) = f;
[x1,f,g,code,status]   = uncmin(x1,'mleq',0);
X(:,2) = x1;
F(2,1) = f;
%
% Move away to see if we get more improvement
%
x2 = x1;
for i=3:50;
  [x2,f,g,code,status]   = uncmin(x2*.99,'mleq',0);
  F(i,1) = f;
  X(:,i) = x2;
  disp(F)
  pause(10)
end;
i           = find(F==min(F));
x           = X(:,i);
L           = F(i);
ibind       = 1;
param       = zeros(30,1);
ind         = 1:30;
param(ind)  = x;

if ibind==0;
  %
  % Standard errors for the case with nonbinding constraints
  %
  del   = diag(max(abs(param)*1e-4,1e-8));
  for i=1:length(param);
    [f1,f2]  = mleseq(param+del(:,i),0);
    [m1,m2]  = mleseq(param-del(:,i),0);
    dL(i,1)  = (f1-m1)/(2*del(i,i));
    dLt(i,:) = (f2-m2)'/(2*del(i,i));
  end;
  [n,m] = size(dLt);
  sum1  = 0;
  for t=1:m;
    sum1=sum1+dLt(:,t)*dLt(:,t)';
  end;
  se = diag(sqrt(inv(sum1)));
else;
  %
  % Bootstrapped standard errors for the case with binding constraints
  %
  [L,Lt,ut,X0,Cbar,A,K,D,gz] = mleseq(param,0);
  B         = 500;   % number of bootstrap replications
  T         = length(ut);
  Ybar      = 0*ut;
  nx        = length(X0);
  Xt        = zeros(T+1,nx);
  Y         = zeros(T+1,4);
  Y(1,:)    = log(ZVAR(1,:));
  Data      = zeros(4*(T+1),B);
  Theta     = zeros(B,length(ind));
  thet0     = param(ind);
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
    theta     = uncmin(thet0,'mleq',0);
    if theta(21)<0;
      theta(21:24) = -theta(21:24);
    end;
    if theta(25)<0;
      theta(25:27) = -theta(25:27);
    end;
    if theta(28)<0;
      theta(28:29) = -theta(28:29);
    end;
    if theta(30)<0;
      theta(30)    = -theta(30);
    end;
    Theta(i,:) = theta';
  end;
  se  = std(Theta)';
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
                                                                                
%save runmle

