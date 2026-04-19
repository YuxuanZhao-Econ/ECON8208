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

x0a=[ ...
  -0.02226872217051
   0.32978559809016
   0.48394207534840
  -1.52999741405026
   0.98801271544095
  -0.00226977309673
  -0.07100307588320
  -0.00999346782540
   0.00244074124495
   0.97757396585901
   0.01693721453774
   0.00674512076127
   0.01337151951881
   0.02802098424106
   0.73607405720696
   0.02662383134457
  -0.00164528635451
  -0.00622796971267
   0.15127132454188
   0.98868759291213
   0.01164416023534
   0.00123692627331
  -0.00533935134032
  -0.00079381859001
   0.00644997326129
   0.00230641582338
   0.00639266420784
   0.00981139512775
   0.01374609848920
   0.00560226065257]; % result from runmle (with no adjustment costs)

x0b =[ ...
   0.53361937810378
  -0.18731348653706
   0.26493094443216
  -2.69213183272746
   0.97592720989221
   0.00511700228946
  -0.01095213324967
                  0
   0.01585451398443
   0.99789629719139
  -0.00190027448645
                  0
   0.01124520344064
   0.03728906363076
   0.93942225679282
                  0
                  0
                  0
                  0
   0.98319923810848
   0.02382866874935
  -0.01028081000637
   0.00553372214162
                  0
   0.02739908940082
  -0.03354291115539
                  0
   0.04951600718821
                  0
   0.10194597840286]; % result from initpw with annual adja=3.22
             
x0c=[ ...
   0.53845545582135
  -0.18358973491425
   0.27053078305491
  -2.77729398159333
   0.97946963467802
   0.01756329627717
  -0.04405183153630
                  0
   0.01309612558289
   0.98798065620949
   0.01079347875435
                  0
   0.00477921775529
   0.01933051249273
   0.92488787795934
                  0
                  0
                  0
                  0
   0.98717434831117
   0.02396761427982
  -0.00987436176711
  -0.01693235174207
                  0
   0.02737005516313
  -0.06560608935313
                  0
   0.12084347484485
                  0
   0.10034489721325];  % result from initpw with annual adja=12.88
                       % and maximum eigenvalue on P^(1/4) scaled 

x0      = x0c;
adja    = 4*12.88;
X       = zeros(30,20);
[x1,f,g,code,status]   = uncmin(x0,'mleqadj',adja);
X(:,1) = x1;
F(1,1) = f;
[x1,f,g,code,status]   = uncmin(x1,'mleqadj',adja);
X(:,2) = x1;
F(2,1) = f;
%
% Move away to see if we get more improvement
%
x2 = x1;
for i=3:50;
  [x2,f,g,code,status]   = uncmin(x2*.99,'mleqadj',adja);
  F(i,1) = f;
  X(:,i) = x2;
  disp(F)
  pause(10)
end;
i           = find(F==min(F));
x           = X(:,i(1));
L           = F(i(1));
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
    [f1,f2]  = mleseqadj(param+del(:,i),adja);
    [m1,m2]  = mleseqadj(param-del(:,i),adja);
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
  [L,Lt,ut,X0,Cbar,A,K,D,gz] = mleseqadj(param,adja);
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
    theta     = uncmin(thet0,'mleqadj',adja);
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
                                                                                
%save runmleadj

