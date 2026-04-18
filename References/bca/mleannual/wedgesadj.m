% Wedges     Use results from MLE for the period 1901--1940 to 
%            derive the wedges used in the ./depression codes
%

%            Ellen McGrattan, 2-16-02
%            Revised, ERM, 10-24-06

%------------------------------------------------------------------------------

load uszvar1.dat
global ZVAR
t    = uszvar1(1:40,1);
ZVAR = uszvar1(1:40,2:5);
KBEA = uszvar1(1:40,6);
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

adja = 4*3.22;
x0   = x0(:,3);
%[x,f,g,code,status]         = uncmin(x0,'mle1adj',adja);
[L,Sbar,P0,P,Q,A,B,C,param] = mle1adj(x0,adja);
Y0      = 29;
gn      = param(1);
gz      = param(2);
beta    = param(3);
delta   = param(4);
psi     = param(5);
sigma   = param(6);
theta   = param(7);
adja    = param(8);
adjb    = (1+gn)*(1+gz)-1+delta;
z       = exp(Sbar(1));
taul    = Sbar(2);
taux    = Sbar(3);
g       = exp(Sbar(4));
betah   = beta*(1+gz)^(-sigma);
kl      = ((1+taux)*(1-betah*(1-delta))/(betah*theta))^(1/(theta-1))*z;
yk      = (kl/z)^(theta-1);
xi1     = yk-(1+gz)*(1+gn)+1-delta;
xi2     = (1-taul)*(1-theta)*(kl)^theta*z^(1-theta)/psi;
k       = (xi2+g)/(xi1+xi2/kl);
c       = xi1*k-g;
l       = k/kl;
y       = yk*k;
x       = y-c-g;
lk      = log(k);
lc      = log(c);
ll      = log(l);
ly      = log(y);
lx      = log(x);
lg      = log(g);
lz      = log(z);
T       = length(ZVAR);
Y       = log(ZVAR)-log([(1+gz).^[0:T-1]',(1+gz).^[0:T-1]', ...
                         ones(T,1),(1+gz).^[0:T-1]']);
lkbea   = log(KBEA)-log((1+gz).^[0:T-1]');
lyt     = Y(:,1);
lxt     = Y(:,2);
llt     = Y(:,3);
lgt     = Y(:,4);

lkt(Y0,1)= lk;
Kt(Y0,1) = exp(lkbea(Y0));

for i=Y0:T;
  lktp(i,1)  = lk+((1-delta)*(lkt(i)-lk)+x/k*(lxt(i)-lx))/(1+gz)/(1+gn);
  lkt(i+1,1) = lktp(i);
  phit(i,1)  = adja/2*(exp(lxt(i))/Kt(i)-adjb)^2;
  Ktp(i,1)   = ((1-delta)*Kt(i)+exp(lxt(i))-phit(i)*Kt(i))/(1+gz)/(1+gn);
  Kt(i+1,1)  = Ktp(i);
end;
for i=Y0-1:-1:1;
  lktp(i,1)  = lkt(i+1,1);
  lkt(i,1)   = lk+((1+gz)*(1+gn)*(lktp(i)-lk)-x/k*(lxt(i)-lx))/(1-delta);
  Ktp(i,1)   = Kt(i+1,1);
  tem        = roots([1-delta-adja*adjb^2/2,(1+adja*adjb)*exp(lxt(i))- ...
                     (1+gn)*(1+gz)*Ktp(i),-adja/2*exp(lxt(i))^2]);
  Kt(i,1)    = max(tem);
end;
lkt      = lkt(1:T);
Kt       = Kt(1:T);
lct      = lc+(y*(lyt-ly)-x*(lxt-lx)-g*(lgt-lg))/c;
lzt      = lz+(lyt-ly-theta*(lkt-lk))/(1-theta)-llt+ll;
tault    = taul+(1-taul)*(lyt-ly-lct+lc-1/(1-l)*(llt-ll));
tauxt    = (lxt-C(2,1)*lkt-C(2,2)*lzt-C(2,3)*tault-C(2,5)*lgt-C(2,6))/C(2,4);
tauxchk  = (lyt-C(1,1)*lkt-C(1,2)*lzt-C(1,3)*tault-C(1,5)*lgt-C(1,6))/C(1,4);
Ct       = exp(lyt)-exp(lxt)-exp(lgt);
Zt       = (exp(lyt)./(Kt.^theta.*exp(llt).^(1-theta))).^(1/(1-theta));
Ztbea    = (exp(lyt)./(exp(lkbea).^theta.*exp(llt).^(1-theta))).^(1/(1-theta));
Tault    = 1-psi/(1-theta)* (Ct./exp(lyt)) .*(exp(llt)./(1-exp(llt)));

z        = [log(Zt),Tault,tauxt,log(exp(lgt))];
