% Wedges     Use results from MLE for the period 1901--1940 to 
%            derive the wedges used in the ./depression codes
%

%            Ellen McGrattan, 2-16-02
%            Revised, ERM, 10-10-06

%------------------------------------------------------------------------------

load uszvar1.dat
global ZVAR
t    = uszvar1(1:40,1);
ZVAR = uszvar1(1:40,2:5);
KBEA = uszvar1(1:40,6);
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

[x,f,g,code,status]         = uncmin(x0,'mle1',0);
[L,Sbar,P0,P,Q,A,B,C,param] = mle1(x);
Y0      = 29;
gn      = param(1);
gz      = param(2);
beta    = param(3);
delta   = param(4);
psi     = param(5);
sigma   = param(6);
theta   = param(7);
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
  Ktp(i,1)   = ((1-delta)*Kt(i)+exp(lxt(i)))/(1+gz)/(1+gn);
  Kt(i+1,1)  = Ktp(i);
end;
for i=Y0-1:-1:1;
  lktp(i,1)  = lkt(i+1,1);
  lkt(i,1)   = lk+((1+gz)*(1+gn)*(lktp(i)-lk)-x/k*(lxt(i)-lx))/(1-delta);
  Ktp(i,1)   = Kt(i+1,1);
  Kt(i,1)    = ((1+gz)*(1+gn)*Ktp(i)-exp(lxt(i)))/(1-delta);
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
