% Wedgescu     Use results from MLE for the period 1901--1940 to
%              derive the wedges used in the ./depression codes
%
                                                                                
%              Ellen McGrattan, 3-14-03
%              Revised, ERM, 10-10-06

%------------------------------------------------------------------------------

load uszvar1.dat
global ZVAR
t    = uszvar1(1:40,1);
ZVAR = uszvar1(1:40,2:5);
KBEA = uszvar1(1:40,6);
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

[x,f,g,code,status]         = uncmin(x0,'mle1cu',0);
[L,Sbar,P0,P,Q,A,B,C,param] = mle1cu(x);
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
xi      = (1+taux)*(1-betah*(1-delta))/(betah*theta);
xi1     = (psi+1-taul)*xi-psi*((1+gn)*(1+gz)-1+delta);
xi2     = psi*g;
xi3     = (1-taul)*z;
k       = (xi3/xi1)^(1/(1-theta));
for i=1:10;
  k     = k-(xi1*k-xi2-xi3*k^theta)/(xi1-xi3*theta*k^(theta-1));
end;
l       = xi*k^(1-theta)/z;
c       = (xi-(1+gn)*(1+gz)+1-delta)*k-g;
y       = xi*k;
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
lzt      = lz+lyt-ly-theta*(lkt-lk)-llt+ll;
tault    = taul+(1-taul)*(lyt-ly-lct+lc-1/(1-l)*(llt-ll));
tauxt    = (lxt-C(2,1)*lkt-C(2,2)*lzt-C(2,3)*tault-C(2,5)*lgt-C(2,6))/C(2,4);
tauxchk  = (lyt-C(1,1)*lkt-C(1,2)*lzt-C(1,3)*tault-C(1,5)*lgt-C(1,6))/C(1,4);
Ct       = exp(lyt)-exp(lxt)-exp(lgt);
Zt       = exp(lyt)./(Kt.^theta.*exp(llt));
Ztbea    = exp(lyt)./(exp(lkbea).^theta.*exp(llt));
Tault    = 1-psi* (Ct./exp(lyt)) .*(exp(llt)./(1-exp(llt)));

z        = [log(Zt),Tault,tauxt,log(exp(lgt))];

