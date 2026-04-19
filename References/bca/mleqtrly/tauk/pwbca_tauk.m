% PWBCA      Use results from MLE for the period 1959:1--2004:3 to
%            (1) plot output, labor and investment for the model and
%            the data and to (2) compute variance decompositions
%            for the 1982 recession.  This code also allows for 
%            3 levels of adjustment costs
%

%            Ellen McGrattan, 2-16-02
%            Revised, ERM, 11-07-06

%------------------------------------------------------------------------------

load ../uszvarq.dat
global ZVAR
t    = uszvarq(:,1);
ZVAR = uszvarq(:,2:5);

josh
adja = 51.20951263046717;

[L,Sbar,P0,P,Q,A,B,C,param] = mleqadj_tauk(x0,adja);
Y0       = 81;
gn       = param(1);
gz       = param(2);
beta     = param(3);
delta    = param(4);
psi      = param(5);
sigma    = param(6);
theta    = param(7);
adja     = param(8);
adjb     = (1+gn)*(1+gz)-1+delta;
z        = exp(Sbar(1));
taul     = Sbar(2);
tauk     = Sbar(3);
g        = exp(Sbar(4));
betah    = beta*(1+gz)^(-sigma)*(1-tauk);
kl       = ((1-betah*(1-delta))/(betah*theta))^(1/(theta-1))*z;
yk       = (kl/z)^(theta-1);
xi1      = yk-(1+gz)*(1+gn)+1-delta;
xi2      = (1-taul)*(1-theta)*(kl)^theta*z^(1-theta)/psi;
k        = (xi2+g)/(xi1+xi2/kl);
c        = xi1*k-g;
l        = k/kl;
y        = yk*k;
x        = y-c-g;
lk       = log(k);
lc       = log(c);
ll       = log(l);
ly       = log(y);
lx       = log(x);
lg       = log(g);
lz       = log(z);
T        = length(ZVAR);
Y        = log(ZVAR)-log([(1+gz).^[0:T-1]',(1+gz).^[0:T-1]', ...
                         ones(T,1),(1+gz).^[0:T-1]']);
lyt      = Y(:,1);
lxt      = Y(:,2);
llt      = Y(:,3);
lgt      = Y(:,4);
lkt(1,1) = lk;
Kt(1,1)  = exp(lk);

for i=1:T;
  lktp(i,1)  = lk+((1-delta)*(lkt(i)-lk)+x/k*(lxt(i)-lx))/(1+gz)/(1+gn);
  lkt(i+1,1) = lktp(i);
  phit(i,1)  = adja/2*(exp(lxt(i))/Kt(i)-adjb)^2;
  Ktp(i,1)   = ((1-delta)*Kt(i)+exp(lxt(i))-phit(i)*Kt(i))/(1+gz)/(1+gn);
  Kt(i+1,1)  = Ktp(i);
end;
lkt      = lkt(1:T);
Kt       = Kt(1:T);
lct      = lc+(y*(lyt-ly)-x*(lxt-lx)-g*(lgt-lg))/c;
lzt      = lz+(lyt-ly-theta*(lkt-lk))/(1-theta)-llt+ll;
tault    = taul+(1-taul)*(lyt-ly-lct+lc-1/(1-l)*(llt-ll));
taukt    = (lxt-C(2,1)*lkt-C(2,2)*lzt-C(2,3)*tault-C(2,5)*lgt-C(2,6))/C(2,4);
taukchk  = (lyt-C(1,1)*lkt-C(1,2)*lzt-C(1,3)*tault-C(1,5)*lgt-C(1,6))/C(1,4);
Ct       = exp(lyt)-exp(lxt)-exp(lgt);
Zt       = (exp(lyt)./(Kt.^theta.*exp(llt).^(1-theta))).^(1/(1-theta));
Tault    = 1-psi/(1-theta)* (Ct./exp(lyt)) .*(exp(llt)./(1-exp(llt)));
Xt0      = [lkt,lzt,tault,taukt,lgt,ones(T,1)];
YM0      = Xt0*C';

s0       = [lzt(Y0);tault(Y0);taukt(Y0);lgt(Y0)];

[i,j,C0] = fixexpadj_tauk(Sbar,P,Q,s0,[0;0;0;0],adja);
[i,j,C1] = fixexpadj_tauk(Sbar,P,Q,s0,[1;0;0;0],adja);
[i,j,C2] = fixexpadj_tauk(Sbar,P,Q,s0,[0;1;0;0],adja);
[i,j,C3] = fixexpadj_tauk(Sbar,P,Q,s0,[0;0;1;0],adja);
[i,j,C4] = fixexpadj_tauk(Sbar,P,Q,s0,[0;0;0;1],adja);
o        = ones(183,1);
YMn      = (Xt0-o*Xt0(Y0,:))*C0'+o*YM0(Y0,:);
YMz      = (Xt0-o*Xt0(Y0,:))*(C1-C0)'+o*YM0(Y0,:);
YMl      = (Xt0-o*Xt0(Y0,:))*(C2-C0)'+o*YM0(Y0,:);
YMx      = (Xt0-o*Xt0(Y0,:))*(C3-C0)'+o*YM0(Y0,:);
YMg      = (Xt0-o*Xt0(Y0,:))*(C4-C0)'+o*YM0(Y0,:);
YMall    = (Xt0-o*Xt0(Y0,:))*C'+o*YM0(Y0,:);
YMnox    = (Xt0-o*Xt0(Y0,:))*(C1+C2+C4-2*C0)'+o*YM0(Y0,:);
YMnoz    = (Xt0-o*Xt0(Y0,:))*(C2+C3+C4-2*C0)'+o*YM0(Y0,:);

figure(1)
fig1 =[t,[exp(lyt-lyt(Y0)),(Zt/Zt(Y0)).^(1-theta), ...
       (1-Tault)/(1-Tault(Y0)), ...
       (1-taukt)/(1-taukt(Y0))]*100];
orient portrait
plot(t,exp(lyt-lyt(Y0)), ...
     t,(Zt/Zt(Y0)).^(1-theta), ...
     t,(1-Tault)/(1-Tault(Y0)), ...
     t,(1-taukt)/(1-taukt(Y0)))
title('Figure 1. U.S. Output and Measured Wedges')
legend('Data','Efficiency Wedge','Labor Wedge','Investment Wedge')
axis([1979,1986,.8,1.2])

figure(2)
fig2d = [t,[exp(Y(:,1)-Y(Y0,1)),exp(Y(:,3)-Y(Y0,3)),exp(Y(:,2))]*100];
fig2mz= [t,[exp(YMz(:,1)-YMz(Y0,1)),exp(YMz(:,3)-YMz(Y0,3)),exp(YMz(:,2))]*100];
fig2ml= [t,[exp(YMl(:,1)-YMl(Y0,1)),exp(YMl(:,3)-YMl(Y0,3)),exp(YMl(:,2))]*100];
orient tall
subplot(311)
plot(t,exp(Y(:,1)-Y(Y0,1)),t,exp(YMz(:,1)-YMz(Y0,1)),t,exp(YMl(:,1)-YMl(Y0,1)))
title('Figure 2. U.S.~Data and Models with an Efficiency or Labor Wedge')
legend('Data','Efficiency wedge','Labor wedge')
axis([1979,1986,.8,1.2])
subplot(312)
plot(t,exp(Y(:,3)-Y(Y0,3)),t,exp(YMz(:,3)-YMz(Y0,3)),t,exp(YMl(:,3)-YMl(Y0,3)))
axis([1979,1986,.9,1.1])
subplot(313)
plot(t,exp(Y(:,2)),t,exp(YMz(:,2)),t,exp(YMl(:,2)))
axis([1979,1986,.1,.4])

figure(3)
fig3d = [t,[exp(Y(:,1)-Y(Y0,1)),exp(Y(:,3)-Y(Y0,3)),exp(Y(:,2))]*100];
fig3mg= [t,[exp(YMg(:,1)-YMg(Y0,1)),exp(YMg(:,3)-YMg(Y0,3)),exp(YMg(:,2))]*100];
fig3mx= [t,[exp(YMx(:,1)-YMx(Y0,1)),exp(YMx(:,3)-YMx(Y0,3)),exp(YMx(:,2))]*100];
orient tall
subplot(311)
plot(t,exp(Y(:,1)-Y(Y0,1)),t,exp(YMg(:,1)-YMg(Y0,1)),t,exp(YMx(:,1)-YMx(Y0,1)))
title('Figure 3. U.S.~Data and Models with an Investment or Government Wedge')
legend('Data','Government wedge','Investment wedge')
axis([1979,1986,.8,1.2])
subplot(312)
plot(t,exp(Y(:,3)-Y(Y0,3)),t,exp(YMg(:,3)-YMg(Y0,3)),t,exp(YMx(:,3)-YMx(Y0,3)))
axis([1979,1986,.8,1.2])
subplot(313)
plot(t,exp(Y(:,2)),t,exp(YMg(:,2)),t,exp(YMx(:,2)))
axis([1979,1986,.1,.4])

figure(4)
fig4d   = [t,[exp(Y(:,1)-Y(Y0,1)),exp(Y(:,3)-Y(Y0,3)),exp(Y(:,2))]*100];
fig4mnox= [t,[exp(YMnox(:,1)-YMnox(Y0,1)),exp(YMnox(:,3)-YMnox(Y0,3)), ...
              exp(YMnox(:,2))]*100];
fig4mnoz= [t,[exp(YMnoz(:,1)-YMnoz(Y0,1)),exp(YMnoz(:,3)-YMnoz(Y0,3)), ...
              exp(YMnoz(:,2))]*100];
orient tall
subplot(311)
plot(t,exp(Y(:,1)-Y(Y0,1)),t,exp(YMnox(:,1)-YMnox(Y0,1)), ...
     t,exp(YMnoz(:,1)-YMnoz(Y0,1)))
title(['Figure 4. U.S.~Data and Models without an Investment or ', ...
                  'Efficiency Wedge'])
legend('Data','No investment wedge','No efficiency wedge')
axis([1979,1986,.8,1.2])
subplot(312)
plot(t,exp(Y(:,3)-Y(Y0,3)),t,exp(YMnox(:,3)-YMnox(Y0,3)), ...
     t,exp(YMnoz(:,3)-YMnoz(Y0,3)))
axis([1979,1986,.8,1.2])
subplot(313)
plot(t,exp(Y(:,2)),t,exp(YMnox(:,2)),t,exp(YMnoz(:,2)))
axis([1979,1986,.1,.4])


