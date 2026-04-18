%USDATA  construct US time series 1901-2000 and extract samples for
%        estimation in mle*.m.
%
%        Measures constructed:
%
%           Y     = GNP - Sales tax 
%                       - Military compensation 
%                       + Services and depreciation of durables
%           C     = Consumer nondurables (CND) plus services (CS)
%                       - Portion of sales tax for CND&CS
%                       + Services and depreciation of durables
%           X     = Gross private domestic investment (fixed+inventory)
%                       + Government investment
%                       + Consumer durables (CD) 
%                       - Portion of sales tax for CD 
%                       - Military equipment 
%                       - 1/2 military facilities
%                       + Net factor payments
%           K     = Capital stock corresponding to X
%           G     = Government consumption
%                       - Military compensation 
%                       + Military equipment 
%                       + 1/2 military facilities
%                       + Net exports in goods and services
%           P     = GNP deflator
%           H     = Civilian manhours  
%           P16   = Population over 16
%

%        Ellen McGrattan, 2-16-02
%        Revised, ERM, 2-8-05


load data/kenn_us.dat
load data/kenr_us.dat
load data/mhrs_us.dat
load data/popu_us.dat
load data/na101.dat
load data/na109.dat
load data/na110.dat
load data/na305.dat
load data/na307.dat
load data/na309.dat
load data/na501.dat
load data/na511a.dat
load data/na511b.dat
load data/na602a.dat
load data/na602b.dat
load data/na602c.dat
load data/na605a.dat
load data/na603a.dat
load data/na605b.dat
load data/na605c.dat
load data/na608a.dat
load data/na608b.dat
load data/na608c.dat
load data/na609b.dat
load data/na609c.dat
load data/fa11.dat
load data/fa71.dat
load data/fa15.dat
load data/fa75.dat

%
% Adjust series to common units
%
fa11           = fa11/1000;
fa71           = fa71/1000;
fa15           = fa15/1000;
fa75           = fa75/1000;
na602a         = na602a/1000;
na602b         = na602b/1000;
na602c         = na602c/1000;
na603a         = na603a/1000;
na605a         = na605a/1000000;
na605b         = na605b/1000000;
na605c         = na605c/1000000;
na608a         = na608a/1000000;
na608b         = na608b/1000000;
na608c         = na608c/1000000;
na609b         = na609b/1000 *1.1;
na609c         = na609c/1000 *1.1;
na511          = [na511a(1:2,:),na511b(1:2,2:5)];
na605          = [na605a([1:4,80],1:19),na605b([1:4,81],1:39), ...
                                        na605c([1:4,81],:)];
na608          = [na608a([1:4,80],1:19),na608b([1:4,81],1:39), ...
                                        na608c([1:4,81],:)];
na609          = [NaN*ones(4,19),na609b(1:4,1:39),na609c(1:4,:);
                  NaN*ones(1,19),1700*[na605b(81,1:39),na605c(81,:)]];
kenn_us        = kenn_us/1000;
mhrs_us        = mhrs_us/1000;
popu_us        = popu_us/1000000;
fa15(11,1:13)  = zeros(1,13);
fa75(19,1:13)  = zeros(1,13);
t              = [1901:2000]';

%
% Population
% Sources: Historical Statistics, Colonial to 1970, Series A6-8
%          Economic Report of the President 2001, Table B-34
%
T      = 113:212;
p16    = popu_us(T,5);

%
% Pre-1929 National Accounts
% Source: Kendrick (1961)   
%
T      = 113:141;   % 1901--1929
NA     = NaN*ones(71,1);
kc     = [kenn_us(T,2);NA];
kgpdf  = [kenn_us(T,3);NA];
kgpdi  = [kenn_us(T,4);NA];
knfi   = [kenn_us(T,5);NA];
kg     = [kenn_us(T,6);NA];
kgnp   = [kenn_us(T,7);NA];
krgnp  = [kenr_us(T,7);NA];
krgpdi = [kenr_us(T,4);NA];

%
% Post-1929 National Accounts
% Source: NIPA at www.bea.doc.gov (downloaded 6/24/02)
%
T      = 1:72;     % 1929--2000 
NA     = NaN*ones(28,1);
nrgnp  = [NA;na110(5,T)'];
nrgpdi = [NA;na511(2,T)'];
ngnp   = [NA;na109(5,T)'];
ngdp   = [NA;na101(2,T)'];
nc     = [NA;na101(3,T)'];
ncd    = [NA;na101(4,T)'];
ncnd   = [NA;na101(5,T)'];
ncs    = [NA;na101(6,T)'];
nipd   = [NA;na101(7,T)'];
ngpdf  = [NA;na101(8,T)'];
ngpdi  = [NA;na101(13,T)'];
nnx    = [NA;na101(14,T)'];
nnxe   = [NA;na101(15,T)'];
nnxi   = [NA;na101(18,T)'];
ng     = [NA;na101(21,T)'];
ngc    = [NA;na309(2,T)'];
ngi    = [NA;na501(21,T)'];
ngims  = [NA;na307(13,T)'];
ngime  = [NA;na307(14,T)'];
nnfi   = ngnp-ngdp+nnx;
nfac   = ngnp-ngdp;
nstx   = [zeros(28,1);sum(na305([4,18,34],T))'];
nmilc  = [zeros(1,28),na603a(80,:).*na602a(76,:)./na603a(76,:), ...
                      na602b(81,2:40),na602c(81,2:14)]';

%
% Pre-1947 Manhours
% Source: Kendrick (1961)   
%
T      = 113:210;
kh     = [mhrs_us(T,3);NaN;NaN];

%
% Post-1947 Manhours
% Source: NIPA at www.bea.doc.gov (downloaded 6/24/02)
%
T      = 1:72;     % 1929--2000 
NA     = NaN*ones(28,1);
nh     = [NA;
          na609(2,T)'.*(1+(na608(2,T)-na605(2,T))'./na605(2,T)') - ...
          na609(5,T)'.*(1+(na608(5,T)-na605(5,T))'./na605(5,T)')];

%
% Investments and Capital Stocks 
% Source: Fixed Assets at www.bea.doc.gov (downloaded 6/24/02)
%
igpdf  = fa15(4,:)';
icd    = fa15(14,:)';
igi    = sum(fa15(11:13,:))';
igime  = fa75(19,:)';
igims  = fa75(26,:)';
igimf  = fa75(30,:)';

cgpdf  = [NaN*ones(24,1);fa11(4,:)'];
ccd    = [NaN*ones(24,1);fa11(14,:)'];
cgi    = [NaN*ones(24,1);fa11(9,:)'];
cgime  = [NaN*ones(24,1);fa71(19,:)'];
cgims  = [NaN*ones(24,1);fa71(26,:)'];
cgimf  = [NaN*ones(24,1);fa71(30,:)'];

dgpdf  = NaN*ones(100,1);
dcd    = dgpdf;
dgi    = dgpdf;
dgime  = dgpdf;
dgims  = dgpdf;
dgimf  = dgpdf;
for i =100:-1:26;
  dgpdf(i)  = -cgpdf(i)+cgpdf(i-1)+igpdf(i);
  dcd(i)    = -ccd(i)  +ccd(i-1)  +icd(i);
  dgi(i)    = -cgi(i)  +cgi(i-1)  +igi(i);
  dgime(i)  = -cgime(i)+cgime(i-1)+igime(i);
  dgims(i)  = -cgims(i)+cgims(i-1)+igims(i);
  dgimf(i)  = -cgimf(i)+cgimf(i-1)+igimf(i);
end;
drgpdf = avgnan(dgpdf./cgpdf);
drcd   = avgnan(dcd./ccd);
drgi   = avgnan(dgi./cgi);
drgime = avgnan(dgime./cgime);
drgims = avgnan(dgims./cgims);
drgimf = avgnan(dgimf./cgimf);
drnfi  = drgpdf;
drfac  = drgpdf;

for i =24:-1:1;
  cgpdf(i)  = (cgpdf(i+1)-igpdf(i+1))/(1-drgpdf);
  ccd(i)    = (ccd(i+1)  -icd(i+1)  )/(1-drcd*.9575);
  cgi(i)    = (cgi(i+1)  -igi(i+1)  )/(1-drgi);
  cgime(i)  = (cgime(i+1)-igime(i+1))/(1-drgime);
  cgims(i)  = (cgims(i+1)-igims(i+1))/(1-drgims);
  cgimf(i)  = (cgimf(i+1)-igimf(i+1))/(1-drgimf);
end;
for i =25:-1:2;
  dcd(i)    = -ccd(i)  +ccd(i-1)  +icd(i);
end;
dcd(1) = dcd(2);

%
% Combine data to obtain long time series:
%

rgnp   = [krgnp(1:29)*nrgnp(29)/krgnp(29); nrgnp(30:100)];
rgpdi  = [krgpdi(1:29)*nrgpdi(29)/krgpdi(29); nrgpdi(30:100)];
gnp    = [kgnp(1:28); ngnp(29:100)];
stx    = nstx;
milc   = nmilc;
c      = [kc(1:28);    nc(29:100)];
cd     = [icd(1:28);   ncd(29:100)];
g      = [kg(1:28);    ng(29:100)];
gi     = [igi(1:28);   ngi(29:100)];
gime   = [igime(1:28); ngime(29:100)];
gimf   = [igimf(1:28); igimf(29:100)];
gpdf   = [kgpdf(1:28); ngpdf(29:100)];
gpdi   = [kgpdi(1:28); ngpdi(29:100)];
nfi    = [knfi(1:28);  nnfi(29:100)];
fac    = [.005*kgnp(1:28); ngnp(29:100)-ngdp(29:100)];
nx     = nfi-fac;
h      = [kh(1:47);    nh(48:100)];
gc     = g-gi;

crgpdi(100)  = 1507.1;
for i =99:-1:1;
  crgpdi(i)  = crgpdi(i+1)-rgpdi(i+1);
end;

cnfi(1)     = 0;
cfac(1)     = 0;
for i = 2:100;
  cnfi(i)   = cnfi(i-1)*(1-drnfi)+nfi(i);
  cfac(i)   = cfac(i-1)*(1-drfac)+fac(i);
end;
crgpdi = crgpdi(:);
cnfi   = cnfi(:);
cfac   = cfac(:);
bgpdf  = [(cgpdf(1)-igpdf(1))/(1-drgpdf);cgpdf(1:99)];
bcd    = [(ccd(1)  -icd(1))  /(1-drcd);  ccd(1:99)];
bgi    = [(cgi(1)  -igi(1))  /(1-drgi);  cgi(1:99)];
bgime  = [(cgime(1)-igime(1))/(1-drgime);cgime(1:99)];
bgims  = [(cgims(1)-igims(1))/(1-drgims);cgims(1:99)];
bgimf  = [(cgimf(1)-igimf(1))/(1-drgimf);cgimf(1:99)];
brgpdi = [(crgpdi(1)-rgpdi(1))          ;crgpdi(1:99)];
bnfi   = [(cnfi(1) -nfi(1))  /(1-drnfi); cnfi(1:99)];
bfac   = [(cfac(1) -fac(1))  /(1-drfac); cfac(1:99)];

%
% Measures to match growth model analogues:
%
P      = gnp./rgnp;
Y      = gnp -stx -milc +.04*ccd +dcd;
C      = (c-cd)  -(c-cd)./c .*stx +.04*ccd +dcd;
X      = gpdf +gpdi +fac +gi -gime -.5*gimf +cd -cd./c .*stx;
K      = bgpdf +P.*brgpdi +bfac +bgi -bgime -.5*bgimf +bcd;
G      = gc -milc +gime +.5*gimf + nx;
H      = h;
P16    = p16;
L      = (H./P16)/5000;
lam    = 1.016.^(t-1901);
Ypc    = Y./(P.*P16);
Cpc    = C./(P.*P16);
Xpc    = X./(P.*P16);
Gpc    = G./(P.*P16);
Kpc    = K./(P.*P16);

uszvar1= [t,Ypc/Ypc(29)*lam(29),Xpc/Ypc(29)*lam(29),L, ...
            Gpc/Ypc(29)*lam(29),Kpc/Ypc(29)*lam(29)];
uszvar2= [t,Ypc/Ypc(79)*lam(25),Xpc/Ypc(79)*lam(25),L, ...
            Gpc/Ypc(79)*lam(25),Kpc/Ypc(79)*lam(25)];
disp(['Year, Output, Investment, Labor, Government Spending, ', ...
      'Beginning-of-Period Capital 1901-2000'])
disp(sprintf(' %4g %7.4f %7.4f %7.4f %7.4f %7.4f\n',uszvar1'))
disp(' ')
disp(['Year, Output, Investment, Labor, Government Spending, ', ...
      'Beginning-of-Period Capital 1955-2000'])
disp(sprintf(' %4g %7.4f %7.4f %7.4f %7.4f %7.3f\n',uszvar2(55:100,:)'))

psi    = 2.24;
theta  = 0.35;
DY     = Ypc./lam*lam(29);
DK     = Kpc./lam*lam(29);
A      = DY./(DK.^theta .* L.^(1-theta));
taun   = 1- (psi/(1-theta)) *(Cpc./Ypc) .* (L./(1-L));
long   = [t,log(A),1-taun];
