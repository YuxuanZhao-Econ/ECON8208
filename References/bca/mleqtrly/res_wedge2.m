function R = res_wedge2(Z,param);
%RES_WEDGE     residual for the growth model in ``Business Cycle Accounting.''
%

%              Ellen R. McGrattan, 11-1-02
%              Revised, ERM, 11-1-02

%______________________________________________________________________________
%
% PARAMETERS

gn         = param(1);
gz         = param(2);
beta       = param(3);
delta      = param(4);
psi        = param(5);
sigma      = param(6);
theta      = param(7);
beth       = beta*(1+gz)^(-sigma);
logzbar    = param(8);
taulbar    = param(9);
tauxbar    = param(10);
loggbar    = param(11);
Az         = param(12);
Al         = param(13);
Ax         = param(14);
Ag         = param(15);

%______________________________________________________________________________
%
% VARIABLES APPEARING IN RESIDUALS

k2         = exp(Z(1));
k1         = exp(Z(2));
k          = exp(Z(3));
sz1        = exp(Z(4));
sz         = exp(Z(5));
staul1     = Z(6);
staul      = Z(7);
staux1     = Z(8);
staux      = Z(9);
sg1        = exp(Z(10));
sg         = exp(Z(11));

z1         = exp(Az*log(sz1)+(1-Az)*logzbar);
z          = exp(Az*log(sz)+(1-Az)*logzbar);
taul1      = Al*staul1+(1-Al)*taulbar;
taul       = Al*staul+(1-Al)*taulbar;
taux1      = Ax*staux1+(1-Ax)*tauxbar;
taux       = Ax*staux+(1-Ax)*tauxbar;
g1         = exp(Ag*log(sg1)+(1-Ag)*loggbar);
g          = exp(Ag*log(sg)+(1-Ag)*loggbar);

l          = 1/(1+.75*psi/(1-taul)/(1-theta));
l1         = 1/(1+.75*psi/(1-taul1)/(1-theta));
for i=1:5;
  y        = k^theta*(z*l)^(1-theta);
  y1       = k1^theta*(z1*l1)^(1-theta);
  c        = y-(1+gz)*(1+gn)*k1+(1-delta)*k-g;
  c1       = y1-(1+gz)*(1+gn)*k2+(1-delta)*k1-g1;
  res      = psi*c*l/y-(1-taul)*(1-theta)*(1-l); 
  res1     = psi*c1*l1/y1-(1-taul1)*(1-theta)*(1-l1); 

  lp       = l+.0001;
  l1p      = l1+.0001;
  y        = k^theta*(z*lp)^(1-theta);
  y1       = k1^theta*(z1*l1p)^(1-theta);
  c        = y-(1+gz)*(1+gn)*k1+(1-delta)*k-g;
  c1       = y1-(1+gz)*(1+gn)*k2+(1-delta)*k1-g1;
  dres     = (psi*c*lp/y-(1-taul)*(1-theta)*(1-lp)-res)/.0001; 
  dres1    = (psi*c1*l1p/y1-(1-taul1)*(1-theta)*(1-l1p)-res1)/.0001; 
 
  l        = l-res/dres; 
  l1       = l1-res1/dres1; 
end;
y          = k^theta*(z*l)^(1-theta);
y1         = k1^theta*(z1*l1)^(1-theta);
c          = y-(1+gz)*(1+gn)*k1+(1-delta)*k-g;
c1         = y1-(1+gz)*(1+gn)*k2+(1-delta)*k1-g1;

R          = (1+taux)*c^(-sigma)*(1-l)^(psi*(1-sigma))- ...
             beth*c1^(-sigma)*(1-l1)^(psi*(1-sigma))* ...
             (theta*y1/k1+(1-delta)*(1+taux1));

