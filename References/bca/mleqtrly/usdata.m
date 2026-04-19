%USDATA    Set up data file for maximum likelihood estimation,
%          after adjusting NIPA output to exclude sales tax and 
%          include consumer durable services.  Store the matrix 
%          `mled' in uszvarq.dat.

%          Ellen McGrattan, 3-15-04
%          Revised, ERM, 2-1-05

load data/nipa115.dat
load data/nipa116.dat
load data/nipa119.dat
load data/nipa32.dat
load data/nipa33.dat
load data/nipa394.dat
load data/nipa395.dat
load data/atab10d.dat
load data/btab100d.dat
load data/nipa395.dat
load data/hours.dat

T     = 49:231;  % 1959:1-2004:3
rGDP  = nipa116(2,T)';
rPCE  = nipa116(3,T)';
pCD   = nipa119(4,T)';
rCD   = nipa115(4,T)'./nipa119(4,T)'*100;
rCND  = nipa115(5,T)'./nipa119(5,T)'*100;
rCS   = nipa115(6,T)'./nipa119(6,T)'*100;
rGPDI = nipa116(7,T)';
rEX   = nipa116(15,T)';
rIM   = nipa116(18,T)';
rG    = nipa116(21,T)';
rGC   = nipa395(3,T)'./nipa394(3,T)'*100;
rGI   = nipa395(4,T)'./nipa394(4,T)'*100;
rSTX  = (nipa32(6,T)+sum(nipa33([8,10],T)))'./nipa119(3,T)'*100;

T     = 29:211; % 1959:1-2004:3
nKCD  = btab100d(T,8);
nDCD  = atab10d(T,28)/1000;
rKCD  = nKCD./pCD;
rDCD  = nDCD./pCD;

T     = 1:183;  % 1959:1-2004:3
Pop   = hours(T,2);
H     = hours(T,3);

Y     = rGDP-rSTX+.04*rKCD+rDCD;
C     = rCND+rCS-(rCND+rCS)./(rCND+rCS+rCD).*rSTX+.04*rKCD+rDCD;
X     = rCD+rGPDI+rGI-rCD./(rCND+rCS+rCD).*rSTX;
G     = rGC+rEX-rIM;

t     = ptime(1959.1,length(H),4);
hpc   = H./Pop;
prd   = Y./H*10^9;
ypc   = Y./Pop*10^9;
xpc   = X./Pop*10^9;
gpc   = G./Pop*10^9;

gz    = (1.016^(1/4))^81;
mled  = [t,ypc/ypc(81)*gz,xpc/ypc(81)*gz,hpc/1300,gpc/ypc(81)*gz];
