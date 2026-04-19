%ADJCOSTPLOTS   Plot summaries of the adjustment cost results. See details 
%               about model economy in the Fortran code.
                                                                                
%               Ellen McGrattan, 12-01-01
%               Revised, ERM, 10-20-06

load bench.dat.case1
taux   = bench(:,13);
load bench.dat.case4
hrx    = bench(:,4);
ipx    = bench(:,5);
gnpx   = bench(:,6);
load bench.dat.case9
hr0    = bench(:,4);
ip0    = bench(:,5);
gnp0   = bench(:,6);
load adjcost.dat.case1
tauxa  = adjcost(:,13);
load adjcost.dat.case4
hrxa   = adjcost(:,4);
ipxa   = adjcost(:,5);
gnpxa  = adjcost(:,6);
load adjcost.dat.case9
hr0a   = adjcost(:,4);
ip0a   = adjcost(:,5);
gnp0a  = adjcost(:,6);
load adjcost.dat.case1e
tauxae = adjcost(:,13);
load adjcost.dat.case4e
hrxae  = adjcost(:,4);
ipxae  = adjcost(:,5);
gnpxae = adjcost(:,6);
load adjcost.dat.case9e
hr0ae  = adjcost(:,4);
ip0ae  = adjcost(:,5);
gnp0ae = adjcost(:,6);
s      = adjcost(:,1);
t0     = 1929;
t1     = t0-1+length(s);
t      = [t0:t1]';
wedges = [(1+taux(1))*ones(length(t),1)./(1+taux), ...
          (1+tauxa(1))*ones(length(t),1)./(1+tauxa), ...
          (1+tauxae(1))*ones(length(t),1)./(1+tauxae)];
load ../mleannual/uszvar1.dat
grate  = 1.016.^([0:99]');
nipal  = uszvar1./[ones(100,1),grate,grate,ones(100,1),grate,grate];
nipa   = nipal(29:100,:);
gnpdat = nipa(1:30,2);
ipdat  = nipa(1:30,3);
hrdat  = nipa(1:30,4);

figure(1)
  plot(t,gnpdat*100,t,wedges*100)
  axis([t0,t0+10,60,120])
title('Investment Wedges and US GNP')
legend('Data','No Costs','Costs at BGG level','Costs at 4 x BGG level')

figure(2)
  gnpmod = [(gnpx-gnp0-gnpx(1)+gnp0(1)), ...
            (gnpxa-gnp0a-gnpxa(1)+gnp0a(1)), ...
            (gnpxae-gnp0ae-gnpxae(1)+gnp0ae(1))]+gnpdat(1);
  plot(t,gnpdat*100,t,gnpmod*100)
  title('GNP')
  legend('Data','No Costs','Costs at BGG level','Costs at 4 x BGG level')
  axis([1929,1939,60,120])

figure(3)
  hrmod  = [(hrx-hr0-hrx(1)+hr0(1)), ...
            (hrxa-hr0a-hrxa(1)+hr0a(1)), ...
            (hrxae-hr0ae-hrxae(1)+hr0ae(1))]+hrdat(1);
  plot(t,hrdat*100/hrdat(1),t,hrmod*100/hrdat(1))
  title('Hours')
  legend('Data','No Costs','Costs at BGG level','Costs at 4 x BGG level')
  axis([1929,1939,60,120])

figure(4)
  ipmod  = [(ipx-ip0-ipx(1)+ip0(1)), ...
            (ipxa-ip0a-ipxa(1)+ip0a(1)), ...
            (ipxae-ip0ae-ipxae(1)+ip0ae(1))]+ipdat(1);
  plot(t,ipdat*100,t,ipmod*100)
  title('Investment')
  legend('Data','No Costs','Costs at BGG level','Costs at 4 x BGG level')
  axis([1929,1939,0,45])

fig1     = [t,gnpdat*100,wedges*100];
fig3mx   = [t,gnpmod(:,1)*100,hrmod(:,1)*100/hrdat(1),ipmod(:,1)*100];
fig3mxa  = [t,gnpmod(:,2)*100,hrmod(:,2)*100/hrdat(1),ipmod(:,2)*100];
fig3mxae = [t,gnpmod(:,3)*100,hrmod(:,3)*100/hrdat(1),ipmod(:,3)*100];

