%BENCHPLOTS   Plot summaries of the benchmark results. See details about
%             model economy in the Fortran code.

%             Ellen McGrattan, 12-01-01
%             Revised, ERM, 10-20-06

load bench.dat.case1
s      = bench(:,1);
z      = bench(:,14);
taul   = bench(:,10);
taux   = bench(:,13);
g      = bench(:,9);
t0     = 1929;
t1     = t0-1+length(s);
t      = [t0:t1]';
wedges = [(z/z(1)).^(.65),(1-taul)/(1-taul(1)),(1+taux(1))* ...
           ones(length(t),1)./(1+taux)];
hr     = bench(:,4);
ip     = bench(:,5);
gnp    = bench(:,6);
load bench.dat.case2
hrz    = bench(:,4);
ipz    = bench(:,5);
gnpz   = bench(:,6);
load bench.dat.case3
hrl    = bench(:,4);
ipl    = bench(:,5);
gnpl   = bench(:,6);
load bench.dat.case4
hrx    = bench(:,4);
ipx    = bench(:,5);
gnpx   = bench(:,6);
load bench.dat.case5
hrnox   = bench(:,4);
ipnox   = bench(:,5);
gnpnox  = bench(:,6);
cpnox   = bench(:,3);
load bench.dat.case6
hrnoz   = bench(:,4);
ipnoz   = bench(:,5);
gnpnoz  = bench(:,6);
load bench.dat.case7
hrg    = bench(:,4);
ipg    = bench(:,5);
gnpg   = bench(:,6);
load bench.dat.case9
hr0    = bench(:,4);
ip0    = bench(:,5);
gnp0   = bench(:,6);
load ../mleannual/uszvar1.dat
grate  = 1.016.^([0:99]');
nipal  = uszvar1./[ones(100,1),grate,grate,ones(100,1),grate,grate];
nipa   = nipal(29:100,:);
gnpdat = nipa(1:30,2);
ipdat  = nipa(1:30,3);
hrdat  = nipa(1:30,4);
gdat   = nipa(1:30,5);
cpdat  = gnpdat-ipdat-gdat;

gnpchk = [gnpdat,(gnpz-gnp0-gnpz(1)+gnp0(1))+ ...
                 (gnpl-gnp0-gnpl(1)+gnp0(1))+ ...
                 (gnpx-gnp0-gnpx(1)+gnp0(1))+ ...
                 (gnpg-gnp0-gnpg(1)+gnp0(1))+ ...
                  gnp0-gnp0(1)+ ...
                  gnpdat(1)];

ipchk  = [ipdat,(ipz-ip0-ipz(1)+ip0(1))+ ...
                (ipl-ip0-ipl(1)+ip0(1))+ ...
                (ipx-ip0-ipx(1)+ip0(1))+ ...
                (ipg-ip0-ipg(1)+ip0(1))+ ...
                 ip0-ip0(1)+ ...
                 ipdat(1)];
hrchk  = [hrdat,(hrz-hr0-hrz(1)+hr0(1))+ ...
                (hrl-hr0-hrl(1)+hr0(1))+ ...
                (hrx-hr0-hrx(1)+hr0(1))+ ...
                (hrg-hr0-hrg(1)+hr0(1))+ ...
                 hr0-hr0(1)+ ...
                 hrdat(1)];

figure(1)
  plot(t,gnpdat*100,t,wedges*100)
  axis([t0,t0+10,60,120])
title('Wedges and US GNP')
legend('Data','Efficiency wedge','Labor wedge','Investment wedge')

figure(2)
  gnpmod = [(gnpz-gnp0-gnpz(1)+gnp0(1)), ...
            (gnpl-gnp0-gnpl(1)+gnp0(1)), ...
            (gnpx-gnp0-gnpx(1)+gnp0(1)), ...
            (gnpg-gnp0-gnpg(1)+gnp0(1))]+gnpdat(1);

  plot(t,gnpdat*100,t,gnpmod*100)
  title('GNP')
  legend('Data','z only','tau_l only','tau_x only','g only')
  axis([1929,1939,60,120])

figure(3)
  hrmod  = [(hrz-hr0-hrz(1)+hr0(1)), ...
            (hrl-hr0-hrl(1)+hr0(1)), ...
            (hrx-hr0-hrx(1)+hr0(1)), ...
            (hrg-hr0-hrg(1)+hr0(1))]+hrdat(1);
  plot(t,hrdat*100/hrdat(1),t,hrmod*100/hrdat(1))
  title('Hours')
  legend('Data','z only','tau_l only','tau_x only','g only')
  axis([1929,1939,60,120])

figure(4)
  ipmod  = [(ipz-ip0-ipz(1)+ip0(1)), ...
            (ipl-ip0-ipl(1)+ip0(1)), ...
            (ipx-ip0-ipx(1)+ip0(1)), ...
            (ipg-ip0-ipg(1)+ip0(1))]+ipdat(1);

  plot(t,ipdat*100,t,ipmod*100)
  title('Investment')
  legend('Data','z only','tau_l only','tau_x only','g only')
  axis([1929,1939,0,45])

figure(5)
  gnpmnox = gnpnox-gnpnox(1)+gnpdat(1);
  gnpmnoz = gnpnoz-gnpnoz(1)+gnpdat(1);
  plot(t,gnpdat*100,t,gnpmnoz*100,t,gnpmnox*100)
  title('GNP, All Wedges But One')
  legend('Data','No z','No tau_x')
  axis([1929,1939,50,100])

figure(6)
  hrmnox = hrnox-hrnox(1)+hrdat(1);
  hrmnoz = hrnoz-hrnoz(1)+hrdat(1);
  plot(t,hrdat*100/hrdat(1),t,hrmnoz*100/hrdat(1),t,hrmnox*100/hrdat(1))
  title('Hours, All Wedges But One')
  legend('Data','No z','No tau_x')
  axis([1929,1939,50,100])

figure(7)
  ipmnox = ipnox-ipnox(1)+ipdat(1);
  ipmnoz = ipnoz-ipnoz(1)+ipdat(1);
  plot(t,ipdat*100,t,ipmnoz*100,t,ipmnox*100)
  title('Investment, All Wedges But One')
  legend('Data','No z','No tau_x')
  axis([1929,1939,0,45])

  
  cpmnox  = cpnox-cpnox(1)+cpdat(1);

fig1     = [t,gnpdat*100,wedges*100,g/g(1)*100];
fig2_4d  = [t,gnpdat*100,hrdat*100/hrdat(1),ipdat*100];
fig2mz   = [t,gnpmod(:,1)*100,hrmod(:,1)*100/hrdat(1),ipmod(:,1)*100];
fig2ml   = [t,gnpmod(:,2)*100,hrmod(:,2)*100/hrdat(1),ipmod(:,2)*100];
fig3mx   = [t,gnpmod(:,3)*100,hrmod(:,3)*100/hrdat(1),ipmod(:,3)*100];
fig3mg   = [t,gnpmod(:,4)*100,hrmod(:,4)*100/hrdat(1),ipmod(:,4)*100];
fig4mnox = [t,gnpmnox*100,hrmnox*100/hrdat(1),ipmnox*100];
fig4mnoz = [t,gnpmnoz*100,hrmnoz*100/hrdat(1),ipmnoz*100];
