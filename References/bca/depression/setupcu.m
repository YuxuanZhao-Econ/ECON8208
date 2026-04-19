%SETUPCU  sets up the state vector and transition matrix for the 
%         Markov chain used as an input in CAPUTIL.f90. The following
%         cases will be considered:
%
%          (1) The actual depression states for z,taul,taux, and g
%          (2) Only z varying
%          (3) Only taul varying
%          (4) Only taux varying
%          (5) All but taux varying
%          (6) All but z varying
%          (7) Only g varying
%          (8) No shocks
%
                                                                                
%         Ellen McGrattan, 5-24-03
%         Revised, ERM, 10-10-06


% Step 1. run ../mleannual/wedgescu (calling mle1cu) to get the matrices
%         P0,P,Q,z=[log(Zt),Tault,tauxt,log(exp(lgt))]
%
load mleoutcu;      
Case = 9;

%
% Step 2. set computep to 1 if transition matrix has not yet been computed;
%         set computep_ext to 1 if transitions have to be computed;
%         set checkdisc to 1 if the discrete approximation is to be checked.
%
computep     = 0;
computep_ext = 0;
checkdisc    = 0;

%
% Step 3. if computep=1, run tauch3D to generate MC transition
%
if computep;
  A   = P(1:3,1:3);
  Q   = chol(Q*Q')';
  Sig = Q(1:3,1:3)*Q(1:3,1:3)';
  N   = [7;8;8];
  grid0=[ ...
     -0.2500 -0.1000 -0.0500  0.0300  0.1000  0.1300  0.2000  0.2500   NaN;
     -0.3770 -0.0130  0.0650  0.1365  0.1690  0.2210  0.2730  0.3250  0.377;
     -0.3000 -0.2060 -0.1620 -0.1180 -0.0740 -0.0300  0.0500  0.1200  0.25];
  [yd,Py,piy,psumy]=tauch3D(A,Sig,N,grid0);
  save tauchencu A Sig N grid0 yd Py piy psumy
else
  load tauchencu
end;

%
% Step 4.  update tauxt to get a match with nonlinear simulation. 
%
yt  = z(:,1:3);
Gt  = exp(z(:,4));
mu  = (eye(3)-A)\P0(1:3);
yt(29:40,3) = [ ...
    0.2100
    0.2300
    0.2550
    0.2000
    0.0930
    0.0800
    0.1100
    0.1450
    0.1400
    0.1600
    0.1720
    0.1720];

%
% Step 5.  Add 11 additional states on the time series:
%
yt = [yt;yt(29:39,:)];
Gt = [Gt;Gt(29:39)];
if computep_ext==1;
  Pobs=tauch3D_ext(A,Sig,N,grid0,yt(41:51,:)-ones(11,1)*mu');
  save tauchenextcu Pobs
else
  load tauchenextcu
end

%
% Step 6. Find indices S of yd so that yd(S,i) looks like yobs(:,i).
%
nt             = length(yt(:,1));
yobs           = yt-ones(nt,1)*mu';
TN             = prod(N);
if checkdisc==1;
  ind            = ones(TN,3);
  ind(:,1)       = vec([1:N(1)]'*ones(1,N(2)*N(3)));
  ind(:,2)       = vec(vec(ones(N(1),1)*[1:N(2)])*ones(1,N(3)));
  ind(:,3)       = vec(ones(N(1)*N(2),1)*[1:N(3)]);
  SS             = zeros(nt,3);
  for i=1:3;
    bins         = [grid0(i,1:N(i))',grid0(i,2:N(i)+1)'];
    bins(1,1)    = -inf;
    bins(N(i),2) = inf;
    for j=1:nt;
      find( yobs(j,i)> bins(:,1) & yobs(j,i) <= bins(:,2));
      SS(j,i)    = find( yobs(j,i)> bins(:,1) & yobs(j,i) <= bins(:,2) );
    end;
  end;
  for i=1:nt;
    k = 1;
    for j=1:TN;
      if (ind(j,:)==SS(i,:)); k=j; end;
    end;
    S(i,1) = k;
  end;
end;

%
% Step 7. Add in additional states to state vector.
%
[lPy,j]= size(Py);
P      = [[zeros(11);Pobs'./(ones(lPy,1)*sum(Pobs'))],[zeros(11,lPy);Py']];
Y      = [yobs(41:51,:);yd]+ones(TN+11,1)*mu';
G      = [Gt(41:51);exp(z(29,4))*ones(TN,1)];
o      = ones(TN+11,1);
states = [G,0*o,Y(:,2),0*o,Y(:,3),exp(Y(:,1))];

%
% Step 8. For Cases 2-7, use 1929 values to set nonvarying wedges
%

if Case==2;
  states(:,1) = states(1,1)*o;
  states(:,3) = states(1,3)*o;
  states(:,5) = states(1,5)*o;
end;
if Case==3;
  states(:,1) = states(1,1)*o;
  states(:,5) = states(1,5)*o;
  states(:,6) = states(1,6)*o;
end;
if Case==4;
  states(:,1) = states(1,1)*o;
  states(:,3) = states(1,3)*o;
  states(:,6) = states(1,6)*o;
end;
if Case==5;
  states(:,5) = states(1,5)*o;
end;
if Case==6;
  states(:,6) = states(1,6)*o;
end;
if Case==7;
  states(:,3) = states(1,3)*o;
  states(:,5) = states(1,5)*o;
  states(:,6) = states(1,6)*o;
end;
if Case==9;
  states(:,1) = states(1,1)*o;
  states(:,3) = states(1,3)*o;
  states(:,5) = states(1,5)*o;
  states(:,6) = states(1,6)*o;
end;

%
% Step 9. Write out states and P(:) to a file and load it in caputil.inp
%
%save tempcu

