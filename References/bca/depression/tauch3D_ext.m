function [P,S] = tauch3D(A,Sige,N,grid,yobs)
%TAUCH3D     finds the transition probabilities for a Markov chain intended
%            to mimic the properties of the autoregressive process for 
%            the 3x1 vector y:
%
%            (1)  y[t] = A y[t-1] + e[t],  e[t] ~ N(0,Sige), Sige not diagonal.
%
%            [y,P,pi,S,psum]=tauch3D(A,Sige,N,grid,yobs) computes 
%            the transition matrix P and stationary distribution pi from
%            equation (1) for an N*-state Markov-chain (N*=N(1)N(2)N(3)).
%            The ith value for the first variable in y is the midpoint
%            of [grid(1,i),grid(1,i+1)].  Similarly the ith value for the
%            second variable is the midpoint of [grid(2,i),grid(2,i+1)].
%            If the user has time series for y[t], namely the (nt x 3) 
%            matrix yobs, then the program computes the indices S for 
%            the Markov Chain that most closely mimic the observed 
%            series.  Output psum is a check on the domain sizes -- psum 
%            is the sum of rows in P before they are normalized to sum 
%            to 1.  If only three inputs are provided, then even spacing
%            is used for all variables with the first node at -3 times
%            the standard deviation of the variable and the last at +3
%            times the standard deviation.  With no yobs input, S=0.
%
%            See also VECTAUCH.m which assumes Sige is diagonal MxM
%            as in Tauchen's article.  See also ESTAUCH3D.m which
%            uses even spacing by default.

%            References: 
%             [1] Tauchen, G. ``Finite State Markov-chain approximations
%                 to Univariate and Vector Autoregressions,'' Economics 
%                 Letters 20, pp. 177-181, 1986.

%            Ellen McGrattan, 1-21-97
%            Revised, ERM, 5-14-03

[M,tem]        = size(A);
if nargin<4; m = 3; end;
if M~=3;
  error('TAUCH3D for cases with 3-dimensional system')
end;
N              = N(:);
Sigy           = dlyap(A,Sige);
sigy           = sqrt(diag(Sigy));
TN             = prod(N);
y              = zeros(3,max(N));
wlag           = y;
wlead          = y;
for i=1:3;
  if nargin<4;
    y(i,1:N(i))     = linspace(-3*sigy(i),3*sigy(i),N(i));
    wlag(i,1:N(i))  = y(i,N(i))/(N(i)-1);
    wlead(i,1:N(i)) = wlag(i,1:N(i));
  else
    y(i,1:N(i))     = .5*grid(i,1:N(i))+.5*grid(i,2:N(i)+1);
    wlag(i,1:N(i))  = y(i,1:N(i))-grid(i,1:N(i));
    wlead(i,1:N(i)) = grid(i,2:N(i)+1)-y(i,1:N(i));
  end
end;

%
% ind = indices of the TN discrete states
%
yd             = zeros(TN,3);
ind            = ones(TN,3);
ind(:,1)       = vec([1:N(1)]'*ones(1,N(2)*N(3)));
ind(:,2)       = vec(vec(ones(N(1),1)*[1:N(2)])*ones(1,N(3)));
ind(:,3)       = vec(ones(N(1)*N(2),1)*[1:N(3)]);
yd(:,1)        = y(1,ind(:,1))';
yd(:,2)        = y(2,ind(:,2))';
yd(:,3)        = y(3,ind(:,3))';
wm(:,1)        = wlag(1,ind(:,1))';
wm(:,2)        = wlag(2,ind(:,2))';
wm(:,3)        = wlag(3,ind(:,3))';
wp(:,1)        = wlead(1,ind(:,1))';
wp(:,2)        = wlead(2,ind(:,2))';
wp(:,3)        = wlead(3,ind(:,3))';

[nobs,jnk]     = size(yobs);
P              = zeros(nobs,TN);
for j=1:nobs;
  mu           = A*yobs(j,:)';
  for l=1:TN;
    a1         = yd(l,1)-mu(1)-wm(l,1);
    b1         = yd(l,1)-mu(1)+wp(l,1);
    a2         = yd(l,2)-mu(2)-wm(l,2);
    b2         = yd(l,2)-mu(2)+wp(l,2);
    a3         = yd(l,3)-mu(3)-wm(l,3);
    b3         = yd(l,3)-mu(3)+wp(l,3);
    [x1,wgt1]  = qgausl(a1,b1,20);
    [x2,wgt2]  = qgausl(a2,b2,20);
    [x3,wgt3]  = qgausl(a3,b3,20);
    sum1       = 0;
    for i1=1:20
      for i2=1:20
        for i3=1:20
          x      = [x1(i1);x2(i2);x3(i3)];
          sum1   = sum1+wgt1(i1)*wgt2(i2)*wgt3(i3)*exp(-.5*x'*inv(Sige)*x);
        end;
      end;
    end;
    P(j,l)     = sum1/(15.74960994572242*sqrt(det(Sige)));
  end;
end;

S              = 0;
if nargout>1;
  nt           = length(yobs);
  SS           = zeros(nt,3);
  for i=1:3;
    bins       = [grid(i,1:N(i))',grid(i,2:N(i)+1)'];
    bins(1,1)  = -inf;
    bins(N(i),2)= inf;
    for j=1:nt;
      find( yobs(j,i)> bins(:,1) & yobs(j,i) <= bins(:,2));
      SS(j,i)  = find( yobs(j,i)> bins(:,1) & yobs(j,i) <= bins(:,2) );
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


