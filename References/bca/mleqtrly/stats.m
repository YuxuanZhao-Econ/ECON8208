function tab = stats(X,y,ilog,labels)
%STATS   This code computes summary statistics and cross correlations 
%        for raw series in matrix X and an output series in matrix y.  
%        tab = stats(X,y,ilog,labels) takes a Txn input matrix X and 
%        a Tx1 input matrix y and returns and prints a table of results.  
%        If ilog=1 the series are already logged (but not filtered).  
%        The default is ilog=0.  The fourth input is optional.  It is
%        a nxm matrix of characters with row i being the label to
%        use for variable i of matrix X.

%        Ellen McGrattan, 2-7-05
%        Revised, 2-9-05


[T,n]   = size(X);
if nargin<4;
  for i=1:n
    j           = length(int2str(i));
    labels(i,1:10+j) = ['Variable ',int2str(i),' '];
  end;
end;
if nargin<3;
  ilog = 0;
end;

%
% Remove trends:
%

y       = y(:);
if length(y)~=T;
  disp('Warning: X and y have to have the same length')
end;
dlX     = zeros(T,n);
if ilog==0;
  X     = log(X);
  y     = log(y);
end;
dly     = y - hptrend(y,1600);
for i=1:n;
  dlX(:,i) = X(:,i) - hptrend(X(:,i),1600);
end;

%
% Compute standard deviations relative to output
%

stds    = std(dlX)'/std(dly);

%
% Compute correlations
%

clag2   = corrcoef([dly(1:T-2),dlX(3:T,:)]);
clag1   = corrcoef([dly(1:T-1),dlX(2:T,:)]);
cc      = corrcoef([dly,dlX]);
clead1  = corrcoef([dly(2:T),dlX(1:T-1,:)]);
clead2  = corrcoef([dly(3:T),dlX(1:T-2,:)]);

tab     = [stds,[clead2(2:n+1,1),clead1(2:n+1,1),cc(2:n+1,1), ...
                 clag1(2:n+1,1), clag2(2:n+1,1)   ]];

for i=1:n;
  for j=i+1:n;
    clag2   = corrcoef([dlX(1:T-2,j),dlX(3:T,i)]);
    clag1   = corrcoef([dlX(1:T-1,j),dlX(2:T,i)]);
    cc      = corrcoef([dlX(:,j),dlX(:,i)]);
    clead1  = corrcoef([dlX(2:T,j),dlX(1:T-1,i)]);
    clead2  = corrcoef([dlX(3:T,j),dlX(1:T-2,i)]);
    tab     = [tab;
               NaN,[clead2(2,1),clead1(2,1),cc(2,1),clag1(2,1),clag2(2,1)]];
  end;
end;

disp('------------------------------------------------------------')
disp('A. Summary Statistics                                       ')
disp('------------------------------------------------------------')
disp('             Standard     Cross-Correlation with Y(t-k), k= ')
disp('             Deviation    ----------------------------------')
disp('Variable     Rel. to Y     -2     -1      0      1      2   ')
disp('------------------------------------------------------------')
for i=1:n;
  disp(sprintf([labels(i,:) , ...
       ' %5.2f        %5.2f  %5.2f  %5.2f  %5.2f  %5.2f'],tab(i,:)))
end;
disp('------------------------------------------------------------')
disp('B. Cross Correlations                                       ')
disp('------------------------------------------------------------')
disp('                          Cross-Correlation X(t), Z(t-k), k=')
disp('                          ----------------------------------')
disp('Variable X,Z               -2     -1      0      1      2   ')
disp('------------------------------------------------------------')
k = n+1;
for i=1:n;
  for j=i+1:n;
     disp(sprintf([labels(i,:),', ',labels(j,:), ...
     ' %5.2f  %5.2f  %5.2f  %5.2f  %5.2f'],tab(k,2:6)))
     k = k+1;
  end;
end;
disp('------------------------------------------------------------')
disp('Note: Statistics based on logged and HP-filtered series     ')

