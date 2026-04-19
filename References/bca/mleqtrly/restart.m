ii=i;
  for i=ii:B;
    %
    % Draw u's uniformly from sample {ut(1),ut(2)....ut(T)}
    % 
    Xt(1,:)   = X0';
    for j=1:T;
      k             = ceil(rand*T);  % draw number between 1 and T
      Ybar(j,:)     = Xt(j,:)*Cbar'+ut(k,:);
      Xt(j+1,:)     = Xt(j,:)*A'+ut(k,:)*K';
    end;
    for j=2:T+1;
      Y(j,:) = Y(j-1,:)*D'+Ybar(j-1,:);
    end;
    ZVAR      = exp(Y+log([(1+gz).^[0:T]',(1+gz).^[0:T]', ...
                             ones(T+1,1),(1+gz).^[0:T]']));
    Data(:,i) = ZVAR(:);
    theta     = uncmin(thet0,'mleq',0);
    if theta(21)<0;
      theta(21:24) = -theta(21:24);
    end;
    if theta(25)<0;
      theta(25:27) = -theta(25:27);
    end;
    if theta(28)<0;
      theta(28:29) = -theta(28:29);
    end;
    if theta(30)<0;
      theta(30)    = -theta(30);
    end;
    Theta(i,:) = theta';
  end;
  se  = std(Theta)';

%
% Print results 
%

disp('Results from Maximum Likelihood Estimation')
disp('------------------------------------------')
disp(' ')
disp('  [Theta, Standard Errors] ')
disp(sprintf(' %10.3e %10.3e\n', [param(ind),se]'))
disp(' ')
fprintf('  L(Theta) = %g ',L)
disp(' ')
                                                                                
save runmlenew

