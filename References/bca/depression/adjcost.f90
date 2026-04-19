program adjcost
!
!  DESCRIPTION:  This program computes an equilibrium for the growth 
!                model with variation in taxes, government consumption,
!                and productivity.  Capital adjustment costs are included. 
!                See the appendix and comments below for details.
!
!  MODEL         Households solve:  
!  ECONOMY:      
!                  max       E sum_t beta^t  U(c[t],1-l[t]) N[t]
!                 {c,x,l}                      
!                                            
!                  subject to
!
!                      (1+tau_c[t])c[t] + (1+tau_x[t])x[t] 
!                                  <= (1-tau_k[t]) r[t] k[t] 
!                                    +(1-tau_l[t]) w[t] l[t] 
!                                    + tau_k[t]delta k[t] + T[t]
!                      k[t+1] = [(1-delta) k[t] + i[t]- adj.costs]/(1+gn)
!
!
!                Firms solve the static problem:
!
!                  max       F(K,Z L) - r K - w L 
!                  K,L 
!
!                Government's period t budget constraint:
!                    
!                    G[t] + T[t] = tau_c[t] C[t] + tau_x[t] X[t]
!                                 +tau_k[t] (r[t]-delta) K[t] 
!                                 +tau_l[t] w[t] L[t] 
!
!                Resource Constraint:
!
!                    C[t]+X[t]+G[t] = F(K[t],Z[t]L[t])
!
!                where capital letters denote aggregates, i.e., C[t]=N[t]c[t]
!
!                Processes for exogenous states:
!                    
!                  s is a Markov chain with transition matrix pi
!                  s is an index in the set {1,...,ns}
!
!                detrended G, tau_k, tau_l, tauc, taux, and z are 
!                functions of s.
!
!  SOLUTION      Let x = capital, i = index of exogenous states. 
!                Represent c as a weighted sum of bilinear basis functions:
!
!                c(x,i)   = sum_k alpha_k^i N_a(x)   
!
!                N_a(x)   = (x  -x_{a-1})       on [x_{a-1},x_a]
!                           ------------- 
!                           (x_a-x_{a-1}) 
!
!                         = (x_{a+1}-x  )       on [x_a,x_{a+1}]
!                           ------------- 
!                           (x_{a+1}-x_a) 
!
!                         = 0,  elsewhere
!
!                Plug approximate c(x,i) into Euler equation for capital
!                and set it equal to 0.  Assume that this equation can be
!                represented as R(x,i;alpha) = 0.  Then the approximate
!                solution computed here is the function c given above with
!                alpha's chosen to satisfy the integral equations:
!
!                  int int R(x,i;alpha) N_a(x) dx dy = 0,  all i, all a
!
!  FUNCTIONAL    u(c+gamma*g,1-l)  = ([c+gamma*g]*(1-l)^psi)^phi/phi
!  FORMS:        F(k,l)    = k^theta * l^(1-theta)
!                penalty   = zeta * min(x,0)^3
!                adj.cost  = a/2 *(x/k-b)^2
!
!  TEST CASE:    psi = 0, delta = 1, g=0, a=1
!                ===>   c = (1-beta*theta)* k^theta
!
!  INPUT FILE:   adjcost.inp assumes the following order for inputs:
!
!                    a         : adjustment cost parameter
!                    b         : adjustment cost parameter
!                    beta      : discount factor 
!                    delta     : depreciation 
!                    gamma     : weight on govt. spending in utility
!                    gn        : growth of population 
!                    gz        : growth of technology
!                    psi       : preference parameter
!                    theta     : technology parameter      
!                    zeta      : penalty function parameter 
!                    ns        : number of exogenous states
!                    g         : government spending (1 x ns)
!                    tauk      : tax rate on capital (1 x ns)
!                    taul      : tax rate on labor (1 x ns)
!                    tauc      : tax rate on consumption (1 x ns)
!                    taux      : tax rate on investment (1 x ns)
!                    z         : level of technology (1 x ns)
!                    pi'       : transpose transition matrix (ns x ns) 
!                                (columns sum to 1)
!                    k         : grid on capital (nx x 1)
!                    init      : 1=initial guess of c included, 0 otherwise 
!                    c         : if init=1, guess for c (nx x ns)
!                    icomp     : 1=compute equilibrium, 0 otherwise
!                    isimul    : 1=simulate time series, 0 otherwise
!                    nsim      : number of simulations
!                    nt        : number of time periods per simulation
!                    idraw     : 1=draw random realizations, 0 otherwise
!                    iseed     : integer seed for random number generator
!                    x0,ind    : if idraw=0, initial x indices for
!                                exogenous states (1 x 2+nt)
!             
!  Ellen McGrattan, 9-1-99
!  Revised, ERM, 12-12-01


   implicit none
   integer, parameter           :: nx=35,ns=459,na=16065,m=3,            &
                                   msim=100,mt=2000,nm=40,ldr=50000000,  &
                                   maxit=100
   real, parameter              :: crit= 0.0000000001
   real, dimension (nx-1,m)     :: ax,wx
   real, dimension (nx)         :: xa
   real, dimension (ns)         :: g,tauk,taul,tauc,taux
   real, dimension (ns)         :: z
   real, dimension (ns,ns)      :: pi = 0.0
   real, dimension (nx,ns)      :: c0,c1
   real, dimension (na,1)       :: r,sol,wrk
   real, dimension (ldr,1)      :: dr,alu
   real, dimension (na,nm+1)    :: vv
   real, dimension (msim)       :: x0
   real                         :: beta,delta,gamma,gn,gz,omega,psi,phi,  &
                                   theta,delt1,thet1,grate,               &
                                   betg,tlc1,tlc2,tc1,tc2,tx1,tx2,tk2,    &
                                   x,v,x1,x2,cp,c,hr,hr0,                 &
                                   lei,u,u1,u11,u2,u12,u22,f,f1,f2,f22,   &
                                   dhr,ip,xt,yt,x1t,x2t,thri,tipi,txti,   &
                                   basis1,basis2,basis1t,basis2t,         &
                                   dbasis1,dbasis2,                       &
                                   cpt,ct,hrt,ut,u1t,u11t,u2t,u12t,u22t,  &
                                   ft,f1t,f2t,f11t,f12t,f22t,ipt,         & 
                                   tcptxt,tcpti,thrti,tipti,tcptj,thrtj,  &
                                   tiptj,sum1,tlai,tsai,tsaj,res,dres,    &
                                   uni,ustart,useed,zeta,pen,pent,eps,    &
                                   a,b,adj,dadj,adjt,dadjt,d2adj,d2adjt
   integer, dimension (na,1)    :: iwrk,ju
   integer, dimension (na+1,1)  :: ipr
   integer, dimension (ldr,1)   :: ir,jr,indr,jlu
   integer, dimension (msim,mt) :: ind
   integer                      :: init,info,ix,ixt,i,j,k,l,n,ne,k1,k2,   &
                                   l1,l2,icomp,isimul,idraw,nsim,nt,      &
                                   iseed,nnz,ierr,im,iout
                       
!
!
!  Parameters:
!
   open(unit=5,  file='adjcost.inp')
   open(unit=7,  file='adjcost.nxt')
   open(unit=8,  file='adjcost.dat')
   open(unit=9,  file='acfunc.dat')

   read(5,*) a
   read(5,*) b
   read(5,*) beta
   read(5,*) delta
   read(5,*) gamma
   read(5,*) gn
   read(5,*) gz
   read(5,*) psi
   read(5,*) phi
   read(5,*) theta
   read(5,*) zeta
   read(5,*) i
   if (i /= ns) then
     write(*,*) 'ERROR: wrong number of exogenous states. Edit adjcost.f90 '
     write(*,*) '       and change parameter ns.'
     stop
   endif
   do i=1,ns
     read(5,*) g(i),tauk(i),taul(i),tauc(i),taux(i),z(i)
   end do
   do i=1,ns
     do j=1,ns
       read(5,*) pi(i,j)
     end do
   end do

   read(5,*) xa
   read(5,*) init

   if (init == 1) then
     do j=1,ns 
       do i=1,nx
         read(5,*) c0(i,j)
       end do 
     end do
   else
     do j=1,ns
       c0(:,j)   = (1.0-beta*theta)*xa**theta*z(j)**thet1
     end do
   endif

   read(5,*) icomp
   read(5,*) isimul

   if (isimul == 1) then
     read(5,*) nsim
     if (nsim > msim) then
       write(*,*) 'ERROR: Parameter governing maximum simulations is too '
       write(*,*) '       small. Edit file and increase parameter msim.'
       stop
     endif
     read(5,*) nt
     if (nt > mt) then
       write(*,*) 'ERROR: Parameter governing maximum simulation periods is'
       write(*,*) '       too small. Edit file and increase parameter mt.'
       stop
     endif
     read(5,*) idraw
     read(5,*) iseed
     if (idraw == 0) then
       do i=1,nsim
         read(5,*) x0(i),(ind(i,j),j=1,nt)
       end do
     endif
   endif

!
!  Intermediate parameters:
!

   betg    = beta*(1.0+gz)**(phi-1.)
   delt1   = 1.0-delta 
   grate   = (1.0+gn)*(1.0+gz)
   thet1   = 1.0-theta
   hr0     = 0.30

!
!  Quadarature abscissas and weights:
!
   do i=1,nx-1
     call qgausl (m,xa(i),xa(i+1),ax(i,:),wx(i,:))
   end do

!
!  Compute equilibrium
!
   if (icomp == 1) then

!
!  Solve the fixed point problem: int int R(x,y,i;alpha) dx dy = 0 
!  for alpha using a Newton method.
!

   newton: do

!
!    Derive residuals (r) of the first order conditions and their
!    derivatives (dr) with respect to the unknowns in alpha.
!

     r    = 0.0
     dr   = 0.0
     nnz  = 0
     ne   = nx-1

     do k=1,ns

       tlc1   = (1.0-taul(k))/(1+tauc(k))  
       tc1    = 1.0+tauc(k)  
       tx1    = 1.0+taux(k)  

       do n=1,ne

!
!        Element n is the interval [xa(n),xa(n+1)].
!

         x1   = xa(n)
         x2   = xa(n+1)

!
!        Compute r at all quadrature points on the element.
!
         do i=1,m
           x         = ax(n,i)
           v         = wx(n,i)
           basis1    = (x2-x)/(x2-x1)
           basis2    = (x-x1)/(x2-x1)

!
!          Compute consumption using finite element approximation.
!
           cp        = c0(n,k)*basis1 + c0(n+1,k)*basis2
           c         = cp+gamma*g(k)

!
!          Back out hours of work.
!

           hr        = hr0
           inner_newton_1: do

             lei     = 1.0-hr
             u       = (c*lei**psi)**phi/phi    
             u1      = phi*u/c
             u2      = psi*phi*u/lei
             u11     = (phi-1.)*u1/c
             u12     = psi*phi*u1/lei
             u22     = (psi*phi-1.)*u2/lei
             f       = x**theta*(z(k)*hr)**thet1
             f2      = thet1*f/hr
             f22     =-theta*f2/hr
             dhr     = (u2-tlc1*f2*u1)/(-u22-tlc1*f22*u1+tlc1*f2*u12) 

             if (abs(dhr) <= crit) exit inner_newton_1 

             hr      = hr - dhr

           end do inner_newton_1

!
!          Back out investment.
!

           ip        = f-cp-g(k)
           pen       = min(ip,0.0)

!
!          Update capital stocks and find tomorrow's state on the grid.
!

           adj       = 0.5*a*(ip/x-b)*(ip/x-b)
           dadj      = a*(ip/x-b)
           d2adj     = a
           xt        = (delt1*x+ip-adj*x)/grate

           ixt       = 1
           do l = 1,nx-1
              if (xt > xa(l)) ixt   = l
           end do
           x1t       = xa(ixt)
           x2t       = xa(ixt+1)

           basis1t   = (x2t-xt)/(x2t-x1t)
           basis2t   = (xt-x1t)/(x2t-x1t)
           dbasis1   =-1.0/(x2t-x1t)
           dbasis2   =-dbasis1

!
!          Compute derivatives of hours and capital to be used later.
!

           thri      = (u12-u11*f2*tlc1)/(u22-(u12*f2-u1*f22)*tlc1)
           tipi      = f2*thri-1.0
           txti      = (1.-dadj)*tipi/grate

!
!          Initialize work vector.
!

           wrk       = 0.0
           sum1      = 0.0
           tsai      = 0.0

!
!          Compute beta sum_j pi(i,j){(1-tauk')*F_1(x',y',l')+tauk'*delta
!                                     +(1-taux')*(1-delta)}/[c'*(1+tauc')]
!

           do l=1,ns

             tlc2      = (1.0-taul(l))/(1+tauc(l))
             tk2       = 1.0-tauk(l)
             tx2       = 1.0+taux(l)
             tc2       = 1.0+tauc(l)
!
!            Next period consumption using finite element approximation.
!
             cpt       = c0(ixt,l)*basis1t + c0(ixt+1,l)*basis2t
             ct        = cpt+gamma*g(l)
               
!
!            Back out next period hours of work.
!
             hrt       = hr0
             inner_newton_2: do

               lei     = 1.0-hrt
               ut      = (ct*lei**psi)**phi/phi    
               u1t     = phi*ut/ct
               u2t     = psi*phi*ut/lei
               u11t    = (phi-1.)*u1t/ct
               u12t    = psi*phi*u1t/lei
               u22t    = (psi*phi-1.)*u2t/lei
               ft      = xt**theta*(z(l)*hrt)**thet1
               f2t     = thet1*ft/hrt
               f22t    =-theta*f2t/hrt
               dhr     = (u2t-tlc2*f2t*u1t)/(-u22t-tlc2*f22t*u1t+tlc2*f2t*u12t) 

               if (abs(dhr) <= crit) exit inner_newton_2 

               hrt     = hrt - dhr

             end do inner_newton_2

!
!            Back out next period investment.
!

             ipt       = ft-cpt-g(l)


!
!            Additional derivatives of utility and production needed.
!

             f1t       = ft*theta/xt
             f11t      =-ft*theta*thet1/xt/xt
             f12t      = thet1*f1t/hrt

!
!            Derivatives of variables with respect to coefficients
!            in finite element approximation needed.
!
             tcptxt    = c0(ixt,l)*dbasis1 + c0(ixt+1,l)*dbasis2
             tcpti     = tcptxt*txti
             thrti     = ((u12t-u11t*f2t*tlc2)*tcpti-u1t*f12t*tlc2*txti)/  &
                          (u22t-(u12t*f2t-u1t*f22t)*tlc2)
             tipti     = f1t*txti+f2t*thrti-tcpti
             tcptj     = 1.0
             thrtj     = (u12t-u11t*f2t*tlc2)*tcptj/                       &
                         (u22t-(u12t*f2t-u1t*f22t)*tlc2)
             tiptj     = f2t*thrtj-tcptj

!
!            Add term to sum in Euler equation for capital.
!
            
             adjt      = 0.5*a*(ipt/xt-b)*(ipt/xt-b)
             dadjt     = a*(ipt/xt-b)
             d2adjt    = a
             pent      = min(ipt,0.0)
             sum1      = sum1 +betg*pi(k,l) * ((tk2*f1t+tauk(l)*delta)*    &
                         u1t/tc2+(u1t*tx2/tc2-zeta*pent*pent)*             &
                         (delt1-adjt+dadjt*ipt/xt)/(1.-dadjt))

!
!            Compute derivatives of sum1 with respect to the 
!            coefficients in the finite element approximation.
!

             tsai      = tsai +  &
                         betg*pi(k,l)*((u11t*tcpti-u12t*thrti)/tc2*        &
                         (tk2*f1t+tauk(l)*delta)+tk2*(f12t*thrti+f11t*     &
                         txti)*u1t/tc2+((u11t*tcpti-u12t*thrti)*tx2/tc2-   &
                         2.0*zeta*pent*tipti)*(delt1-adjt+dadjt*ipt/xt)/   &
                         (1.-dadjt)+(u1t*tx2/tc2-zeta*pent*pent)*          &
                         d2adjt*ipt*(tipti-txti*ipt/xt)/(xt*xt*            &
                         (1.-dadjt))+(u1t*tx2/tc2-zeta*pent*pent)*         &
                         (delt1-adjt+dadjt*ipt/xt)*d2adjt*(tipti-          &
                         txti*ipt/xt)/(xt*(1.-dadjt)*(1.-dadjt)))

             tsaj      = betg*pi(k,l)*((u11t*tcptj-u12t*thrtj)/tc2*        &
                         (tk2*f1t+tauk(l)*delta)+tk2*f12t*thrtj*           &
                         u1t/tc2+((u11t*tcptj-u12t*thrtj)*tx2/tc2-         &
                         2.0*zeta*pent*tiptj)*(delt1-adjt+dadjt*ipt/xt)/   &
                         (1.-dadjt)+(u1t*tx2/tc2-zeta*pent*pent)*          &
                         d2adjt*ipt*tiptj/(xt*xt*(1.-dadjt))+              &
                         (u1t*tx2/tc2-zeta*pent*pent)*(delt1-adjt+         &
                         dadjt*ipt/xt)*d2adjt*tiptj/(xt*(1.-dadjt)*        &
                         (1.-dadjt)))


!
!            Hold results in work vectors for later.
!
             l1        = ixt+(l-1)*nx
             l2        = ixt+1+(l-1)*nx
             wrk(l1,1) = wrk(l1,1) + tsaj * basis1t
             wrk(l2,1) = wrk(l2,1) + tsaj * basis2t

           end do
!
!          The vector `wrk' contains derivatives of the residual.
!
           k1        = n+(k-1)*nx
           k2        = n+1+(k-1)*nx

           dres      = tsai - ((u11-u12*thri)*tx1/tc1-2.0*zeta*pen*tipi)/  &
                       (1.-dadj)-(u1*tx1/tc1-zeta*pen*pen)*d2adj*tipi/     &
                       (x*(1.-dadj)*(1.-dadj))
           wrk(k1,1) = wrk(k1,1) + dres * basis1 
           wrk(k2,1) = wrk(k2,1) + dres * basis2
!
!          Add the residual x quadrature weights x basis functions to r(.).
!
           res       = sum1 - (u1*tx1/tc1 - zeta*pen*pen)/(1.-dadj)
           r(k1,1)   = r(k1,1) + res * v * basis1
           r(k2,1)   = r(k2,1) + res * v * basis2

!
!          Add derivatives of the residual x quadrature weights x basis
!          functions to dr(.).
!
           do l=1,na
             if (abs(wrk(l,1)) > 0.0) then
               nnz         = nnz + 1
               indr(nnz,1) = (k1-1)*na+l
               dr(nnz,1)   = wrk(l,1) * v * basis1
               nnz         = nnz + 1
               indr(nnz,1) = (k2-1)*na+l
               dr(nnz,1)   = wrk(l,1) * v * basis2
             endif
           end do
         end do
       end do
     end do

!
!    Sort elements of dr and combine any with same index.
!

     call qcksrt3(nnz,indr,dr)
     i       = 1
     j       = indr(1,1)
     do k=2,nnz
       if (indr(k,1) == j) then
         dr(i,1) = dr(i,1)+dr(k,1)
       else
         i         = i+1
         j         = indr(k,1)
         indr(i,1) = indr(k,1)
         dr(i,1)   = dr(k,1)
       endif
     end do
     nnz     = i

!
!    Convert (j-1)*na+i stored in indr(.) into i and j.
!
     do i=1,nnz
       jr(i,1)   = mod(indr(i,1)-1,na)+1
       ir(i,1)   = (indr(i,1)-1)/na+1
     end do

!
!    Use ipr(.) to determine pointers to rows in sparse matrix.
!
     ipr(1,1) = 1
     j        = 1
     do i=2,nnz
       if (ir(i,1) /= ir(i-1,1)) then
         j         = j+1
         ipr(j,1)  = i
       endif
     end do
     ipr(na+1,1)   = nnz+1

!
!    Invert the Jacobian matrix dr with the SPARSKIT iterative solver
!
     call ilu0(na,dr,jr,ipr,alu,jlu,ju,iwrk,ierr)
     if (ierr /= 0) then
       write(*,*) 'ERROR: ierr not equal 0 in ilu0. Program stopped.'
       stop
     endif
     eps      = 1.0e-10
     im       = nm
     iout     = 0
     call pgmres(na,im,r,sol,vv,eps,maxit,iout,dr,jr,ipr,alu,jlu,ju,ierr)
     if (ierr == 1) write(*,*) 'WARNING: max iterations in pgmres in ADJCOST.F'
!
!    Do Newton update.
!
     do k=1,ns
       c1(:,k)  = c0(:,k) - sol((k-1)*nx+1:k*nx,1)
     end do
     sum1     = dot_product(sol(:,1),sol(:,1))
     sum1     = sqrt(sum1)/float(na)

     write(*,*) 'Residual = ',sum1

!
!    Check to see if the solution is converged.   If so, stop.
!
     if (sum1 < crit) exit newton
     c0       = c1

   end do newton
   end if

!
!  Simulate time series.  
!
   if (isimul == 1) then

     if (idraw == 1) then
!
!      Pick seed for random number generator
!
       useed = ustart(iseed)

       do j=1,nsim
!
!        Randomly pick a number between 1 and ns
!         
         v        = float(ns)*uni()
         do l=ns,1,-1
           if (v <= float(l)) k = l
         end do
         ind(j,1) = k
!
!        Let capital stocks be steady states associated with the 
!        initial exogenous state.  
!         

         xt       = xa(nx/2) 
         tlc1     = (1.0-taul(k))/(1.0+tauc(k))
         tc1      = 1.0+tauc(k)

         do i=1,20

           x        = xt
           ix       = 1
           do l = 1,nx-1
             if (x > xa(l)) ix = l
           end do
           x1       = xa(ix)
           x2       = xa(ix+1)
           basis1   = (x2-x)/(x2-x1)
           basis2   = (x-x1)/(x2-x1)

           cp       = c0(ix,k)*basis1 + c0(ix+1,k)*basis2 
           c        = cp+gamma*g(k)

           hr       = hr0
           inner_newton_3: do

             lei    = 1.0-hr
             u      = (c*lei**psi)**phi/phi    
             u1     = phi*u/c
             u2     = psi*phi*u/lei
             u11    = (phi-1.)*u1/c
             u12    = psi*phi*u1/lei
             u22    = (psi*phi-1.)*u2/lei
             f      = x**theta*(z(k)*hr)**thet1
             f2     = thet1*f/hr
             f22    =-theta*f2/hr
             dhr    = (u2-tlc1*f2*u1)/(-u22-tlc1*f22*u1+tlc1*f2*u12) 

             if (abs(dhr) <= crit) exit inner_newton_3

             hr     = hr - dhr

           end do inner_newton_3

           ip       = f-cp-g(k)
           adj      = 0.5*a*(ip/x-b)*(ip/x-b)
           xt       = (delt1*x+ip-adj*x)/grate

         end do
         x0(j)      = xt

         do i=2,nt
           v        = uni()
           sum1     = 1.0
           do l=ns,1,-1
             if (v <= sum1)  k = l
             sum1   = sum1 - pi(ind(j,i-1),l)
           end do
           ind(j,i) = k  
         end do
       end do

     endif

     do j=1,nsim

       xt      = x0(j)

       do i=1,nt
!
!        Initialize states and basis functions.
!

         x        = xt
         k        = ind(j,i)
         tlc1     = (1.0-taul(k))/(1.0+tauc(k))
         tc1      = 1.0+tauc(k)

         ix       = 1
         do l = 1,nx-1
           if (x > xa(l)) ix = l
         end do
         x1       = xa(ix)
         x2       = xa(ix+1)
         basis1   = (x2-x)/(x2-x1)
         basis2   = (x-x1)/(x2-x1)

!
!        Compute consumption using finite element approximation.
!
         cp       = c0(ix,k)*basis1 + c0(ix+1,k)*basis2 
         c        = cp+gamma*g(k)
!
!        Back out hours of work.
!
         hr       = hr0
     
         inner_newton_4: do

           lei    = 1.0-hr
           u      = (c*lei**psi)**phi/phi    
           u1     = phi*u/c
           u2     = psi*phi*u/lei
           u11    = (phi-1.)*u1/c
           u12    = psi*phi*u1/lei
           u22    = (psi*phi-1.)*u2/lei
           f      = x**theta*(z(k)*hr)**thet1
           f2     = thet1*f/hr
           f22    =-theta*f2/hr
           dhr    = (u2-tlc1*f2*u1)/(-u22-tlc1*f22*u1+tlc1*f2*u12) 

           if (abs(dhr) <= crit) exit inner_newton_4

           hr     = hr - dhr

         end do inner_newton_4

!
!        Back out investment.
!

         ip      = f-cp-g(k)

!
!        Update capital stocks.
!
 
         adj     = 0.5*a*(ip/x-b)*(ip/x-b)
         xt      = (delt1*x+ip-adj*x)/grate

!
!        Print results for period i, simulation j.
!
         write(8,'(1X,I2,1X,19F11.4)') k,x,cp,hr,ip,f,f2,f1, &
            g(k),taul(k),tauk(k),tauc(k),taux(k),z(k)

       end do
       write(8,*) 
     end do

   endif

!
!  Write out new input file with results.
!
   write(7,'(1X,F7.4,12X,''a      >= 0          '')') a
   write(7,'(1X,F7.4,12X,''b      in [0,1]      '')') b
   write(7,'(1X,F7.4,12X,''beta   in [0,1]      '')') beta
   write(7,'(1X,F7.4,12X,''delta  in [0,1]      '')') delta
   write(7,'(1X,F7.4,12X,''gamma  in [0,1]      '')') gamma
   write(7,'(1X,F7.4,12X,''gn                   '')') gn
   write(7,'(1X,F7.4,12X,''gz                   '')') gz
   write(7,'(1X,F7.4,12X,''psi    >= 0          '')') psi
   write(7,'(1X,F7.4,12X,''phi    <= 0          '')') phi
   write(7,'(1X,F7.4,12X,''theta  in [0,1]      '')') theta
   write(7,'(1X,E7.1,12X,''zeta   >= 0          '')') zeta
   write(7,*)
   write(7,  &
   '(1X,I3,16X,''No. of indices in g,tauk,taul,tauc,taux,z,pi below'')') ns
   write(7,*)
   do i=1,ns
     write(7,'(6(1X,F7.4))') g(i),tauk(i),taul(i),tauc(i),taux(i),z(i)
   enddo
   write(7,*)
   do i=1,ns
     do j=1,ns
       write(7,'(1X,F7.4)') pi(i,j)
     end do
   end do
   write(7,*) 
   do i=1,nx-1
     write(7,'(1X,F8.5)') xa(i) 
   end do
   write(7,'(1X,F8.5,11X,''grid on capital'')') xa(nx)
   write(7,*) 
   write(7,'(2X,I1,17X,''Input initial consumption? (1=yes,0=no)'')') init
   write(7,*) 
   do j=1,ns 
     do i=1,nx
        write(7,'(1X,F8.5)') c0(i,j)
     end do 
   end do
   write(7,*) 
   write(7,'(1X,I6,13X,''Compute equilibrium?  (1=yes,0=no)'')') icomp
   write(7,'(1X,I6,13X,''Simulate time series? (1=yes,0=no)'')') isimul

   if (isimul == 1) then
     write(7,'(1X,I6,13X,''Number of simulations'')') nsim
     write(7,'(1X,I6,13X,''Number of time periods per simulation'')') nt
     write(7,'(1X,I6,13X,''Draw random realizations? (1=yes,0=no)'')') idraw
     write(7,'(1X,I6,13X,''Seed for random numbers?  (integer)'')') iseed
     if (idraw == 0) then
       do i=1,nsim
         write(7,*)
         write(7,'(1X,F8.5,70(1X,I2))') x0(i),(ind(i,j),j=1,nt)
       end do
     endif
   endif

!
!  Write out parameters and decision functions for plotting later.
!
   write(9,'(1X,F7.4)') a
   write(9,'(1X,F7.4)') b
   write(9,'(1X,F7.4)') beta
   write(9,'(1X,F7.4)') delta
   write(9,'(1X,F7.4)') gamma
   write(9,'(1X,F7.4)') gn
   write(9,'(1X,F7.4)') gz
   write(9,'(1X,F7.4)') psi
   write(9,'(1X,F7.4)') phi
   write(9,'(1X,F7.4)') theta
   write(9,'(1X,F7.4)') zeta
   write(9,'(1X,I7)') nx
   write(9,'(1X,I7)') ns
   do i=1,ns
     write(9,'(1X,F8.5)') g(i) 
   end do
   do i=1,ns
     write(9,'(1X,F8.5)') tauk(i) 
   end do
   do i=1,ns
     write(9,'(1X,F8.5)') taul(i) 
   end do
   do i=1,ns
     write(9,'(1X,F8.5)') tauc(i) 
   end do
   do i=1,ns
     write(9,'(1X,F8.5)') taux(i) 
   end do
   do i=1,ns
     write(9,'(1X,F8.5)') z(i) 
   end do
   do i=1,nx
     write(9,'(1X,F8.5)') xa(i) 
   end do
   do j=1,ns
     do i=1,nx
       write(9,'(1X,F8.5)') c0(i,j)
     end do
   end do

end program adjcost
