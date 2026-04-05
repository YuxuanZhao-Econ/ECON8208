include(raw"C:/Users/30945/Desktop/Yuxuan ZHAO/minnesota/PhD_first_year/ECON8208/HW4_Yichen/Riccati.jl")
include(raw"C:/Users/30945/Desktop/Yuxuan ZHAO/minnesota/PhD_first_year/ECON8208/HW4_Yichen/vaughan.jl")
A_prod=1.0
theta=0.36
beta_d=0.96
k_ss=(A_prod*beta_d*theta)^(1/(1-theta))
c_ss=A_prod*k_ss^theta-k_ss
x_bar=[k_ss]
u_bar=[c_ss]
r_fn(x,u)=log(u[1])
g_fn(x,u)=[A_prod*x[1]^theta-u[1]]
Q,W,R=compute_QWR(r_fn,x_bar,u_bar)
A_mat,B_mat=compute_AB(g_fn,x_bar,u_bar)
Fc,Pc=solve_riccati(Q,W,R,A_mat,B_mat,beta_d)
A_clc=A_mat-B_mat*Fc
Q_tilde,A_tilde,B_tilde=transform_to_standard(Q,W,R,A_mat,B_mat,beta_d)
F_v,P_v=solve_vaughan(Q_tilde,R,A_tilde,B_tilde,W)
println("k_ss=$k_ss c_ss=$c_ss")
println("Q=$Q W=$W R=$R A=$A_mat B=$B_mat")
println("Fc=$Fc A_clc=$A_clc F_v=$F_v")
T=10
k0=k_ss/2
k_path=zeros(T+1)
c_path=zeros(T)
y_path=zeros(T)
k_path[1]=k0
for t in 1:T
    dk=k_path[t]-k_ss
    c_path[t]=c_ss-F_v[1,1]*dk
    y_path[t]=A_prod*k_path[t]^theta
    k_path[t+1]=y_path[t]-c_path[t]
    println((t=t,k=k_path[t],dk=dk,c=c_path[t],y=y_path[t],knext=k_path[t+1]))
end
