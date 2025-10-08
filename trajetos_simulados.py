import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')


#CONFIGURAÇÃO SIMULAÇÃO
np.random.seed(42)
dt = 0.1
t = np.arange(0, 60, dt)
g = 9.80665
pathsize = 15
traj_choice =  4 #Escolha entre 1,2,3,4


#FUNÇÃO PARA GERAR TRAJETÓRIA

def gerar_trajetoria(path, t):
    if path==1:
        w=0.25
        x=pathsize*np.cos(w*t)
        y=pathsize*np.sin(w*t)
        vx=-w*pathsize*np.sin(w*t)
        vy=w*pathsize*np.cos(w*t)
        ax=-(w**2)*pathsize*np.cos(w*t)
        ay=-(w**2)*pathsize*np.sin(w*t)
    elif path==2:
        w1,w2=0.5,1.0
        x=pathsize*np.cos(w1*t)
        y=pathsize*np.sin(w2*t)
        vx=-pathsize*w1*np.sin(w1*t)
        vy=pathsize*w2*np.cos(w2*t)
        ax=-pathsize*(w1**2)*np.cos(w1*t)
        ay=-pathsize*(w2**2)*np.sin(w2*t)
    elif path==3:
        x=pathsize*np.cos(0.125*t)
        y=pathsize*np.sin(0.25*t)
        vx=-0.125*pathsize*np.sin(0.125*t)
        vy=0.25*pathsize*np.cos(0.25*t)
        ax=-0.015625*pathsize*np.cos(0.125*t)
        ay=-0.0625*pathsize*np.sin(0.25*t)
    elif path==4:
        w1,w2=1.5,1.0
        x=pathsize*np.sin(w1*t)
        y=pathsize*np.cos(w2*t)
        vx=pathsize*w1*np.cos(w1*t)
        vy=-pathsize*w2*np.sin(w2*t)
        ax=-pathsize*(w1**2)*np.sin(w1*t)
        ay=-pathsize*(w2**2)*np.cos(w2*t)
    return x,y,vx,vy,ax,ay

#Trajetória selecionada
x_true, y_true, vx_true, vy_true, ax_true, ay_true = gerar_trajetoria(traj_choice, t)

#IMU SIMULADO
acc_noise_std = 0.05
gyro_noise_std = np.deg2rad(0.01)
mag_noise_std = np.deg2rad(2.0)

a_world_3d = np.vstack([ax_true, ay_true, np.zeros_like(t)])
gravity = np.array([0,0,g]).reshape(3,1)

psi = np.arctan2(vy_true, vx_true) 
psi_dot = np.gradient(np.unwrap(psi), dt)

def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],[s, c, 0],[0,0,1]])

acc_body = np.zeros((3, len(t)))
for k in range(len(t)):
    Rbw = Rz(psi[k]).T
    acc_body[:,k] = (Rbw @ (a_world_3d[:,k].reshape(3,1)+gravity)).ravel()
acc_body += np.random.normal(0, acc_noise_std, size=acc_body.shape)
gyro_z = psi_dot + np.random.normal(0, gyro_noise_std, size=len(t))

#GPS
gps_noise_std = 0.2
x_gps = x_true + np.random.normal(0, gps_noise_std, size=len(t))
y_gps = y_true + np.random.normal(0, gps_noise_std, size=len(t))


#MADGWICK + KALMAN LINEAR
beta = 0.15

def quat_from_yaw(psi):
    cz = np.cos(psi/2)
    sz = np.sin(psi/2)
    return np.array([cz,0,0,sz])

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v/n

def yaw_from_quat(q):
    w,x,y,z = q
    return 2*np.arctan2(z,w)

q_est = np.zeros((4,len(t)))
q = quat_from_yaw(psi[0])
q_est[:,0] = q
mag_yaw_noisy = psi + np.random.normal(0, mag_noise_std, size=len(t))

for k in range(1,len(t)):
    dq = quat_from_yaw(gyro_z[k-1]*dt)
    q_pred = normalize(np.array([
        q[0]*dq[0]-q[1]*dq[1]-q[2]*dq[2]-q[3]*dq[3],
        q[0]*dq[1]+q[1]*dq[0]+q[2]*dq[3]-q[3]*dq[2],
        q[0]*dq[2]-q[1]*dq[3]+q[2]*dq[0]+q[3]*dq[1],
        q[0]*dq[3]+q[1]*dq[2]-q[2]*dq[1]+q[3]*dq[0]
    ]))
    q_meas = quat_from_yaw(mag_yaw_noisy[k])
    q = normalize((1-beta)*q_pred + beta*q_meas)
    q_est[:,k] = q

a_world_est = np.zeros((3,len(t)))
for k in range(len(t)):
    psi_hat = yaw_from_quat(q_est[:,k])
    Rwb = Rz(psi_hat)
    a_world_est[:,k] = (Rwb @ acc_body[:,k].reshape(3,1)).ravel() - gravity.ravel()

ax_hat_in = a_world_est[0,:]
ay_hat_in = a_world_est[1,:]

F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
B = np.array([[0.5*dt**2,0],[0,0.5*dt**2],[dt,0],[0,dt]])
H = np.array([[1,0,0,0],[0,1,0,0]])
xk = np.array([x_gps[0], y_gps[0],vx_true[0],vy_true[0]])
P = np.eye(4)*10
Q = np.diag([0.01, 0.01, 0.1, 0.1])
R_mat = np.eye(2)*(gps_noise_std**2)

x_est_mk, y_est_mk, vx_est_mk, vy_est_mk = [],[],[],[]
for k in range(len(t)):
    uk = np.array([ax_hat_in[k], ay_hat_in[k]])
    xk = F@xk + B@uk
    P = F@P@F.T + Q
    zk = np.array([x_gps[k], y_gps[k]])
    yk = zk - H@xk
    S = H@P@H.T + R_mat
    K = P@H.T @ np.linalg.inv(S)
    xk = xk + K@yk
    P = (np.eye(4)-K@H)@P
    x_est_mk.append(xk[0]); y_est_mk.append(xk[1])
    vx_est_mk.append(xk[2]); vy_est_mk.append(xk[3])

x_est_mk = np.array(x_est_mk)
y_est_mk = np.array(y_est_mk)
vx_est_mk = np.array(vx_est_mk)
vy_est_mk = np.array(vy_est_mk)
ax_est_mk = np.gradient(vx_est_mk, dt)
ay_est_mk = np.gradient(vy_est_mk, dt)


#EKF
def quat_mul(q1,q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def quat_from_gyro(wz,dt):
    dpsi = wz*dt; return np.array([np.cos(dpsi/2),0,0,np.sin(dpsi/2)])
def R_from_quat(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def f_nonlin(xk, uk):
    x,y,vx,vy,qw,qx,qy,qz = xk
    q = normalize(np.array([qw,qx,qy,qz]))
    Rwb = R_from_quat(q)
    a_world = (Rwb @ np.array(uk[:3]).reshape(3,1)).ravel() - np.array([0,0,g])
    vxn = vx + a_world[0]*dt; vyn = vy + a_world[1]*dt
    xn = x + vx*dt + 0.5*a_world[0]*dt**2; yn = y + vy*dt + 0.5*a_world[1]*dt**2
    dq = quat_from_gyro(uk[3], dt)
    qn = normalize(quat_mul(q,dq))
    return np.array([xn,yn,vxn,vyn,qn[0],qn[1],qn[2],qn[3]])
def h_meas(xk): return np.array([xk[0],xk[1]])
def numerical_jacobian_f(xk,uk,eps=1e-6):
    n=len(xk); F=np.zeros((n,n)); fx=f_nonlin(xk,uk)
    for i in range(n):
        dx=np.zeros(n); dx[i]=eps
        F[:,i]=(f_nonlin(xk+dx,uk)-fx)/eps
    return F
def numerical_jacobian_h(xk,eps=1e-6):
    n=len(xk); m=2; H=np.zeros((m,n)); hx=h_meas(xk)
    for i in range(n):
        dx=np.zeros(n); dx[i]=eps
        H[:,i]=(h_meas(xk+dx)-hx)/eps
    return H

xk_ekf = np.array([x_gps[0],y_gps[0],vx_true[0],vy_true[0],1,0,0,0])
P_ekf = np.diag([1,1,1,1,0.01,0.01,0.01,0.01])*10
Q_ekf = np.diag([0.05,0.05,0.1,0.1,0.001,0.001,0.001,0.001])
R_ekf = np.eye(2)*gps_noise_std**2
x_est_ekf,y_est_ekf,vx_est_ekf,vy_est_ekf = [],[],[],[]
for k in range(len(t)):
    uk = [acc_body[0,k], acc_body[1,k], acc_body[2,k], gyro_z[k]]
    Fk = numerical_jacobian_f(xk_ekf,uk)
    x_pred = f_nonlin(xk_ekf,uk)
    P_pred = Fk@P_ekf@Fk.T + Q_ekf
    zk = np.array([x_gps[k],y_gps[k]])
    Hk = numerical_jacobian_h(x_pred)
    yk = zk - h_meas(x_pred)
    Sk = Hk@P_pred@Hk.T + R_ekf
    K = P_pred@Hk.T @ np.linalg.inv(Sk)
    xk_ekf = x_pred + K@yk
    xk_ekf[4:8] = normalize(xk_ekf[4:8])
    P_ekf = (np.eye(len(xk_ekf))-K@Hk)@P_pred
    x_est_ekf.append(xk_ekf[0]); y_est_ekf.append(xk_ekf[1])
    vx_est_ekf.append(xk_ekf[2]); vy_est_ekf.append(xk_ekf[3])
x_est_ekf=np.array(x_est_ekf); y_est_ekf=np.array(y_est_ekf)
vx_est_ekf=np.array(vx_est_ekf); vy_est_ekf=np.array(vy_est_ekf)
ax_est_ekf=np.gradient(vx_est_ekf,dt); ay_est_ekf=np.gradient(vy_est_ekf,dt)


# ERROS

err_pos_mk = np.sqrt((x_est_mk - x_true)**2 + (y_est_mk - y_true)**2)
err_pos_ekf = np.sqrt((x_est_ekf - x_true)**2 + (y_est_ekf - y_true)**2)
mae_pos_mk = np.mean(err_pos_mk); rmse_pos_mk = np.sqrt(np.mean(err_pos_mk**2))
mae_pos_ekf = np.mean(err_pos_ekf); rmse_pos_ekf = np.sqrt(np.mean(err_pos_ekf**2))
err_vel_mk = np.sqrt((vx_est_mk - vx_true)**2 + (vy_est_mk - vy_true)**2)
err_vel_ekf = np.sqrt((vx_est_ekf - vx_true)**2 + (vy_est_ekf - vy_true)**2)
mae_vel_mk = np.mean(err_vel_mk); rmse_vel_mk = np.sqrt(np.mean(err_vel_mk**2))
mae_vel_ekf = np.mean(err_vel_ekf); rmse_vel_ekf = np.sqrt(np.mean(err_vel_ekf**2))
err_acc_mk = np.sqrt((ax_est_mk - ax_true)**2 + (ay_est_mk - ay_true)**2)
err_acc_ekf = np.sqrt((ax_est_ekf - ax_true)**2 + (ay_est_ekf - ay_true)**2)
mae_acc_mk = np.mean(err_acc_mk); rmse_acc_mk = np.sqrt(np.mean(err_acc_mk**2))
mae_acc_ekf = np.mean(err_acc_ekf); rmse_acc_ekf = np.sqrt(np.mean(err_acc_ekf**2))


#PLOTS DE ERRO
plt.figure(figsize=(6,6))
plt.plot(x_true,y_true,'k-',label="Real", color='black')
plt.scatter(x_gps,y_gps,s=10,c='g',alpha=0.5,label="GPS")
plt.plot(x_est_mk,y_est_mk,'b-',label="MK",alpha=0.8)
plt.plot(x_est_ekf,y_est_ekf,'r-',label="EKF", alpha=0.8)
plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.grid(); plt.legend(loc='best')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.plot(t,err_pos_mk,'b',label="MK")
plt.plot(t,err_pos_ekf,'r',label="EKF")
plt.xlabel("Tempo [s]"); plt.ylabel("Erro Posição [m]"); plt.grid(); plt.legend(loc='upper right')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.plot(t,err_vel_mk,'b',label="MK")
plt.plot(t,err_vel_ekf,'r',label="EKF")
plt.xlabel("Tempo [s]"); plt.ylabel("Erro Velocidade [m/s]"); plt.grid(); plt.legend(loc='upper right')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.plot(t,err_acc_mk,'b',label="MK")
plt.plot(t,err_acc_ekf,'r',label="EKF")
plt.xlabel("Tempo [s]"); plt.ylabel("Erro Aceleração [m/s²]"); plt.grid(); plt.legend(loc='upper right')
plt.tight_layout(); plt.show()

#Função para calcular a redução percentual
def reduction_pct(mk, ekf):
    return 100.0 * (mk - ekf) / mk if mk != 0 else 0.0

# --- Cálculo das reduções ---
red_mae_pos  = reduction_pct(mae_pos_mk,  mae_pos_ekf)
red_rmse_pos = reduction_pct(rmse_pos_mk, rmse_pos_ekf)

red_mae_vel  = reduction_pct(mae_vel_mk,  mae_vel_ekf)
red_rmse_vel = reduction_pct(rmse_vel_mk, rmse_vel_ekf)

red_mae_acc  = reduction_pct(mae_acc_mk,  mae_acc_ekf)
red_rmse_acc = reduction_pct(rmse_acc_mk, rmse_acc_ekf)

#tabela em LaTeX
print(r"""\begin{table}[h]
\centering
\small
\caption{Comparação de erros entre MK e EKF para a trajetória %d}
\label{tab:error_comparison_final}
\begin{tabular}{lccccrr}
\hline\hline
\textbf{Variável} & \textbf{MK (MAE)} & \textbf{EKF (MAE)} & \textbf{MK (RMSE)} & \textbf{EKF (RMSE)} & \textbf{Redução MAE} & \textbf{Redução RMSE} \\
\hline""" % traj_choice)

print(rf"Posição [m] & {mae_pos_mk:.3f} & {mae_pos_ekf:.3f} & {rmse_pos_mk:.3f} & {rmse_pos_ekf:.3f} & {red_mae_pos:.1f}\% & {red_rmse_pos:.1f}\% \\")
print(rf"Velocidade [m/s] & {mae_vel_mk:.3f} & {mae_vel_ekf:.3f} & {rmse_vel_mk:.3f} & {rmse_vel_ekf:.3f} & {red_mae_vel:.1f}\% & {red_rmse_vel:.1f}\% \\")
print(rf"Aceleração [m/s²] & {mae_acc_mk:.3f} & {mae_acc_ekf:.3f} & {rmse_acc_mk:.3f} & {rmse_acc_ekf:.3f} & {red_mae_acc:.1f}\% & {red_rmse_acc:.1f}\% \\")

print(r"""\hline\hline
\end{tabular}
\end{table}""")



