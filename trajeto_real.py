import numpy as np
import matplotlib.pyplot as plt
import quaternion
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

#Filtro de Madgwick
class MadgwickFilter:
    def __init__(self, beta=0.1, sample_rate=100.0):
        self.beta = beta
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.q = quaternion.quaternion(1, 0, 0, 0)

    def update(self, gyr, acc, mag=None):
        acc = np.array(acc)
        if np.linalg.norm(acc) == 0:
            return self.q
        acc = acc / np.linalg.norm(acc)
        
        f = np.array([
            2*(self.q.x*self.q.z - self.q.w*self.q.y) - acc[0],
            2*(self.q.w*self.q.x + self.q.y*self.q.z) - acc[1],
            2*(0.5 - self.q.x**2 - self.q.y**2) - acc[2]
        ])
        J = np.array([
            [-2*self.q.y,  2*self.q.z, -2*self.q.w, 2*self.q.x],
            [ 2*self.q.x,  2*self.q.w,  2*self.q.z, 2*self.q.y],
            [0, -4*self.q.x, -4*self.q.y, 0]
        ])

        gradient = J.T @ f
        if np.linalg.norm(gradient) != 0:
            gradient = gradient / np.linalg.norm(gradient)
        
        q_dot_from_gyr = 0.5 * self.q * quaternion.quaternion(0, *gyr)
        
        q_dot = q_dot_from_gyr - self.beta * quaternion.quaternion(*gradient)
        self.q += q_dot * self.dt
        self.q = self.q.normalized()
        return self.q

    def get_euler(self):
        q = self.q
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x**2 + q.y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])


#Funções auxiliares
def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])

def rmse(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.sqrt(np.mean((a-b)**2))

def mae(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.mean(np.abs(a-b))

def perc_red(a, b):
    return 100 * (a - b) / a if a != 0 else 0


#Carregar dados
trajeto = np.loadtxt("trajeto5.dat", unpack=True, delimiter=',')
dN, dE, vN, vE, aN, aE, _, _, roll, pitch, yaw_meas = trajeto
n = len(dN)
dt_arquivo = 1.0
atraso = int(1/dt_arquivo) if dt_arquivo > 0 else 1

#Ground truth alinhado
dN_gt, dE_gt = dN[:-atraso], dE[:-atraso]
vN_gt, vE_gt = vN[:-atraso], vE[:-atraso]
aN_gt, aE_gt = aN[:-atraso], aE[:-atraso]

#Pré-processamento
a_body = np.zeros((n, 2))
gyro_z = np.gradient(np.unwrap(yaw_meas), dt_arquivo)
for i in range(n):
    a_body[i, :] = rotz(yaw_meas[i]).T @ np.array([aN[i], aE[i]])

#Método 1: Madgwick + Kalman (MK)
madgwick = MadgwickFilter(beta=0.1, sample_rate=1/dt_arquivo)
madgwick_orientations = []
for i in range(n):
    madgwick.update(np.array([0., 0., gyro_z[i]]), np.array([a_body[i, 0], a_body[i, 1], 9.81]))
    madgwick_orientations.append(madgwick.get_euler())
yaw_mk = np.array(madgwick_orientations)[:, 2]
a_nav_mk = np.zeros((n, 2))
for i in range(n):
    a_nav_mk[i] = rotz(yaw_mk[i]) @ a_body[i]

def apply_kf_1d(pos_meas, vel_meas, acc_input, Rd, Rv, Q_var, dt):
    n = len(pos_meas); kf = KalmanFilter(dim_x=2, dim_z=2)
    kf.x = np.array([[pos_meas[0]], [vel_meas[0]]]); kf.F = np.array([[1, dt], [0, 1]]); kf.H = np.eye(2)
    kf.B = np.array([[0.5*dt**2], [dt]]); kf.P *= 100; kf.R = np.diag([Rd, Rv])
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var); est = np.zeros((n, 2))
    for i in range(n):
        kf.predict(u=acc_input[i]); kf.update(np.array([[pos_meas[i]], [vel_meas[i]]])); est[i] = kf.x.flatten()
    return est

N_kf_mk = apply_kf_1d(dN, vN, a_nav_mk[:, 0], 0.1, 0.1, 0.1, dt_arquivo)
E_kf_mk = apply_kf_1d(dE, vE, a_nav_mk[:, 1], 0.1, 0.1, 0.1, dt_arquivo)
dN_mk, vN_mk = N_kf_mk[:, 0], N_kf_mk[:, 1]; dE_mk, vE_mk = E_kf_mk[:, 0], E_kf_mk[:, 1]
aN_mk, aE_mk = a_nav_mk[:, 0], a_nav_mk[:, 1]
dN_mk, dE_mk, vN_mk, vE_mk, aN_mk, aE_mk = [x[:-atraso] for x in [dN_mk, dE_mk, vN_mk, vE_mk, aN_mk, aE_mk]]

#Método 2: EKF

high_freq_dt = 0.1  # Filtro roda a 10 Hz
gps_dt = 1.0        # Medições chegam a 1 Hz
updates_per_gps = int(gps_dt / high_freq_dt)

#Estado: [dN, dE, vN, vE, aN, aE, yaw, yaw_rate]
dim_x = 8
#Medição: [dN, dE, yaw]
dim_z = 3
kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

#Estado inicial
kf.x = np.array([dN[0], dE[0], vN[0], vE[0], aN[0], aE[0], yaw_meas[0], gyro_z[0]])

#Incerteza inicial
kf.P = np.diag([1., 1., 1., 1., 1., 1., np.deg2rad(1)**2, np.deg2rad(1)**2])

#Incerteza da medição (R) - agora inclui o yaw
sigma_pos_meas = 0.1
sigma_yaw_meas = np.deg2rad(2.0) # Incerteza da medição de yaw
kf.R = np.diag([sigma_pos_meas**2, sigma_pos_meas**2, sigma_yaw_meas**2])

#Matriz de medição (H) - Medimos os estados 0, 1 e 6 (dN, dE, yaw)
kf.H = np.zeros((dim_z, dim_x))
kf.H[0, 0] = 1 # dN
kf.H[1, 1] = 1 # dE
kf.H[2, 6] = 1 # yaw

#Matriz de transição de estado (F) para 8 estados
dt = high_freq_dt
F_pos = np.array([[1, 0, dt, 0, 0.5*dt**2, 0],
                   [0, 1, 0, dt, 0, 0.5*dt**2],
                   [0, 0, 1, 0, dt, 0],
                   [0, 0, 0, 1, 0, dt],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]])
F_yaw = np.array([[1, dt],
                  [0, 1]])
kf.F = np.block([
    [F_pos, np.zeros((6, 2))],
    [np.zeros((2, 6)), F_yaw]
])

#Ruído do processo (Q)
q_var_pos = 0.5 #Ajuste para a parte de posição/velocidade
q_var_yaw = 0.1 #Ajuste para a parte de yaw
Q_pos = Q_discrete_white_noise(dim=3, dt=dt, var=q_var_pos, block_size=2, order_by_dim=True)
Q_yaw = Q_discrete_white_noise(dim=2, dt=dt, var=q_var_yaw)
kf.Q = np.block([
    [Q_pos, np.zeros((6, 2))],
    [np.zeros((2, 6)), Q_yaw]
])


#LOOP DE EXECUÇÃO
kf_high_freq_states = [kf.x]
for i in range(n - 1):
    # Definimos as "entradas de controle" para o próximo segundo
    kf.x[4] = aN[i]     #Assume aceleração N constante
    kf.x[5] = aE[i]     #Assume aceleração E constante
    kf.x[7] = gyro_z[i] #Assume taxa de yaw constante

    for _ in range(updates_per_gps):
        kf.predict()
        kf_high_freq_states.append(kf.x.copy())

    #Correção com os dados de 1Hz do arquivo
    z_k = np.array([dN[i+1], dE[i+1], yaw_meas[i+1]])
    kf.update(z_k)
    kf_high_freq_states[-1] = kf.x.copy()

#EXTRAÇÃO DOS RESULTADOS
kf_states_hf = np.array(kf_high_freq_states)
dN_ekf, dE_ekf, vN_ekf, vE_ekf, aN_ekf, aE_ekf, yaw_ekf, _ = kf_states_hf.T

#cálculo de erro
indices_gps = np.arange(n) * updates_per_gps
dN_ekf_aligned = dN_ekf[indices_gps]; dE_ekf_aligned = dE_ekf[indices_gps]
vN_ekf_aligned = vN_ekf[indices_gps]; vE_ekf_aligned = vE_ekf[indices_gps]
aN_ekf_aligned = aN_ekf[indices_gps]; aE_ekf_aligned = aE_ekf[indices_gps]

dN_ekf_err, dE_ekf_err = dN_ekf_aligned[:-atraso], dE_ekf_aligned[:-atraso]
vN_ekf_err, vE_ekf_err = vN_ekf_aligned[:-atraso], vE_ekf_aligned[:-atraso]
aN_ekf_err, aE_ekf_err = aN_ekf_aligned[:-atraso], aE_ekf_aligned[:-atraso]


# Métricas
rmse_pos_mk = np.sqrt(rmse(dN_mk, dN_gt)**2 + rmse(dE_mk, dE_gt)**2)
rmse_vel_mk = np.sqrt(rmse(vN_mk, vN_gt)**2 + rmse(vE_mk, vE_gt)**2)
rmse_acc_mk = np.sqrt(rmse(aN_mk, aN_gt)**2 + rmse(aE_mk, aE_gt)**2)
rmse_pos_ekf = np.sqrt(rmse(dN_ekf_err, dN_gt)**2 + rmse(dE_ekf_err, dE_gt)**2)
rmse_vel_ekf = np.sqrt(rmse(vN_ekf_err, vN_gt)**2 + rmse(vE_ekf_err, vE_gt)**2)
rmse_acc_ekf = np.sqrt(rmse(aN_ekf_err, aN_gt)**2 + rmse(aE_ekf_err, aE_gt)**2)
mae_pos_mk = mae(dN_mk, dN_gt) + mae(dE_mk, dE_gt)
mae_vel_mk = mae(vN_mk, vN_gt) + mae(vE_mk, vE_gt)
mae_acc_mk = mae(aN_mk, aN_gt) + mae(aE_mk, aE_gt)
mae_pos_ekf = mae(dN_ekf_err, dN_gt) + mae(dE_ekf_err, dE_gt)
mae_vel_ekf = mae(vN_ekf_err, vN_gt) + mae(vE_ekf_err, vE_gt)
mae_acc_ekf = mae(aN_ekf_err, aN_gt) + mae(aE_ekf_err, aE_gt)


# Gráficos
plt.style.use('seaborn-v0_8')

# Gráfico da Trajetória
plt.figure(figsize=(8, 8))
plt.plot(dE_gt, dN_gt, 'k', label='Ground truth', linewidth=2, marker='o', markersize=4)
plt.plot(dE_mk, dN_mk, color='blue', label='Madgwick + Kalman')
plt.plot(dE_ekf, dN_ekf, color='red', label='EKF', linewidth=2, alpha=0.8)
plt.xlabel('Leste [m]'); plt.ylabel('Norte [m]'); plt.title('Trajetória Comparativa')
plt.legend(); plt.grid(True); plt.axis('equal')

# Gráfico do Erro de Posição
fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
tempo_eixo = np.arange(len(dN_gt)) * dt_arquivo
axs[0].plot(tempo_eixo, dN_mk - dN_gt, color='blue', label='Erro Madgwick + Kalman')
axs[0].plot(tempo_eixo, dN_ekf_err - dN_gt, color='red', label='Erro EKF')
axs[0].set_ylabel('Erro Norte [m]'); axs[0].legend(); axs[0].grid(True)
axs[1].plot(tempo_eixo, dE_mk - dE_gt, color='blue', label='Erro Madgwick + Kalman')
axs[1].plot(tempo_eixo, dE_ekf_err - dE_gt, color='red', label='Erro EKF')
axs[1].set_ylabel('Erro Leste [m]'); axs[1].set_xlabel('Tempo [s]'); axs[1].legend(); axs[1].grid(True)
fig.suptitle('Erro de Posição', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Gráfico do Erro de Velocidade
fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axs[0].plot(tempo_eixo, vN_mk - vN_gt, color='blue', label='Erro Madgwick + Kalman')
axs[0].plot(tempo_eixo, vN_ekf_err - vN_gt, color='red', label='Erro EKF', linewidth=1)
axs[0].set_ylabel('Erro Vel. Norte [m/s]'); axs[0].legend(); axs[0].grid(True)
axs[1].plot(tempo_eixo, vE_mk - vE_gt, color='blue', label='Erro Madgwick + Kalman')
axs[1].plot(tempo_eixo, vE_ekf_err - vE_gt, color='red', label='Erro EKF', linewidth=1)
axs[1].set_ylabel('Erro Vel. Leste [m/s]'); axs[1].set_xlabel('Tempo [s]'); axs[1].legend(); axs[1].grid(True)
fig.suptitle('Erro de Velocidade', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Gráfico do Erro de Aceleração
fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axs[0].plot(tempo_eixo, aN_mk - aN_gt, color='blue', label='Erro Madgwick + Kalman')
axs[0].plot(tempo_eixo, aN_ekf_err - aN_gt, color='red', label='Erro EKF', linewidth=1)
axs[0].set_ylabel('Erro Acc Norte [m/s²]'); axs[0].legend(); axs[0].grid(True)
axs[1].plot(tempo_eixo, aE_mk - aE_gt, color='blue', label='Erro Madgwick + Kalman')
axs[1].plot(tempo_eixo, aE_ekf_err - aE_gt, color='red', label='Erro EKF', linewidth=1)
axs[1].set_ylabel('Erro Acc Leste [m/s²]'); axs[1].set_xlabel('Tempo [s]'); axs[1].legend(); axs[1].grid(True)
fig.suptitle('Erro de Aceleração', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


plt.show()

#Tabela LaTeX 
print("\n--- Tabela LaTeX para o Relatório ---")

mae_pos_kf, rmse_pos_kf = mae_pos_ekf, rmse_pos_ekf
mae_vel_kf, rmse_vel_kf = mae_vel_ekf, rmse_vel_ekf
mae_acc_kf, rmse_acc_kf = mae_acc_ekf, rmse_acc_ekf

latex_table = f"""
\\begin{{table}}[h]
\\centering
\\small
\\caption{{Comparação de erros entre MK e o Filtro de Fusão}}
\\label{{tab:error_comparison_final}}
\\rowcolors{{2}}{{gray!10}}{{white}}
\\begin{{tabular}}{{lcccc}}
\\hline
& \\textbf{{MK (MAE/RMSE)}} & \\textbf{{Filtro Fusão (MAE/RMSE)}} & \\textbf{{Redução MAE}} & \\textbf{{Redução RMSE}} \\\\
\\hline
Posição [m] & {mae_pos_mk:.3f} / {rmse_pos_mk:.3f} & {mae_pos_kf:.3f} / {rmse_pos_kf:.3f} & {perc_red(mae_pos_mk, mae_pos_kf):.1f}\\% & {perc_red(rmse_pos_mk, rmse_pos_kf):.1f}\\% \\\\
Velocidade [m/s] & {mae_vel_mk:.3f} / {rmse_vel_mk:.3f} & {mae_vel_kf:.3f} / {rmse_vel_kf:.3f} & {perc_red(mae_vel_mk, mae_vel_kf):.1f}\\% & {perc_red(rmse_vel_mk, rmse_vel_kf):.1f}\\% \\\\
Aceleração [m/s²] & {mae_acc_mk:.3f} / {rmse_acc_mk:.3f} & {mae_acc_kf:.3f} / {rmse_acc_kf:.3f} & {perc_red(mae_acc_mk, mae_acc_kf):.1f}\\% & {perc_red(rmse_acc_mk, rmse_acc_kf):.1f}\\% \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
print(latex_table)