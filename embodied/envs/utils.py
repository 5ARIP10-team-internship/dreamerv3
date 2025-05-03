import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("plots", exist_ok=True)

## ------------ Clarke Park Tranformation ------------ ##
class ClarkePark:
    def __init__(self):
        return

    @staticmethod
    def abc_to_alphabeta0(a, b, c):
        alpha = (2 / 3) * (a - b / 2 - c / 2)
        beta = (2 / 3) * (np.sqrt(3) * (b - c) / 2)
        z = (2 / 3) * ((a + b + c) / 2)
        return alpha, beta, z

    @staticmethod
    def alphabeta0_to_abc(alpha, beta, z):
        a = alpha + z
        b = -alpha / 2 + beta * np.sqrt(3) / 2 + z
        c = -alpha / 2 - beta * np.sqrt(3) / 2 + z
        return a, b, c

    @staticmethod
    def abc_to_dq0_d(a, b, c, wt, delta=0):
        d = (2 / 3) * (a * np.cos(wt + delta) + b * np.cos(wt + delta - (2 * np.pi / 3)) + c * np.cos(wt + delta + (2 * np.pi / 3)))
        q = (2 / 3) * (-a * np.sin(wt + delta) - b * np.sin(wt + delta - (2 * np.pi / 3)) - c * np.sin(wt + delta + (2 * np.pi / 3)))
        z = (2 / 3) * (a + b + c) / 2
        return d, q, z

    @staticmethod
    def abc_to_dq0_q(a, b, c, wt, delta=0):
        d = (2 / 3) * (a * np.sin(wt + delta) + b * np.sin(wt + delta - (2 * np.pi / 3)) + c * np.sin(wt + delta + (2 * np.pi / 3)))
        q = (2 / 3) * (a * np.cos(wt + delta) + b * np.cos(wt + delta - (2 * np.pi / 3)) + c * np.cos(wt + delta + (2 * np.pi / 3)))
        z = (2 / 3) * (a + b + c) / 2
        return d, q, z

    @staticmethod
    def dq0_to_abc_d(d, q, z, wt, delta=0):
        a = d * np.cos(wt + delta) - q * np.sin(wt + delta) + z
        b = d * np.cos(wt - (2 * np.pi / 3) + delta) - q * np.sin(wt - (2 * np.pi / 3) + delta) + z
        c = d * np.cos(wt + (2 * np.pi / 3) + delta) - q * np.sin(wt + (2 * np.pi / 3) + delta) + z
        return a, b, c

    @staticmethod
    def dq0_to_abc_q(d, q, z, wt, delta=0):
        a = d * np.sin(wt + delta) + q * np.cos(wt + delta) + z
        b = d * np.sin(wt - (2 * np.pi / 3) + delta) + q * np.cos(wt - (2 * np.pi / 3) + delta) + z
        c = d * np.sin(wt + (2 * np.pi / 3) + delta) + q * np.cos(wt + (2 * np.pi / 3) + delta) + z
        return a, b, c

# def plot_results():
class SSAnalysis:
    def __init__(self):
        return

    def continuous(self,a, b, w=None, plot_current=False):
        if w is not None:
            # dx/dt = a*x + b*u + w
            # Steady-state
            # 0 = a * x_ss + b * u_ss + w
            # x_ss = - a^-1 * (b * u_ss + w)
            x_ss = lambda vdq: -np.linalg.inv(a) @ (b @ vdq + np.kron(np.ones((vdq.shape[1], 1)), w).T) if vdq.shape == (
            2, 1000) else -np.linalg.inv(a) @ (b @ vdq + w)

            x_ss1 = x_ss(np.array([0, self.vdq_max]))
            x_ss2 = x_ss(np.array([self.vdq_max, 0]))
            x_ss3 = x_ss(np.array([0, -self.vdq_max]))
            x_ss4 = x_ss(np.array([-self.vdq_max, 0]))
            x_ss5 = x_ss(np.array([self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss6 = x_ss(np.array([self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))
            x_ss7 = x_ss(np.array([-self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss8 = x_ss(np.array([-self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))

            # 0 = a * x_ss + b * u_ss + w
            # u_ss = - b^-1 * (a * x_ss + w)
            u_ss = lambda idq: -np.linalg.inv(b) @ (a @ idq + w)

            u_ss1 = u_ss([0, self.i_max])
            u_ss2 = u_ss([self.i_max, 0])
            u_ss3 = u_ss([0, -self.i_max])
            u_ss4 = u_ss([-self.i_max, 0])
            u_ss5 = u_ss([self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss6 = u_ss([self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])
            u_ss7 = u_ss([-self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss8 = u_ss([-self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])

            if plot_current:
                v_d = np.linspace(-1, 1, 1000)
                v_q = np.sqrt(1 - np.power(v_d, 2))
                vdq = self.vdq_max * np.array([v_d, v_q])
                x_ss_data_pos = x_ss(vdq)
                vdq = self.vdq_max * np.array([v_d, -v_q])
                x_ss_data_neg = x_ss(vdq)
                id = np.concatenate((x_ss_data_pos[0], x_ss_data_neg[0]), 0)
                iq = np.concatenate((x_ss_data_pos[1], x_ss_data_neg[1]), 0)
                plt.plot(id, iq, label="Current by voltage limitation")
                id = np.linspace(-1, 1, 1000)
                iq = np.sqrt(1 - np.power(id, 2))
                id_circle = self.i_max * np.concatenate((id, id), 0)
                iq_circle = self.i_max * np.concatenate((iq, -iq), 0)
                plt.plot(id_circle, iq_circle, label="Maximum current circle")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                           fancybox=True, shadow=True, ncol=2)
                plt.title("Continuous state-space model")
                plt.show()
        else:
            # dx/dt = a*x + b*u
            # Steady-state
            # 0 = a * x_ss + b * u_ss
            # x_ss = - a^-1 * (b * u_ss )
            x_ss = lambda vdq: -np.linalg.inv(a) @ (b @ vdq)

            x_ss1 = x_ss(np.array([0, self.vdq_max]))
            x_ss2 = x_ss(np.array([self.vdq_max, 0]))
            x_ss3 = x_ss(np.array([0, -self.vdq_max]))
            x_ss4 = x_ss(np.array([-self.vdq_max, 0]))
            x_ss5 = x_ss(np.array([self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss6 = x_ss(np.array([self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))
            x_ss7 = x_ss(np.array([-self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss8 = x_ss(np.array([-self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))

            # 0 = a * x_ss + b * u_ss
            # u_ss = - b^-1 * (a * x_ss)
            u_ss = lambda idq: -np.linalg.inv(b) @ (a @ idq)

            u_ss1 = u_ss([0, self.i_max])
            u_ss2 = u_ss([self.i_max, 0])
            u_ss3 = u_ss([0, -self.i_max])
            u_ss4 = u_ss([-self.i_max, 0])
            u_ss5 = u_ss([self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss6 = u_ss([self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])
            u_ss7 = u_ss([-self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss8 = u_ss([-self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])

            if plot_current:
                v_d = np.linspace(-1, 1, 1000)
                v_q = np.sqrt(1 - np.power(v_d, 2))
                vdq = self.vdq_max * np.array([v_d, v_q])
                x_ss_data_pos = x_ss(vdq)
                vdq = self.vdq_max * np.array([v_d, -v_q])
                x_ss_data_neg = x_ss(vdq)
                id = np.concatenate((x_ss_data_pos[0], x_ss_data_neg[0]), 0)
                iq = np.concatenate((x_ss_data_pos[1], x_ss_data_neg[1]), 0)
                plt.plot(id, iq, label="Current by voltage limitation")
                id = np.linspace(-1, 1, 1000)
                iq = np.sqrt(1 - np.power(id, 2))
                id_circle = self.i_max * np.concatenate((id, id), 0)
                iq_circle = self.i_max * np.concatenate((iq, -iq), 0)
                plt.plot(id_circle, iq_circle, label="Maximum current circle")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                           fancybox=True, shadow=True, ncol=2)
                plt.title("Continuous state-space model")
                plt.show()

    def discrete(self, ad, bd, wd=None, plot_current=False):
        if wd is not None:
            # x_k+1 = ad * x_k + bd * u_k + wd
            # Steady-state
            # x_ss = ad * x_ss + bd * u_ss + wd
            # x_ss = (I - ad)^-1 * (bd * u_ss + wd)
            x_ss = lambda vdq: -np.linalg.inv(np.eye(2) - ad) @ (
                        bd @ vdq + np.kron(np.ones((vdq.shape[1], 1)), wd).T) if vdq.shape == (
                2, 1000) else -np.linalg.inv(ad) @ (bd @ vdq + wd)

            x_ss1 = x_ss(np.array([0, self.vdq_max]))
            x_ss2 = x_ss(np.array([self.vdq_max, 0]))
            x_ss3 = x_ss(np.array([0, -self.vdq_max]))
            x_ss4 = x_ss(np.array([-self.vdq_max, 0]))
            x_ss5 = x_ss(np.array([self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss6 = x_ss(np.array([self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))
            x_ss7 = x_ss(np.array([-self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss8 = x_ss(np.array([-self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))

            # x_ss = ad * x_ss + bd * u_ss + wd
            # u_ss = bd^-1 * ((I - ad) * x_ss - wd)
            u_ss = lambda idq: -np.linalg.inv(bd) @ ((np.eye(2) - ad) @ idq - wd)

            u_ss1 = u_ss([0, self.i_max])
            u_ss2 = u_ss([self.i_max, 0])
            u_ss3 = u_ss([0, -self.i_max])
            u_ss4 = u_ss([-self.i_max, 0])
            u_ss5 = u_ss([self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss6 = u_ss([self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])
            u_ss7 = u_ss([-self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss8 = u_ss([-self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])

            if plot_current:
                v_d = np.linspace(-1, 1, 1000)
                v_q = np.sqrt(1 - np.power(v_d, 2))
                vdq = self.vdq_max * np.array([v_d, v_q])
                x_ss_data_pos = x_ss(vdq)
                vdq = self.vdq_max * np.array([v_d, -v_q])
                x_ss_data_neg = x_ss(vdq)
                id = np.concatenate((x_ss_data_pos[0], x_ss_data_neg[0]), 0)
                iq = np.concatenate((x_ss_data_pos[1], x_ss_data_neg[1]), 0)
                plt.plot(id, iq, label="Current by voltage limitation")
                id = np.linspace(-1, 1, 1000)
                iq = np.sqrt(1 - np.power(id, 2))
                id_circle = self.i_max * np.concatenate((id, id), 0)
                iq_circle = self.i_max * np.concatenate((iq, -iq), 0)
                plt.plot(id_circle, iq_circle, label="Maximum current circle")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                           fancybox=True, shadow=True, ncol=2)
                plt.title("Discrete state-space model")
                plt.show()
        else:
            # x_k+1 = ad * x_k + bd * u_k
            # Steady-state
            # x_ss = ad * x_ss + bd * u_ss
            # x_ss = (I - ad)^-1 * (bd * u_ss)
            x_ss = lambda vdq: -np.linalg.inv(np.eye(2) - ad) @ (bd @ vdq)

            x_ss1 = x_ss(np.array([0, self.vdq_max]))
            x_ss2 = x_ss(np.array([self.vdq_max, 0]))
            x_ss3 = x_ss(np.array([0, -self.vdq_max]))
            x_ss4 = x_ss(np.array([-self.vdq_max, 0]))
            x_ss5 = x_ss(np.array([self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss6 = x_ss(np.array([self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))
            x_ss7 = x_ss(np.array([-self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss8 = x_ss(np.array([-self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))

            # x_ss = ad * x_ss + bd * u_ss
            # u_ss = bd^-1 * ((I - ad) * x_ss)
            u_ss = lambda idq: -np.linalg.inv(bd) @ ((np.eye(2) - ad) @ idq)

            u_ss1 = u_ss([0, self.i_max])
            u_ss2 = u_ss([self.i_max, 0])
            u_ss3 = u_ss([0, -self.i_max])
            u_ss4 = u_ss([-self.i_max, 0])
            u_ss5 = u_ss([self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss6 = u_ss([self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])
            u_ss7 = u_ss([-self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss8 = u_ss([-self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])

            if plot_current:
                v_d = np.linspace(-1, 1, 1000)
                v_q = np.sqrt(1 - np.power(v_d, 2))
                vdq = self.vdq_max * np.array([v_d, v_q])
                x_ss_data_pos = x_ss(vdq)
                vdq = self.vdq_max * np.array([v_d, -v_q])
                x_ss_data_neg = x_ss(vdq)
                id = np.concatenate((x_ss_data_pos[0], x_ss_data_neg[0]), 0)
                iq = np.concatenate((x_ss_data_pos[1], x_ss_data_neg[1]), 0)
                plt.plot(id, iq, label="Current by voltage limitation")
                id = np.linspace(-1, 1, 1000)
                iq = np.sqrt(1 - np.power(id, 2))
                id_circle = self.i_max * np.concatenate((id, id), 0)
                iq_circle = self.i_max * np.concatenate((iq, -iq), 0)
                plt.plot(id_circle, iq_circle, label="Maximum current circle")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                           fancybox=True, shadow=True, ncol=2)
                plt.title("Discrete state-space model")
                plt.show()

class PlotTest():
    def __init__(self):
        return

    def plot_single_phase(self, idx, observations, actions, reward, env_name, reward_type):
        plt.clf()
        plt.suptitle(f"Reward: {reward_type}\n")
        # Plot State
        ax = plt.subplot(131)
        ax.set_title("State vs step")
        ax.plot(observations, label=['I', 'Iref'])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=2, fancybox=True, shadow=True)
        # Plot action
        ax = plt.subplot(132)
        ax.set_title("Action vs step")
        ax.plot(actions, label=['V'])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=2, fancybox=True, shadow=True)
        # Plot reward
        ax = plt.subplot(133)
        ax.set_title("Reward vs step")
        ax.plot(reward)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        plt.savefig(f"plots/{env_name}_{idx}.pdf", bbox_inches='tight')
        plt.pause(0.001)  # pause a bit so that plots are updated



    def plot_three_phase(self, idx, observations, actions, reward, env_name, reward_type, model_name, speed=None):
        fig = plt.figure(figsize=(12, 4))  # 明确使用一个 figure 对象
        
        if speed is not None:
            fig.suptitle(f"Reward: {reward_type}\nSpeed = {speed} [rad/s]")

        # Plot State
        ax1 = fig.add_subplot(131)
        ax1.set_title("State vs step")
        ax1.plot(observations, label=['Id', 'Iq', 'Idref', 'Iqref'])
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3),
                ncol=2, fancybox=True, shadow=True)

        # Plot Action
        ax2 = fig.add_subplot(132)
        ax2.set_title("Action vs step")
        ax2.plot(actions, label=['Vd', 'Vq'])
        ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3),
                ncol=2, fancybox=True, shadow=True)

        # Plot Reward
        ax3 = fig.add_subplot(133)
        ax3.set_title("Reward vs step")
        ax3.plot(reward)

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 留出 suptitle 空间
        plt.savefig(f"plots/{env_name}_{model_name}_{idx}.pdf", bbox_inches='tight')
        plt.close(fig)  # ✅ 关键：关闭当前 figure