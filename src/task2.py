from math import factorial
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from task1 import AbstractSystemSolver

class MultipleSetupersSystemSolver(AbstractSystemSolver):
    
    def __init__(self, N, tc, ts, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.N = N
        self.tc = tc
        self.ts = ts
        
        self.broken_machines_means = []
        self.waiting_machines_means = []
        self.waiting_machines_probs = []
        
    def _calc_pn_coeffs(self, n):
        pn_worker_coeffs = [factorial(self.N)/(factorial(self.N-pos)*factorial(pos)) * (self.tc / self.ts) ** pos for pos in range(0, n + 1)]
        pn_waiting_coeffs = [pn_worker_coeffs[-1] * factorial(self.N - n) / (factorial(self.N - pos) * n ** pos) * (self.tc/self.ts) ** pos for pos in range(1, self.N - n)]
        self._pn_coeffs = pn_worker_coeffs + pn_waiting_coeffs
    
    # def plot(self, smoothing = 250, interp_model_kind='linear', show_plot = True, savefig = None):
    #     self._convert_to_np_arr()
        
    #     def set_axes_params(ax, title, xlabel, ylabel, ylims: tuple, fontsize=16):
    #         font_params = {
    #             'fontsize': fontsize,
    #             'color': 'black',
    #         }
            
    #         ax.grid(visible=True, which='both')
    #         ax.set_xlim(self.x[0], self.x[-1])
    #         ax.set_xlabel(xlabel, font_params)
    #         ax.set_xticks(range(1, self.x[-1] + 1, 2))
    #         ax.set_ylim(ylims)
    #         ax.set_ylabel(ylabel, font_params)
    #         ax.set_title(title, font_params, pad=20)
        
    #     M_interp_model = interp1d(self.x,
    #                             self.means,
    #                             kind=interp_model_kind)
        
    #     BC_interp_model = interp1d(self.x,
    #                             self.busy_coeffs,
    #                             kind=interp_model_kind)
                
    #     X_ = np.linspace(self.x[0], self.x[-1], smoothing)
    #     M_ = M_interp_model(X_)
    #     BC_ = BC_interp_model(X_)

    #     fig = plt.figure(figsize=(16, 16))
    #     spec = fig.add_gridspec(3, 2)
        
    #     ax00 = fig.add_subplot(spec[0, 0])
    #     ax01 = fig.add_subplot(spec[0, 1])
    #     ax10 = fig.add_subplot(spec[1, 0])
    #     ax11 = fig.add_subplot(spec[1, 1])
    #     ax2 = fig.add_subplot(spec[2, :])
        
    #     ax00.plot(X_, M_)
    #     set_axes_params(ax00,
    #                     'Зависимость матожидания числа занятых операторов от числа операторов',
    #                     'Количество операторов',
    #                     'Матожидание числа занятых операторов',
    #                     (0, round(max(M_) + 0.05, 1)),
    #                     fontsize=12)
    #     ax01.plot(X_, BC_)
    #     set_axes_params(ax01,
    #                     'Зависимость коэффициента загрузки от числа операторов',
    #                     'Количество операторов',
    #                     'Значение коэффициента загрузки',
    #                     (0, round(max(BC_) + 0.05, 1)),
    #                     fontsize=12)
    #     ax10.plot(X_, BC_)
    #     set_axes_params(ax10,
    #                     'Зависимость коэффициента загрузки от числа операторов',
    #                     'Количество операторов',
    #                     'Значение коэффициента загрузки',
    #                     (0, round(max(BC_) + 0.05, 1)),
    #                     fontsize=12)
    #     ax11.plot(X_, BC_)
    #     set_axes_params(ax11,
    #                     'Зависимость коэффициента загрузки от числа операторов',
    #                     'Количество операторов',
    #                     'Значение коэффициента загрузки',
    #                     (0, round(max(BC_) + 0.05, 1)),
    #                     fontsize=12)
    #     ax2.plot(X_, BC_)
    #     set_axes_params(ax2,
    #                     'Зависимость коэффициента загрузки от числа операторов',
    #                     'Количество операторов',
    #                     'Значение коэффициента загрузки',
    #                     (0, round(max(BC_) + 0.05, 1)),
    #                     fontsize=12)
        
    #     if savefig is not None:
    #         plt.savefig(savefig)
        
    #     if show_plot:
    #         plt.show()
            
    #     plt.close()
    
    