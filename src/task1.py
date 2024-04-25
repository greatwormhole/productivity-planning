import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from math import factorial, fabs
from functools import reduce
from sympy.ntheory import factorint

DEBUG = True

if DEBUG:
    from time import sleep

class AbstractSystemSolver(object):
    
    def __init__(self, *args, **kwargs):
        self.x = [1]
        self.failure_probabilities = []
        self.means = []
        self.busy_coeffs = []
        
        self._pn_coeffs = []
        self._p0 = None
        self._prob_values = []
        
        self._curr_x = None
        self._curr_mean = None
            
        if 'tc' in kwargs.keys():
            self.lambda1 = 1 / kwargs['tc']
        
        if 'ts' in kwargs.keys():
            self.mu1 = 1 / kwargs['ts']
            
        if 'tw' in kwargs.keys():
            self.nu1 = 1 / kwargs['tw']
    
    def solve(self, *args, **kwargs):
        self.cycle()
    
    def plot(self, graphs: dict, smoothing, interp_model_kind, *args, **kwargs):
        self._convert_to_np_arr()
        
        def set_axes_params(ax, title, xlabel, ylabel, ylims,
                            fontsize=kwargs.get('fontsize', 16),
                            color=kwargs.get('color', 'black'),
                            pad=kwargs.get('pad', 10)):
            
            font_params = {
                'fontsize': fontsize,
                'color': color,
            }
            
            ax.grid(visible=True, which='both')
            ax.set_xlim(self.x[0], self.x[-1])
            ax.set_xlabel(xlabel, **font_params)
            ax.set_xticks(range(self.x[0], self.x[-1] + 1))
            ax.set_ylim(ylims)
            ax.set_ylabel(ylabel, **font_params)
            ax.set_title(title, pad=pad, **font_params)
            
        for absciss, ordinates in graphs.items():
            fig = plt.figure(figsize=kwargs.get('figsize', (16, 16)))
            grid_num = sum([1 for el in ordinates.keys() if el != 'options'])
            
            _prime_divisors = factorint(grid_num)
            prime_list = []
            for number, power in _prime_divisors.items():
                prime_list += [number] * power
            prime_list.insert(0, 1)
            
            while len(prime_list) != 2:
                min_key = prime_list.pop(0)
                prime_list[0] *= min_key
                prime_list.sort()
                
            ncols, nrows = prime_list
            spec = fig.add_gridspec(nrows, ncols)
            
            counter = 0
            
            X = self.__dict__.get(absciss, None)
            
            if X is None:
                continue
            
            for ordinate, options in ordinates.items():
                
                if ordinate == 'options':
                    continue
                
                ax = fig.add_subplot(spec[counter // ncols, counter % ncols])
                Y = self.__dict__.get(ordinate, None)
                
                if Y is None:
                    continue
                
                Y_interp_model = interp1d(X, Y, kind=interp_model_kind)
                
                X_ = np.linspace(self.x[0], self.x[-1], smoothing)
                Y_ = Y_interp_model(X_)
                
                ax.plot(X_, Y_)
                set_axes_params(ax, ylims=(round(min(Y_) - 0.05, 1), round(max(Y_) + 0.05, 1)), **options)
                
                counter += 1

            post_actions = ordinates.get('options', None)
            
            if post_actions is not None:
                
                savefig = post_actions.get('savefig', None)
                show_plot = post_actions.get('show_plot', False)
                
                if savefig is not None:
                    plt.savefig(savefig)
                
                if show_plot:
                    plt.show()
            
            plt.close()
    
    def solve_and_plot(self, *args, **kwargs):
        self.solve(*args, **kwargs)
        self.plot(*args, **kwargs)
    
    def cycle(self, *args, **kwargs):
        self._curr_x = self.x[-1]
        self._curr_mean = None
        
        self._calc_pn_coeffs(self._curr_x)
        self._calc_p0()
        self._find_pn_value()
        self._failure_probability() 
        self._mean()
        self._busy_coeff()
    
    def _calc_pn_coeffs(self, n):
        raise NotImplementedError

    def _calc_p0(self):
        self._p0 = 1 / np.sum(self._pn_coeffs)
    
    def _find_pn_value(self):
        self._prob_values = [el * self._p0 for el in self._pn_coeffs]
    
    def _failure_probability(self):
        self.failure_probabilities.append(self._prob_values[-1])
    
    @staticmethod
    def _exp(x, y):
        return sum([el[0] * el[1] for el in zip(x, y)])
    
    def _mean(self):
        self._curr_mean = self._exp(range(0, self.x[-1] + 1), self._prob_values[:self.x[-1] + 1])
        self.means.append(self._curr_mean)
    
    def _busy_coeff(self):
        self.busy_coeffs.append(self._curr_mean / self._curr_x)
        
    def _convert_to_np_arr(self):
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                self.__dict__[key] = np.array(self.__dict__[key])

class QueuelessSystemSolver(AbstractSystemSolver):
    
    _PLOT_PARAMS = {
        'x': {
            'failure_probabilities': {
                'title': 'Зависимость вероятности отказа от числа операторов',
                'xlabel': 'Количество операторов',
                'ylabel': 'Вероятность отказа',
            },
            'means': {
                'title': 'Зависимость матожидания числа занятых операторов от числа операторов',
                'xlabel': 'Количество операторов',
                'ylabel': 'Матожидание числа занятых операторов',
            },
            'busy_coeffs': {
                'title': 'Зависимость коэффициента загрузки от числа операторов',
                'xlabel': 'Количество операторов',
                'ylabel': 'Значение коэффициента загрузки',
            },
            'options': {
                'savefig': '../media/task11',
                'show_plot': False,
            }
        }
    }
    
    def solve(self, step = 1, threshold = 0.005, *args, **kwargs):
        super().solve(*args, **kwargs)
        
        while self.failure_probabilities[-1] > threshold:
            self.x.append(self.x[-1] + step)
            self.cycle()
    
    def plot(self, graphs: dict = _PLOT_PARAMS, smoothing = 250, interp_model_kind = 'linear', show_plot = True, savefig = None, *args, **kwargs):
        return super().plot(graphs, smoothing, interp_model_kind, show_plot, savefig, *args, **kwargs)
    
    def _calc_pn_coeffs(self, n):
        self._pn_coeffs = [(self.lambda1 ** pos) / (factorial(pos) * (self.mu1 ** pos)) for pos in range(0, n + 1)]
    
class LimitlessQueueSystemSolver(AbstractSystemSolver):
    
    THRESHOLD = 1e-6
    _PLOT_PARAMS = {
        'x': {
            'means': {
                'title': 'Зависимость матожидания числа занятых операторов от числа операторов',
                'xlabel': 'Количество операторов',
                'ylabel': 'Матожидание числа занятых операторов',
            },
            'busy_coeffs': {
                'title': 'Зависимость коэффициента загрузки от числа операторов',
                'xlabel': 'Количество операторов',
                'ylabel': 'Значение коэффициента загрузки',
            },
            'queue_probs': {
                'title': 'Зависимость вероятности существования очереди от числа операторов',
                'xlabel': 'Количество операторов',
                'ylabel': 'Вероятность существования очереди',
            },
            'queue_means': {
                'title': 'Зависимость матожидания длины очереди от числа операторов',
                'xlabel': 'Количество операторов',
                'ylabel': 'Матожидание длины очереди',
            },
            'options': {
                'savefig': '../media/task13',
                'show_plot': False,
            }
        }
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = [2]
        self.queue_means = []
        self.queue_probs = []
        
        self._curr_queue_mean = None
    
    def solve(self, step = 1, limit = 15, *args, **kwargs):
        super().solve(*args, **kwargs)
        
        while self.x[-1] < limit:
            self.x.append(self.x[-1] + step)
            self.cycle()
            
    def plot(self, graphs: dict = _PLOT_PARAMS, smoothing = 250, interp_model_kind = 'linear', show_plot = True, savefig = None, *args, **kwargs):
        return super().plot(graphs, smoothing, interp_model_kind, show_plot, savefig, *args, **kwargs)
    
    def _calc_pn_coeffs(self, n):
        pn_worker_coeffs = [(self.lambda1 ** pos) / (factorial(pos) * (self.mu1 ** pos)) for pos in range(0, n + 1)]
        pos = 2
        pn_queue_coeffs = [((self.lambda1 / (n * self.mu1)) ** (pos - 1)) * pn_worker_coeffs[-1],
                            ((self.lambda1 / (n * self.mu1)) ** pos) * pn_worker_coeffs[-1]]
        while abs(pn_queue_coeffs[-1] - pn_queue_coeffs[-2]) > self.THRESHOLD:
            pos += 1
            pn_queue_coeffs.append(((self.lambda1 / (n * self.mu1)) ** pos) * pn_worker_coeffs[-1])
        self._pn_coeffs = pn_worker_coeffs + pn_queue_coeffs
    
    def _mean(self):
        self._curr_mean = self._exp(range(0, self.x[-1] + 1), self._prob_values[:self.x[-1] + 1])
        self._curr_mean += self.x[-1] * sum(self._prob_values[self.x[-1] + 1:])
        self.means.append(self._curr_mean)
    
    def _queue_mean(self):
        self._curr_queue_mean = self._exp(range(0, len(self._prob_values[self.x[-1]:]) + 1), self._prob_values[self.x[-1]:])
        self.queue_means.append(self._curr_queue_mean)
        
    def _queue_prob(self):
        self.queue_probs.append(sum(self._prob_values[self.x[-1] + 1:]))
        
    def cycle(self, *args, **kwargs):
        super().cycle(*args, **kwargs)
        
        self._queue_mean()
        self._queue_prob()
        
class LimitlessQueueSystemWithDisbandSolver(LimitlessQueueSystemSolver):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = [1]
        self._PLOT_PARAMS['x']['options']['savefig'] = '../media/task14'
      
    def _calc_pn_coeffs(self, n):
        pn_worker_coeffs = [(self.lambda1 ** pos) / (factorial(pos) * (self.mu1 ** pos)) for pos in range(0, n + 1)]
        pos = 2
        pn_queue_coeffs = [(self.lambda1 ** (pos - 1)) / (reduce(lambda i, j: i * j, [
                self.x[-1] * self.mu1 + k * self.nu1 for k in range(1, pos)
            ])) * pn_worker_coeffs[-1],
            (self.lambda1 ** pos) / (reduce(lambda i, j: i * j, [
            self.x[-1] * self.mu1 + k * self.nu1 for k in range(1, pos + 1)
            ])) * pn_worker_coeffs[-1]]
        while pn_queue_coeffs[-1] > self.THRESHOLD:
            pos += 1
            pn_queue_coeffs.append(
                (self.lambda1 ** pos) / (reduce(lambda i, j: i * j, [
                    self.x[-1] * self.mu1 + k * self.nu1 for k in range(1, pos + 1)
                ])) * pn_worker_coeffs[-1])
        self._pn_coeffs = pn_worker_coeffs + pn_queue_coeffs