class LimitedQueueSystemSolver(AbstractSystemSolver):
    
    QUEUE_SIZE = 15
    WORKER_SIZE = 15
    
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
                'savefig': 'task13',
                'show_plot': False,
            }
        }
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_x = [[1]]
        self.queue_probs = [[]]
        self.queue_means = [[]]
        self.queue_busy_coeffs = [[]]
        self.failure_probabilities = [[]]
        self.means = [[]]
        self.busy_coeffs = [[]]
        
        self._curr_queue_x = None
    
    def solve(self, step = 1, *args, **kwargs):
        super().solve(*args, **kwargs)
        self.queue_x[self.x[-1] - 1].append(self.queue_x[self.x[-1] - 1][-1] + 1)
        
        while self.x[-1] < 15:
            while self.queue_x[self.x[-1] - 1][-1] <= self.QUEUE_SIZE:
                self.cycle()
                self.queue_x[self.x[-1] - 1].append(self.queue_x[self.x[-1] - 1][-1] + 1)
                
            self.x.append(self.x[-1] + step)
            self.queue_x.append([1])
            self.queue_probs.append([])
            self.queue_means.append([])
            self.queue_busy_coeffs.append([])
            self.failure_probabilities.append([])
            self.means.append([])
            self.busy_coeffs.append([])
    
    def _calc_pn_coeffs(self, n):
        pn_worker_coeffs = [(self.lambda1 ** pos) / (factorial(pos) * (self.mu1 ** pos)) for pos in range(0, n + 1)]
        pn_queue_coeffs = [((self.lambda1 / (n * self.mu1)) ** pos) * pn_worker_coeffs[-1] for pos in range(1, self.queue_x[self.x[n - 1] - 1][-1])]
        self._pn_coeffs = pn_worker_coeffs + pn_queue_coeffs

    def _queue_mean(self):
        self._curr_queue_mean = self._exp(range(1, self.QUEUE_SIZE + 1), self._prob_values[self.x[-1] + 1:])
        self.queue_means[self._curr_x - 1].append(self._curr_queue_mean)
        
    def _queue_prob(self):
        self.queue_probs[self._curr_x - 1].append(sum(self._prob_values[self.x[-1] + 1:]))
        
    def _queue_busy_coeff(self):
        self.queue_busy_coeffs[self._curr_x - 1].append(self._curr_queue_mean / self._curr_queue_x)
    
    def _find_pn_value(self):
        self._prob_values = [el * self._p0 for el in self._pn_coeffs]
    
    def _failure_probability(self):
        self.failure_probabilities[self._curr_x - 1].append(self._prob_values[-1])
    
    def _mean(self):
        self._curr_mean = self._exp(range(0, self.x[-1] + 1), self._prob_values[:self.x[-1] + 1])
        self.means[self._curr_x - 1].append(self._curr_mean)
    
    def _busy_coeff(self):
        self.busy_coeffs[self._curr_x - 1].append(self._curr_mean / self._curr_x)
    
    def cycle(self, *args, **kwargs):
        super().cycle(*args, **kwargs)
        self._curr_queue_x = self.queue_x[self._curr_x - 1][-1]
        
        self._queue_mean()
        self._queue_prob()
        self._queue_busy_coeff()
        
    def plot(self, graphs: dict = _PLOT_PARAMS, smoothing = 250, interp_model_kind = 'linear', *args, **kwargs):
        # self._convert_to_np_arr()
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
            
            X = self.__dict__.get(absciss, None)
            
            if X is None:
                continue
            
            for ordinate, options in ordinates.items():
                
                if ordinate == 'options':
                    continue
                    
                Y = self.__dict__.get(ordinate, None)
    
                if Y is None:
                    continue

                for queue_x in range(self.QUEUE_SIZE):
                    print(Y[-1])
                    print(ordinate)
                    Y_ = Y[queue_x]
                    Y_interp_model = interp1d(X, Y_, kind=interp_model_kind)
                
                    X_ = np.linspace(self.x[0], self.x[-1], smoothing)
                    Y_ = Y_interp_model(X_)
                    
                    plt.plot(X_, Y_)
                    
                    post_actions = ordinates.get('options', None)
            
                    if post_actions is not None:
                
                        savefig = post_actions.get('savefig', None)
                        show_plot = post_actions.get('show_plot', False)
                
                        if savefig is not None:
                            plt.savefig('1')
                
                        if show_plot:
                            plt.show()
            
                        plt.close()