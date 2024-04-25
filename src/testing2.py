import matplotlib.pyplot as plt

def _exp(x, y):
    return sum([el[0] * el[1] for el in zip(x, y)])

# task 12

data1 = {
    'tc': 31,
    'ts': 57,
    'tw': 109,
}

lambda1 = 1 / data1['tc']
mu1 = 1 / data1['tw']
alpha = lambda1 / mu1

ns = [i for i in range(2, 16)]
fig = plt.figure(figsize=(16, 16))
spec = fig.add_gridspec(3 , 2)
ax11 = fig.add_subplot(spec[0, 0])
ax12 = fig.add_subplot(spec[0, 1])
ax21 = fig.add_subplot(spec[1, 0])
ax22 = fig.add_subplot(spec[1, 1])
ax31 = fig.add_subplot(spec[2, 0])
ax32 = fig.add_subplot(spec[2, 1])

for m in range(1, 16):
    
    pn_coeffs = [1]
    failure_prob = []
    m_exp_busy = []
    busy_coeffs = []
    queue_prob = []
    m_exp_length = []
    busy_coeff_queue = []
    
    for n in ns:
        for i in range(1, n + 1):
            pn_coeffs.append(pn_coeffs[-1] * alpha / i)
        for i in range(m + 1):
            pn_coeffs.append(pn_coeffs[-1] * alpha / n)
        
        p0 = 1 / sum(pn_coeffs)
        pn = [p0 * i for i in pn_coeffs]
        
        failure_prob.append(pn[-1])
        m_exp_busy.append(_exp(pn, [i for i in range(0, n + 1)] + [n] * m))
        busy_coeffs.append(m_exp_busy[-1] / n)
        queue_prob.append(sum(pn[n + 1:]))
        m_exp_length.append(_exp(pn[n + 1:], range(1, m)))
        busy_coeff_queue.append(m_exp_length[-1] / m)
    
    ax11.plot(ns, failure_prob)
    ax12.plot(ns, m_exp_busy)
    ax21.plot(ns, busy_coeffs)
    ax22.plot(ns, queue_prob)
    ax31.plot(ns, m_exp_length)
    ax32.plot(ns, busy_coeff_queue)
            
plt.show()

font_params = {
    'fontsize': 16,
    'color': 'black',
}

for i in range(5):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot()
    ax.grid(visible=True, which='both')
    ax.plot(ns, Ys[i])
    ax.set_xlim(ns[0], ns[-1])
    ax.set_xlabel('Количество наладчиков', **font_params)
    ax.set_xticks(ns)
    ax.set_ylim(round(min(Ys[i]) - 0.075, round(max(Ys[i])  + 0.075)))
    ax.set_ylabel(y_labels[i], **font_params)
    ax.set_title(titles[i], **font_params, pad=10)
    
    plt.savefig(f'{i}')
    
    plt.close()

# task21

data2 = {
    'N': 39,
    'tc': 121,
    'ts': 42,
}

lambda1 = 1 / data2['tc']
nu1 = 1 / data2['ts']
N = data2['N']
alpha = lambda1 / nu1

m_exp_dis = []
m_exp_waiting = []
p_waiting = []
m_busy = []
busy_coeffs = []
ns = [i for i in range(1, N + 1)]

for n in ns:
    
    pn_coeffs = [1]
    
    for m in range(1, N):
        if m < n:
            pn_coeffs.append(pn_coeffs[-1] * (N - m + 1) / m * alpha)
        else:
            pn_coeffs.append(pn_coeffs[-1] * (N - m + 1) / n * alpha)
    
    p0 = 1 / sum(pn_coeffs)
    pn = [p0 * i for i in pn_coeffs]
    
    m_exp_dis.append(_exp(pn, range(0, N + 1)))
    m_exp_waiting.append(_exp(pn[n + 1:], range(1, N - n)))
    p_waiting.append(sum(pn[n + 1:]))
    m_busy.append(_exp(pn, [i for i in range(0, n + 1)] + [n] * (N - n)))
    busy_coeffs.append(m_busy[-1] / n)
            
Ys = [m_exp_dis, m_exp_waiting, p_waiting, m_busy, busy_coeffs]
titles = [
    'Зависимость математического ожидания числа простаивающих станков от числа наладчиков',
    'Зависимость математического ожидания числа ожидающих станков от числа наладчиков',
    'Зависимость вероятности ожидания обслуживания от числа наладчиков',
    'Зависимость математического ожидания числа занятых наладчиков от числа наладчиков',
    'Зависимость коэффициента занятости наладчиков от числа наладчиков'
]
y_labels = [
    'Математическое ожидание числа простаивающих станков',
    'Математическое ожидание числа ожидающих станков',
    'Вероятность ожидания обслуживания',
    'Математическое ожидание числа занятых наладчиков',
    'Коэффициент занятости наладчиков'
]
font_params = {
    'fontsize': 16,
    'color': 'black',
}

for i in range(5):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot()
    ax.grid(visible=True, which='both')
    ax.plot(ns, Ys[i])
    ax.set_xlim(ns[0], ns[-1])
    ax.set_xlabel('Количество наладчиков', **font_params)
    ax.set_xticks(ns)
    ax.set_ylim(round(min(Ys[i]) - 0.075, round(max(Ys[i])  + 0.075)))
    ax.set_ylabel(y_labels[i], **font_params)
    ax.set_title(titles[i], **font_params, pad=10)
    
    plt.savefig(f'{i}')
    
    plt.close()