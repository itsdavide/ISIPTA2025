#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration code for the paper:
D. Petturiti and B. Vantaggi. 
Dynamic $\alpha$-DS mixture pricing in a market with bid-ask spreads. 
ISIPTA 2025.

The core requires the library pyswarm (https://pythonhosted.org/pyswarm/).
"""

from pyswarm import pso
import pandas as pd
import math as m
import numpy as np
import matplotlib.pyplot as plt

ticker = 'AMZN'

# Maturity in business days
maturities = [
    ('2025-02-21', 20, 'green'),
    ('2025-03-21', 40, 'darkorange'),
    ('2025-05-16', 80, 'blueviolet')
]

# Risk-free return obtained from a 1 month US T-bill
R = (1 + 0.0445)**(1/250)

plt.figure(figsize=(6,4))
plt.title(ticker + ' Optimal Root Mean Squared Error')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'Optimal $\mathrm{RMSE}$')

alphas_min = []
params_min = []

for (maturity, T, color) in maturities:
    # Reset the seed
    np.random.seed(12345)

    file_calls = ticker + '/' + ticker + '_calls_' + maturity + '.csv'
    file_puts = ticker + '/' + ticker + '_puts_' + maturity + '.csv'
        
    # Load stock calls
    STOCK_calls = pd.read_csv('./datasets/t_2025-01-24/' + file_calls)[['strike','bid','ask']]
        
    # Insert the stock as a degenerate call with strike 0
    S_0_bid = 222.92
    S_0_ask = 246.98
    STOCK_calls.loc[len(STOCK_calls)] = [0, S_0_bid, S_0_ask]
        
    # Load stock puts
    STOCK_puts = pd.read_csv('./datasets/t_2025-01-24/' + file_puts)[['strike','bid','ask']]
        
    # Get indices of S_T values
    I_S_T_vals = list(range(T+1))
        
    # Get indices of calls and puts
    I_calls = list(range(len(STOCK_calls)))
    print('# calls:', len(I_calls))
    I_puts = list(range(len(STOCK_puts)))
    print('# puts:', len(I_puts))
    N = len(I_calls) + len(I_puts)
        
    C_0_bid = STOCK_calls['bid']
    C_0_ask = STOCK_calls['ask']
    K_C_T = STOCK_calls['strike']
    P_0_bid = STOCK_puts['bid']
    P_0_ask = STOCK_puts['ask']
    K_P_T = STOCK_puts['strike']
    
    def E(x):
        u = x[0]
        b_d = x[1]
        
        d = 1 / u
        
        b_u = (R - d) / (u - d)
        
        
        E = 0
        
        g_alpha = alpha * b_u + (1 - alpha) * (1 - b_d)
        
        for i in I_calls:
            C_alpha_th = alpha * C_0_bid[i] + (1 - alpha) * C_0_ask[i]
            C_alpha_DS = (1 / (R**T)) * sum(m.comb(T, k) * g_alpha**k * (1 - g_alpha)**(T - k) * max(u**k * d**(T - k) * S_0_bid - K_C_T[i], 0) for k in I_S_T_vals)
            E += (C_alpha_th - C_alpha_DS)**2
        
        d_alpha = alpha * (1 - b_d) + (1 - alpha) * b_u
       
        for i in I_puts:
            P_alpha_th = alpha * P_0_bid[i] + (1 - alpha) * P_0_ask[i]
            P_alpha_DS =  (1 / (R**T)) * sum(m.comb(T, k) * d_alpha**k * (1 - d_alpha)**(T - k) * max(K_P_T[i] - u**k * d**(T - k) * S_0_bid, 0) for k in I_S_T_vals)
            E += (P_alpha_th - P_alpha_DS)**2
              
        return E
    
    
    def con(x):
        u = x[0]
        b_d = x[1]
        return [1 - (R - (1 / u)) / (u - (1 / u)) - b_d]
    
    lb = [R + 0.0005, 0.0005]
    ub = [1.5, 1]
    
    
    alphas = np.arange(0, 1.02, 0.02)
    
    errors = []
    
    params = []
    
    for alpha in alphas:
        print(f'\n\n*** alpha: {alpha} ***\n')
        # OPTIMIZATION WITH PSO
        
        xopt, fopt = pso(E, lb, ub, f_ieqcons=con, maxiter=100)
        
        print(xopt)
        print('Error:', fopt)
        print('Error:', E(xopt))
        
        # Compute the RMSE
        errors.append(np.sqrt((1 / N) * fopt))
        
        (u, b_d) = xopt
        d = 1 / u
        b_u = (R - d) / (u - d)
        print('u = ', u)
        print('d = ', d)
        print('b_u = ', b_u)
        print('b_d = ', b_d)
        print('b_u + b_d = ', b_u + b_d)
        
        params.append((u, d, b_u, b_d))
    
    
    df_error = pd.DataFrame(np.array(errors))
    df_error.to_csv(ticker + '_RMSE_' + maturity + '.csv')
    
    plt.plot(alphas, errors, color=color, label='$T=$' + maturity)
    
    i_min = np.array(errors).argmin()
    alpha_min = alphas[i_min]
    error_min = errors[i_min]
    
    plt.plot([alpha_min], [error_min], marker='o', color=color)
    print('alpha_min:', alpha_min)
    
    alphas_min.append(alpha_min)
    params_min.append(params[i_min])
    
    
plt.legend()
plt.show()
plt.savefig(ticker + '_RMSE_d_constrained.png', dpi=300)

print('\n************************')
print('alphas_min', alphas_min)
print('params_min', params_min)
print('************************\n')