import numpy as np
import matplotlib.pyplot as plt
from grow_trees import grow_trees
from propagate_fire import propagate_fire
from complementary_CDF import complementary_CDF


N_values = [16, 32, 64, 128, 256, 512, 1024]
p = 0.01  # prob. of fire propagating
f = 0.2 # prob. of one tree fired ( lightning occurs)
repeats = 10


alpha_results = []
target_num_fires = 300  
num_fires = 0

for N in N_values:

    if N == 1024:
        repeats = 2

    alpha_results_for_N = []
    r = 0 # count repeat times  for debug

    for _ in range(repeats):
            
        forest = np.zeros([N, N])  # Empty forest.
        fire_size = []  # Empty list of fire sizes.
        fire_history = []  # Empty list of fire history.
        
        num_fires = 0
        while num_fires < target_num_fires:

            forest = grow_trees(forest, p)  # Grow new trees.
            Ni, Nj = forest.shape
            p_lightning = np.random.rand()

            if p_lightning < f:  # Lightning occurs.
                i0 = np.random.randint(Ni)
                j0 = np.random.randint(Nj)
                
                fs, forest = propagate_fire(forest, i0, j0) # fs = firesize

                if fs > 0:
                    fire_size.append(fs) 
                    num_fires += 1 
                    
                fire_history.append(fs)
                
            else:
                fire_history.append(0)

            forest[np.where(forest == -1)] = 0
        
        print(f'N = {N}', f'Target of {target_num_fires} fire events reached')

        c_CDF, s_rel = complementary_CDF(fire_size, forest.size)
        min_rel_size = 1e-3
        max_rel_size = 1e-1

        is_min = np.searchsorted(s_rel, min_rel_size)
        is_max = np.searchsorted(s_rel, max_rel_size)

        # Note!!! The linear dependence is between the logarithms
        fit_result = np.polyfit(np.log(s_rel[is_min:is_max]),
                    np.log(c_CDF[is_min:is_max]), 1)
        beta = fit_result[0]
        alpha = 1 - beta
        alpha_results_for_N.append(alpha)
        r  = r + 1
        print(f'Repeat times = {r}')

    alpha_mean = np.mean(alpha_results_for_N )
    alpha_std = np.std(alpha_results_for_N )
    alpha_results.append((alpha_mean, alpha_std))
    print(f'After {r} times repeat, empirical cCDF has an exponent alpha = {alpha_results[-1]}')


# # Note loglog plot!
# plt.loglog(s_rel, c_CDF, ".-", color='k', markersize=5, linewidth=0.5)
# plt.title('Empirical cCDF')
# plt.xlabel('relative size')
# plt.ylabel('c CDF')
# plt.show()

inv_N = 1 / np.array(N_values)
alpha_means = [result[0] for result in alpha_results] 
alpha_errors = [result[1] for result in alpha_results] 

# Q1 Extrapolate results to 1/N : 0 -->> find fit function
inv_N_fit = inv_N[:7]
alpha_means_fit = alpha_means[:7]
fit_result = np.polyfit(inv_N_fit, alpha_means_fit, 1)
a, b = fit_result 
fit_line = a * inv_N + b

xticks_positions = inv_N
xticks_labels = [f'$\\frac{{1}}{{{N}}}$' for N in N_values]  # 使用 LaTeX 格式化分数

plt.figure(figsize=(12, 6))
plt.plot(inv_N, fit_line, 'k--', label=f'Linear fit: α = {a:.4f} (1/N) + {b:.4f}')  #  line of fit funcion
plt.errorbar(inv_N, alpha_means, yerr=alpha_errors, fmt='o', color='cyan', ecolor='black', capsize=5, label='Alpha values with error bars')
plt.xticks(xticks_positions, xticks_labels)
plt.xticks(xticks_positions, xticks_labels, rotation=45, ha='right', fontsize=10)

plt.xlabel('1/N')
plt.ylabel('Exponent α')
plt.title('Dependence of Power-law Exponent on 1/N')
plt.grid(True)
plt.legend()
plt.tight_layout() 
plt.savefig('power_law_exponent_vs_inv_N.png', format='png', dpi=300) 
plt.show()

