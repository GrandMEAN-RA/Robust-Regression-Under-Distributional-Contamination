
import numpy as np
import pandas as pd
from scipy import stats
from writer import write_file
from setup_tools import progress
import terminal_manager as tm

base_mse_alpha = 0.0
base_mse_beta = 0.0
base_s_sqd_beta = 0.0
base_s_sqd_alpha = 0.0

def sim_analysis(X: np.ndarray,
                 y: np.ndarray,
                 alpha: np.ndarray, 
                 beta: np.ndarray, 
                 true_alpha: float, 
                 true_beta: float, 
                 std_err_alpha,
                 std_err_beta,
                 method: str, 
                 cell: str, 
                 n: 32,
                 flag: str,
                 dist_type: str = None
                 )-> None:

    global base_mse_alpha, base_mse_beta, base_s_sqd_beta, base_s_sqd_alpha

    t_crit = stats.t.ppf(1-0.05/2, df=n-1)

    avg_beta = np.mean(beta) if flag == 'simulation' else beta
    avg_alpha = np.mean(alpha) if flag == 'simulation' else alpha
    
    bias_alpha = avg_alpha - true_alpha
    bias_beta = avg_beta - true_beta

    if flag == 'simulation':
        var_alpha = np.var(alpha, ddof=1)
        var_beta = np.var(beta, ddof=1)

        mse_alpha = np.mean((alpha - true_alpha)**2)
        mse_beta = np.mean((beta - true_beta)**2)

        mcse_alpha = (var_alpha/alpha.shape[0])**0.5
        mcse_beta = (var_beta/beta.shape[0])**0.5

        beta_dev_sqd = (beta - true_beta)**2
        alpha_dev_sqd = (alpha - true_alpha)**2

        s_sqd_beta = np.sum(beta_dev_sqd - mse_beta)**2/(beta.shape[0]-1)
        s_sqd_alpha = np.sum(alpha_dev_sqd - mse_alpha)**2/(alpha.shape[0]-1)

        mcse_mse_beta = (s_sqd_beta/beta.shape[0])**0.5
        mcse_mse_alpha = (s_sqd_alpha/alpha.shape[0])**0.5
        
        # Find the OLS estimates for comparison
        EPSILON = 1e-7
        if method in ['OLS', 'OLSE', 'ols', 'OLSE']:
            base_s_sqd_beta = s_sqd_beta
            base_s_sqd_alpha = s_sqd_alpha
            base_mse_beta = mse_beta
            base_mse_alpha = mse_alpha
                    
        relative_mse_beta = mse_beta / base_mse_beta if abs(base_mse_beta) > EPSILON else mse_beta
        relative_mse_alpha = mse_alpha / base_mse_alpha if abs(base_mse_alpha) > EPSILON else mse_alpha

        cov_mse_alpha = np.sum(s_sqd_beta*base_s_sqd_beta)/(beta.shape[0]-1) 
        cov_mse_beta = np.sum(s_sqd_alpha*base_s_sqd_alpha)/(alpha.shape[0]-1)

        mcse_rmse_beta = relative_mse_beta * ((cov_mse_beta/(mse_beta*base_mse_beta)) + (s_sqd_beta/mse_beta**2) + (base_s_sqd_beta/base_mse_beta**2))**0.5 if (abs(base_mse_beta) > EPSILON and abs(mse_beta) > EPSILON) else 0
        mcse_rmse_alpha = relative_mse_alpha * ((cov_mse_alpha/(mse_alpha*base_mse_alpha)) + (s_sqd_alpha/mse_alpha**2) + (base_s_sqd_alpha/base_mse_alpha**2))**0.5 if (abs(base_mse_alpha) > EPSILON and abs(mse_alpha) > EPSILON) else 0

        repl_LCL_beta = beta - t_crit * std_err_beta/n**0.5
        repl_UCL_beta = beta + t_crit * std_err_beta/n**0.5

        contained_beta = 0
        for i in range(beta.shape[0]):
            if repl_LCL_beta[i,:] <= beta[i,:] <= repl_UCL_beta[i,:]:
                contained_beta += 1
        cov_prob_beta = contained_beta / beta.shape[0]

        mcse_ci_coverage_beta = ((0.05*0.95)/ beta.shape[0])**0.5

        repl_LCL_alpha = alpha - t_crit * std_err_alpha/n**0.5
        repl_UCL_alpha = alpha + t_crit * std_err_alpha/n**0.5

        contained_alpha = 0
        for i in range(alpha.shape[0]):
            if repl_LCL_alpha[i,:] <= beta[i,:] <= repl_UCL_alpha[i,:]:
                contained_alpha += 1
        cov_prob_alpha = contained_alpha / alpha.shape[0]

        mcse_ci_coverage_alpha = ((0.05*0.95)/ alpha.shape[0])**0.5

        repl_CI_width_beta = repl_UCL_beta - repl_LCL_beta
        repl_CI_width_alpha = repl_UCL_alpha - repl_LCL_alpha

        LCL_beta = avg_beta - t_crit * (var_beta/n)**0.5
        UCL_beta = avg_beta + t_crit * (var_beta/n)**0.5

        LCL_alpha = avg_alpha - t_crit * (var_alpha/n)**0.5
        UCL_alpha = avg_alpha + t_crit * (var_alpha/n)**0.5

    elif flag == 'empirical':
        ybar = np.mean(y)
        ycap = alpha + beta*X
        yres = y - ycap
        var_res = np.var(yres)
        
        bias_model = ybar - np.mean(ycap)
        mse_model = var_res + (bias_model**2)

        LCL_beta = avg_beta - t_crit * std_err_beta
        UCL_beta = avg_beta + t_crit * std_err_beta

        LCL_alpha = avg_alpha - t_crit * std_err_alpha
        UCL_alpha = avg_alpha + t_crit * std_err_alpha

    # Store results for table construction
    if flag == 'simulation':
        final_table_beta = {
        'Sample Size': [n],   
        'Simulation Cell': [cell],
        'Distribution': [dist_type],
        'Method': [method],
        'Beta': [round(avg_beta,3)],
        'MCSE (Beta)': [round(mcse_beta,3)],
        'Bias (Beta)': [round(bias_beta,3)],
        'MSE (Beta)': [round(mse_beta,3)],
        'MCSE (MSE Beta)': [round(mcse_mse_beta,3)],
        'Relative MSE (Beta)': [round(relative_mse_beta,3)],
        'MCSE (RMSE Beta)': [round(mcse_rmse_beta,3)],
        'LCL (Beta)': [round(LCL_beta,3)],
        'UCL (Beta)': [round(UCL_beta,3)],
        'Coverage Prob (Beta)': [round(cov_prob_beta,3)],
        'MCSE (Coverage Prob)': [round(mcse_ci_coverage_beta,3)],
        '|||': ['|||'],
        }

        final_table_alpha = {
        'Sample Size': [n],   
        'Simulation Cell': [cell],
        'Distribution': [dist_type],
        'Method': [method],
        'Alpha': [round(avg_alpha,3)],
        'MCSE (Alpha)': [round(mcse_alpha,3)],
        'Bias (Alpha)': [round(bias_alpha,3)],
        'MSE (Alpha)': [round(mse_alpha,3)],
        'MCSE (MSE Alpha)': [round(mcse_mse_alpha,3)],
        'Relative MSE (Alpha)': [round(relative_mse_alpha,3)],
        'MCSE (RMSE Alpha)': [round(mcse_rmse_alpha,3)],
        'LCL (Alpha)': [round(LCL_alpha,3)],
        'UCL (Alpha)': [round(UCL_alpha,3)],
        'Coverage Prob (Alpha)': [round(cov_prob_alpha,3)],
        'MCSE (Coverage Prob)': [round(mcse_ci_coverage_alpha,3)]
        }
        
        # Convert to DataFrame for easy visualization
        sim_results_beta = pd.DataFrame(final_table_beta).reset_index(drop=True)
        sim_results_alpha = pd.DataFrame(final_table_alpha).reset_index(drop=True)

        sim_results = pd.concat([sim_results_alpha, sim_results_beta], axis = 1)
    
    if flag == 'empirical':
        final_table = {
            'Sample Size': [n],   
            'Data Cell': [cell],
            'Method': [method],
            'Alpha': [round(avg_alpha,3)],
            'Bias (Alpha)': [round(bias_alpha,3)],
            'Std. Error (Alpha)': [round(std_err_alpha,3)],
            'LCL (Alpha)': [round(LCL_alpha,3)],
            'UCL (Alpha)': [round(UCL_alpha,3)],
            'Beta': [round(avg_beta,3)],
            'Bias (Beta)': round(bias_beta,3),
            'Std. Error (Beta)': [round(std_err_beta,3)],
            'LCL (Beta)': [round(LCL_beta,3)],
            'UCL (Beta)': [round(UCL_beta,3)],
            'MSE (Model)': round(mse_model,3),
            'Bias (Model)': round(bias_model,3),
        }

        sim_results = pd.DataFrame(final_table).reset_index(drop=True)

    write_file(sim_results,
               method, 
               cell, 
               flag,  
               n,
               dist_type)

    info = f'{tm.bold}{tm.cyan}Simulation cell n = {n} {cell} Model {dist_type} {method} has been succefully completed!{tm.reset}' if flag == 'simulation' else f'{tm.bold}{tm.cyan}Simulation cell n = {n} Data with {cell} {method} has been succefully completed!{tm.reset}'

    progress_info = {'Status':'analysis',
                    'PROGRESS':info
                    }
    progress(**progress_info)

    # Clear variables to free memory
    final_table_alpha = final_table_beta = sim_results_beta = sim_results_alpha = sim_results = None
