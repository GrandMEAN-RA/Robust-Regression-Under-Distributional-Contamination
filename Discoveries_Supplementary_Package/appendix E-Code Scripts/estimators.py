
import time
import numpy as np
import pandas as pd
from analyzer import sim_analysis
from setup_tools import progress, batcher
from writer import write_file
import terminal_manager as tm

# 3. Define OLS function to estimate alpha and beta from simulated data
def OLS(X: np.ndarray, 
        y: np.ndarray,
        cell: str, 
        n: int = None,
        distribution: str = None,
        available_ram: float = None
        ):
    
    if y.shape != X.shape:
        raise ValueError(f"Shape mismatch: X shape {X.shape} and Y shape {y.shape} must be the same.")
    
    batch_size, total_batches, alpha, beta, std_err_alpha, std_err_beta = batcher(X.shape[0], n, available_ram, cell, distribution, 'OLSE')
    
    # initialize progress update generator
    batch_count = 0
    
    start_time = time.perf_counter()
    for i in range(0, X.shape[0], int(batch_size)):
        X_batch = X[i:i+int(batch_size)]
        y_batch = y[i:i+int(batch_size)]

        n_sims, n_points = X_batch.shape
        if X_batch.shape != y_batch.shape or n_points < 2:
            raise ValueError("X and Y must have the same shape and at least 2 points.")

        # 1. Calculate means across columns (axis=1) for each simulation
        mean_X = np.mean(X_batch, axis=1, keepdims=True)
        mean_Y = np.mean(y_batch, axis=1, keepdims=True)
                    
        dev_x = X_batch - mean_X
        dev_y = y_batch - mean_Y

        dev_x_squared = dev_x**2
        dev_xy = dev_x * dev_y

        beta[i:i+int(batch_size)] = np.sum(dev_xy, axis=1, keepdims=True) / np.sum(dev_x_squared, axis=1, keepdims=True)
        alpha[i:i+int(batch_size)] = mean_Y - beta[i:i+int(batch_size)] * mean_X
        
        sse_res = np.sum(np.float64(y_batch)**2, axis=1, keepdims=True) - beta[i:i+int(batch_size)]*np.sum(np.float64(dev_xy), axis=1, keepdims=True)
        std_err_res = (sse_res/(n_points - 2))**0.5
        std_err_beta[i:i+int(batch_size)] = std_err_res / np.sum(dev_x_squared, axis=1, keepdims=True)**0.5

        sse_alpha_num = np.mean(X_batch, axis=1, keepdims=True)**2
        sse_alpha_denum = np.var(X_batch, axis=1, keepdims=True) * n_points
        sse_alpha = (sse_alpha_num / sse_alpha_denum + (1/n_points))**0.5
        std_err_alpha[i:i+int(batch_size)] = std_err_res *  sse_alpha

        # Simulation progress update
        batch_count += 1 if batch_count < total_batches else total_batches

        progress_info = {'Status':'estimation',
                        'Batch_count':batch_count,
                        'Total_batches':total_batches
                        }
        progress(**progress_info)

        X_batch = y_batch = dev_x = dev_x_squared = dev_y = dev_xy = None  # Clear batch variables to free memory
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Clear variables to free memory
    dev_x = dev_x_squared = dev_y = dev_xy = None

    return alpha, std_err_alpha, beta, std_err_beta, elapsed_time

# 4. Define Theil's pairwise median function to estimate alpha and beta from simulated data. 
    # This vectorized version is significantly faster for Monte-Carlo simulations.

def Theil_pm_vec(X: np.ndarray, 
             y: np.ndarray, 
             cell: str,
             n: np.int32 = None,
             distribution: str = None,
             available_ram: float = None
             ):

    if y.shape != X.shape:
        raise ValueError(f"Shape mismatch: X shape {X.shape} and Y shape {y.shape} must be the same.")
            
    batch_size, total_batches, alpha, beta, std_err_alpha, std_err_beta = batcher(X.shape[0], n, available_ram, cell, distribution, 'Theil')

    # initialize progress update generator
    batch_count = 0

    start_time = time.perf_counter()

    for i in range(0, X.shape[0], int(batch_size)):
        X_batch = X[i:i+int(batch_size)]
        y_batch = y[i:i+int(batch_size)]

        # 1. Expand dimensions to find all pairwise differences per simulation
        # Shapes will become: [n_sims, n_points, n_points]
        y_diff = y_batch[:, :, np.newaxis] - y_batch[:, np.newaxis, :]
        x_diff = X_batch[:, :, np.newaxis] - X_batch[:, np.newaxis, :]

        # 2. Extract upper triangle indices (pairs where j > i)
        n_points = X_batch.shape[1]
        triu_rows, triu_cols = np.triu_indices(n_points, k=1)
                    
        # Slice the pairwise differences to get 1D arrays of pairs per simulation
        # Shape changes to: [n_sims, n_pairs]
        delta_y = y_diff[:, triu_rows, triu_cols]
        delta_x = x_diff[:, triu_rows, triu_cols]
        delta_xy = np.float64(delta_x) * np.float64(delta_y)
        delta_xsqd = np.float64(delta_x)**2

        # 3. Calculate slopes. Replace division by zero with NaN
        # np.where handles variable delta_x across different simulations
        slopes = np.where(delta_x != 0, delta_y / delta_x, np.nan)
                    
        # 4. Calculate medians per simulation (axis=1), ignoring NaNs
        beta[i:i+int(batch_size)] = np.nanmedian(slopes, axis=1, keepdims=True)
                        
        # 5. Conover's Intercept per simulation
        median_Y = np.median(y_batch, axis=1, keepdims=True)
        median_X = np.median(X_batch, axis=1, keepdims=True)
        alpha[i:i+int(batch_size)] = median_Y - beta[i:i+int(batch_size)] * median_X

        sse_res = np.sum(np.float64(y_batch)**2, axis=1, keepdims=True) - beta[i:i+int(batch_size)]*np.sum(delta_xy, axis=1, keepdims=True)
        std_err_res = (abs(sse_res)/(n_points - 2))**0.5
        std_err_beta[i:i+int(batch_size)] = std_err_res / np.sum(delta_xsqd, axis=1, keepdims=True)**0.5

        sse_alpha_num = np.mean(X_batch, axis=1, keepdims=True)**2
        sse_alpha_denum = np.var(X_batch, axis=1, keepdims=True) * n_points
        sse_alpha = (sse_alpha_num / sse_alpha_denum + (1/n_points))**0.5
        std_err_alpha[i:i+int(batch_size)] = std_err_res *  sse_alpha

        # Simulation progress update
        batch_count += 1 if batch_count < total_batches else total_batches

        progress_info = {'Status':'estimation',
                        'Batch_count':batch_count,
                        'Total_batches':total_batches
                        }
        progress(**progress_info)

        # Clear batch variables to free memory
        X_batch = y_batch = y_diff = x_diff = delta_y = delta_x = slopes = None
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Clear variables to free memory
    y_diff = x_diff = delta_y = delta_x = slopes = None

    return alpha, std_err_alpha, beta, std_err_beta, elapsed_time
# 5. Define LTS function to estimate alpha and beta from simulated data

def LTS(X: np.ndarray, 
        y: np.ndarray, 
        cell: str, 
        trim_pct: float = None,
        n: np.int32 = None,
        distribution: str = None,
        available_ram: float = None
        ):
            
    if y.shape != X.shape:
        raise ValueError(f"Shape mismatch: X shape {X.shape} and Y shape {y.shape} must be the same.")
            
    batch_size, total_batches, alpha, beta, std_err_alpha, std_err_beta = batcher(X.shape[0], n, available_ram, cell, distribution, 'LTS')
    
    # initialize progress update generator
    batch_count = 0

    start_time = time.perf_counter()
    for i in range(0, X.shape[0], int(batch_size)):
        X_batch = X[i:i+int(batch_size)]
        y_batch = y[i:i+int(batch_size)]

        n_sims, n_points = X_batch.shape
        if X_batch.shape != y_batch.shape or n_points < 2:
            raise ValueError("X and Y must have the same shape and at least 2 points.")
                    
        if not (0 <= trim_pct < 1):
            raise ValueError("trim_pct must be between 0 and 1.")
                        
        # Calculate how many points to keep per simulation
        if trim_pct:
            h = n_points - int(trim_pct * n_points)
        else:
            h = n_points - int(0.2 * n_points)
                    
        # 1. Get initial OLS estimates for all simulations
        mean_X = np.mean(X_batch, axis=1, keepdims=True)
        mean_Y = np.mean(y_batch, axis=1, keepdims=True)
        dev_x = X_batch - mean_X
        dev_y = y_batch - mean_Y
                    
        beta_init = np.sum(dev_x * dev_y, axis=1, keepdims=True) / np.sum(dev_x**2, axis=1, keepdims=True)
        alpha_init = mean_Y - (beta_init * mean_X)
                    
        # 2. Calculate initial residuals and absolute values
        residuals = y_batch - (alpha_init + beta_init * X_batch)
        abs_res = np.abs(residuals)
                    
        # 3. Vectorized sorting to find indices of the smallest 'h' residuals per simulation
        trimmed_indices = np.argsort(abs_res, axis=1)[:, :h]
                    
        # 4. Advanced indexing to extract the trimmed X and Y subsets
        row_indices = np.arange(n_sims)[:, np.newaxis]
        X_trimmed = X_batch[row_indices, trimmed_indices]
        Y_trimmed = y_batch[row_indices, trimmed_indices]
                    
        # 5. Run OLS on the trimmed subsets
        mean_X_tr = np.mean(X_trimmed, axis=1, keepdims=True)
        mean_Y_tr = np.mean(Y_trimmed, axis=1, keepdims=True)
        dev_x_tr = X_trimmed - mean_X_tr
        dev_y_tr = Y_trimmed - mean_Y_tr
        
        dev_xy_tr = np.float64(dev_x_tr) * np.float64(dev_y_tr)
        dev_x_squared_tr = np.float64(dev_x_tr)**2

        beta[i:i+int(batch_size)] = np.sum(dev_xy_tr, axis=1, keepdims=True) / np.sum(dev_x_squared_tr, axis=1, keepdims=True)
        
        alpha[i:i+int(batch_size)] = np.mean(Y_trimmed, axis=1, keepdims=True) - (beta[i:i+int(batch_size)] * np.mean(X_trimmed, axis=1, keepdims=True))

        sse_res = np.sum(np.float64(y_batch)**2, axis=1, keepdims=True) - beta[i:i+int(batch_size)]*np.sum(dev_xy_tr, axis=1, keepdims=True)
        std_err_res = (sse_res/(n_points - 2))**0.5
        std_err_beta[i:i+int(batch_size)] = std_err_res / np.sum(dev_x_squared_tr, axis=1, keepdims=True)**0.5

        sse_alpha_num = np.mean(X_batch, axis=1, keepdims=True)**2
        sse_alpha_denum = np.var(X_batch, axis=1, keepdims=True) * n_points
        sse_alpha = (sse_alpha_num / sse_alpha_denum + (1/n_points))**0.5
        std_err_alpha[i:i+int(batch_size)] = std_err_res *  sse_alpha

        # Simulation progress update
        batch_count += 1 if batch_count < total_batches else total_batches

        progress_info = {'Status':'estimation',
                        'Batch_count':batch_count,
                        'Total_batches':total_batches
                        }
        progress(**progress_info)

        # Clear batch variables to free memory
        X_batch = y_batch = residuals = X_trimmed = Y_trimmed =  None
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Clear variables to free memory
    residuals = X_trimmed = Y_trimmed = None

    return alpha, std_err_alpha, beta, std_err_beta, elapsed_time

# 6. Define Bayesian Linear Estimator function to estimate alpha and beta from simulated data
def Bayesian_lm(X: np.ndarray, 
                y: np.ndarray,
                cell: str, 
                n: np.int32 = None,
                distribution: str = None, 
                normal_beta_prior: float = None,
                normal_beta_sigma_sq_prior: float = None,
                normal_alpha_sigma_sq_prior: float = None,
                normal_alpha_prior: float = None,
                available_ram: float = None  
                ):

    """
    Computes Bayesian Linear Regression parameters using a Conjugate Normal Prior.
    sigma_sq: variance of the error term (assumed known for the study)
    """

    if y.shape != X.shape:
        raise ValueError(f"Shape mismatch: X shape {X.shape} and Y shape {y.shape} must be the same.")
            
    batch_size, total_batches, alpha, beta, std_err_alpha, std_err_beta = batcher(X.shape[0], n, available_ram, cell, distribution, 'Bayes')

    m_alpha_post = m_alpha_zero_post = alpha
    m_beta_post = beta

    # initialize progress update generator
    batch_count = 0

    start_time = time.perf_counter()
    for i in range(0, X.shape[0], int(batch_size)):
        X_batch = X[i:i+int(batch_size)]
        y_batch = y[i:i+int(batch_size)]

        nsims, n_points = X_batch.shape
                    
        # 1. Compute means and sums of squares per simulation
        x_bar = np.mean(X_batch, axis=1, keepdims=True)
        y_bar = np.mean(y_batch, axis=1, keepdims=True)
                    
        ss_x = np.sum((X_batch - x_bar)**2, axis=1, keepdims=True)
        ss_xy = np.sum(np.float64((X_batch - x_bar) * (y_batch - y_bar)), axis=1, keepdims=True)
                    
        # 2. Compute OLS reference points per simulation
        b_ols = np.where(ss_x != 0, ss_xy / ss_x, 0.0)
        a_ols = y_bar  # Intercept centered at the mean of X
                    
        # 3. Setup Prior hyper-parameters (Defaults: Mean = 0, Variance = 1)
        m_beta_prior = normal_beta_prior if normal_beta_prior else 0.0
        s_beta_prior_sq = normal_beta_sigma_sq_prior if normal_beta_sigma_sq_prior else 1.0
                    
        m_alpha_prior = normal_alpha_prior if normal_alpha_prior else 0.0
        s_alpha_prior_sq = normal_alpha_sigma_sq_prior if normal_alpha_sigma_sq_prior else 1.0

        # 4. Update Slope Posterior (Beta) per simulation
        precision_beta_post = (1.0 / s_beta_prior_sq) + (ss_x / s_beta_prior_sq)
        s_beta_post_sq = 1.0 / precision_beta_post
        m_beta_post[i:i+int(batch_size)] = (s_beta_post_sq * (m_beta_prior / s_beta_prior_sq)) + \
            (s_beta_post_sq * (ss_x * b_ols / s_beta_prior_sq))

        # 5. Update Mean-Centered Intercept Posterior (Alpha at x_bar) per simulation
        precision_alpha_post = (1.0 / s_alpha_prior_sq) + (n_points / s_alpha_prior_sq)
        s_alpha_post_sq = 1.0 / precision_alpha_post
        m_alpha_post[i:i+int(batch_size)] = (s_alpha_post_sq * (m_alpha_prior / s_alpha_prior_sq)) + \
            (s_alpha_post_sq * (n_points * a_ols / s_alpha_prior_sq))

        # 6. Transform intercept at mean back to intercept at zero (axis=0)
        m_alpha_zero_post[i:i+int(batch_size)] = m_alpha_post[i:i+int(batch_size)] - (m_beta_post[i:i+int(batch_size)] * x_bar)

        sse_res = np.sum(np.float64(y_batch)**2, axis=1, keepdims=True) - beta[i:i+int(batch_size)]*ss_xy
        std_err_res = (sse_res/(n_points - 2))**0.5
        std_err_beta[i:i+int(batch_size)] = std_err_res / ss_x**0.5

        sse_alpha_num = np.mean(X_batch, axis=1, keepdims=True)**2
        sse_alpha_denum = np.var(X_batch, axis=1, keepdims=True) * n_points
        sse_alpha = (sse_alpha_num / sse_alpha_denum + (1/n_points))**0.5
        std_err_alpha[i:i+int(batch_size)] = std_err_res *  sse_alpha

        # Simulation progress update
        batch_count += 1 if batch_count < total_batches else total_batches

        progress_info = {'Status':'estimation',
                        'Batch_count':batch_count,
                        'Total_batches':total_batches
                        }
        progress(**progress_info)

        # Clear batch variables to free memory
        X_batch = y_batch = None

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Clear variables to free memory
    x_bar = y_bar = ss_x = ss_xy = b_ols = a_ols = None

    return m_alpha_zero_post, std_err_alpha, m_beta_post, std_err_beta, elapsed_time

time_table = {}
def aggregator(X: np.ndarray, 
               y: np.ndarray,  
               method: str,
               cell: str, 
               flag: str, 
               n: int,
               true_alpha: float = None, 
               true_beta: float = None,
               trim_pct: float = None,
               distribution: str = None,
               available_ram: float = None,
               normal_beta_prior: float = None,
               normal_beta_sigma_sq_prior: float = None,
               normal_alpha_sigma_sq_prior: float = None,
               normal_alpha_prior: float = None,
               ):
    
    global time_table  # Use the global time_table dictionary to store elapsed times for each method

    if method in ['OLS', 'OLSE', 'ols', 'olse']:

        alpha, std_err_alpha, beta, std_err_beta, elapsed_time = OLS(X, 
                                                        y, 
                                                        cell=cell, 
                                                        n=n,
                                                        distribution=distribution,
                                                        available_ram=available_ram)
        
    elif method in ['LTS', 'lts']:

        alpha, std_err_alpha, beta, std_err_beta, elapsed_time = LTS(X, 
                                                        y, 
                                                        cell, 
                                                        trim_pct,
                                                        n,
                                                        distribution,
                                                        available_ram
                                                        )
    
    elif method in ['Bayesian', 'bayesian', 'bayes', 'BAYES', 'Bayes']:

        alpha, std_err_alpha, beta, std_err_beta, elapsed_time = Bayesian_lm(X, 
                                                                y,
                                                                cell, 
                                                                n,
                                                                distribution, 
                                                                available_ram = available_ram
                                                                )
    
    elif method in ['Theil', 'theil', 'Theils', 'Theils', 'THEIL', 'THEILS']:

        alpha, std_err_alpha, beta, std_err_beta, elapsed_time = Theil_pm_vec(X, 
                                                                y, 
                                                                cell=cell,
                                                                n=n,
                                                                distribution=distribution,
                                                                available_ram=available_ram
                                                                )
        

    # Store results for table construction
    final_table = {
                    'Sample Size': np.full(X.shape[0], n),   
                    'Simulation Cell': np.full(X.shape[0], cell, dtype=f'<U{len(cell)}'),
                    'Distribution': np.full(X.shape[0], distribution, dtype=f'<U{len(distribution)}'),
                    'Method': np.full(X.shape[0], method, dtype=f'<U{len(method)}'),
                    'Beta': beta.reshape(-1),
                    'Alpha': alpha.reshape(-1),
                    'Std. Error (Beta)': std_err_beta.reshape(-1),
                    'Std. Error (Alpha)': std_err_alpha.reshape(-1),
                    }
            
    # Convert to DataFrame for easy visualization
    sim_results = pd.DataFrame(final_table).reset_index(drop=True)

    write_file(content=sim_results,
               method=method, 
                cell=cell, 
                flag='sim data',  
                n=n,
                dist_type=distribution
                )

    sim_analysis(X,
                 y,
                 alpha, 
                beta, 
                true_alpha, 
                true_beta, 
                std_err_alpha,
                std_err_beta,
                method, 
                cell, 
                n,
                flag,
                distribution
                )
    
    time_table[method] = elapsed_time

    if method == 'Bayesian':
        time_table['Sample Size'] = n
        time_table['Simulation Cell'] = cell
        time_table['Distribution'] = distribution
        time_track = pd.DataFrame([time_table]).reset_index(drop=True)

        write_file(content=time_track,
                flag='time track'
                )
        
        time_table = {}  # Reset time_table for the next simulation cell
        time_track = None  # Clear time_track to free memory
        