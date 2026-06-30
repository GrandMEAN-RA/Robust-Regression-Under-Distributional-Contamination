
"""
MONTE-CARLO STUDY OF SOME ROBUST ESTIMATORS: 
The Simple Linear Regression Case
"""

# Import necessary libraries
from pathlib import Path
import gc

import setup_tools
from setup_tools import parameter_grid, progress, get_system_specs
from writer import write_file
import terminal_manager as tm
from simulator import sim_data
from estimators import aggregator

# Get system hardware specifications for performance monitoring
available_ram, system_specs = get_system_specs()

write_file(flag = 'system_specs',
           content = system_specs,)

# Set path to parent directory
PATH = Path(__file__).resolve().parent

# Define Experimental Parameters
sample_sizes = [10, 30, 50, 100, 500, 1000] 
sim_cells = ['Contaminations'] 
distributions = ['Normal', 'Lognormal', 'Cauchy']
estimators = ['OLSE', 'LTS', 'Theils', 'Bayesian']

replications = 150000                  
true_alpha, true_beta = 5.0, 2.0  
trim_pct = 0.2

# Print basic simulation setup info and create parameter grid 
# over sample sizes, distributions and simulation cells
setup_info = {'Status':'simulation setup',
              'Sample Sizes':sample_sizes, 
              'Replications':replications,
              'Simulation Cells':sim_cells, 
              'Distributions':distributions, 
              'Estimators':estimators,
              'Simulation Grid Parameters':(sim_cells, sample_sizes, distributions)
              }

param_grid = parameter_grid(**setup_info)

# set flag variable to None for later use in controlling function behavior
flag = None 

# Loop through the parameter grid
for cell, n, distr in param_grid:

    shape = (replications, n) # Shape of data arrays

    # Simulate data and errors
    X, Y = sim_data(shape,
                    cell, 
                    trim_pct, 
                    true_alpha, 
                    true_beta,
                    distr,
                    True,
                    len(sample_sizes)
                    )
    
    # Fit Regression Estimators
    flag = 'simulation'
    for method in estimators:
        setup_tools.batch_flag = None
        aggregator(
                    X, 
                    Y,  
                    method,
                    cell, 
                    flag, 
                    n,
                    true_alpha, 
                    true_beta,
                    trim_pct,
                    distr,
                    available_ram
                    )
    
    info = f'{tm.bold}{tm.cyan}Simulation iterations for the whole sample size {n} {cell} {distr} cell successfully completed!{tm.reset}'
    progress_info = {'Status':'simulation',
                    'PROGRESS':info
                    }
    progress(**progress_info)
    
    gc.collect()
    
# End of simulation
info = f'{tm.bold}{tm.cyan}All {len(param_grid) * len(distributions) * len(estimators)} iterations have been succesfully executed.{tm.reset}'
progress_info = {'Status':'simulation',
                'PROGRESS':info
                }
progress(**progress_info)

