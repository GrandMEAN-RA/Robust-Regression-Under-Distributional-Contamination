
# Import needed libraries
import numpy as np
import pandas as pd
from plotter import kde_plot
from writer import write_file

# Initialize random number generator
rng = None

# This function generates random errors from the three distributions under study
def sim_errors(n: int,
               cell: str, 
               distr: str,
               trim_pct: float,  
               rng: np.random.Generator,
               replication: int = 1
               ) -> np.ndarray:

    # Calculate shape and contamination size based on trim percentage
    shape = n if replication == 1 else (replication, n)
    n_perturbs = int(n * trim_pct)  # Ensure integer type for slicing
    perturb_shape = n_perturbs if replication == 1 else (replication, n_perturbs)

    def standard_distributions() -> tuple:

        # Gnerate base distributions
        if distr in ['Normal','normal','NORMAL']:
            errors = rng.normal(loc=0.0, scale=1.0, size=shape) # Unit Normal: N(0,1)
        elif distr in ['Lognormal','lognormal','LOGNORMAL', 'Log-normal']:
            errors = rng.lognormal(mean=0.0, sigma=1.0, size=shape) # Unit Lognormal: LN(0,1)
        elif distr in ['Cauchy','cauchy']:
            errors = rng.standard_cauchy(size=shape) # Standard Cauchy: C(0,1)

        return errors
    
    def outlier_distributions() -> tuple:
        
        # Call the standard error models first
        errors = standard_distributions()
        
        if replication == 1:
            if distr in ['Normal','normal','NORMAL']:
                errors[-n_perturbs:] = rng.normal(loc=0.0, scale=10.0, size=perturb_shape) # Normal: N(0,10)
            elif distr in ['Lognormal','lognormal','LOGNORMAL', 'Log-normal']:
                errors[-n_perturbs:] = rng.lognormal(mean=0.0, sigma=10.0, size=perturb_shape) # Lognormal: LN(0,10)
            elif distr in ['Caucy','cauchy']:
                errors[-n_perturbs:] = rng.standard_cauchy(size=perturb_shape)*10.0 # Cauchy: C(0,10)
        else: 
            if distr in ['Normal','normal','NORMAL']:
                errors[:, -n_perturbs:] = rng.normal(loc=0.0, scale=10.0, size=perturb_shape) # Normal: N(0,10)
            elif distr in ['Lognormal','lognormal','LOGNORMAL', 'Log-normal']:
                errors[:, -n_perturbs:] = rng.lognormal(mean=0.0, sigma=10.0, size=perturb_shape) # Lognormal: LN(0,10)
            elif distr in ['Caucy','cauchy']:
                errors[:, -n_perturbs:] = rng.standard_cauchy(size=perturb_shape)*10.0 # Cauchy: C(0,10)

        return errors
    
    def mixture_distributions() -> tuple:
        
        # Call the standard error models first
        errors = standard_distributions()
        
        if replication == 1:
            if distr in ['Normal','normal','NORMAL']:
                errors[-n_perturbs:] = rng.normal(loc=10.0, scale=10.0, size=perturb_shape) # Normal: N(10,10)
            elif distr in ['Lognormal','lognormal','LOGNORMAL', 'Log-normal']:
                errors[-n_perturbs:] = rng.lognormal(mean=10.0, sigma=10.0, size=perturb_shape) # Lognormal: LN(10,10)
            elif distr in ['Caucy','cauchy']:
                errors[-n_perturbs:] = rng.standard_cauchy(size=perturb_shape)*10.0 + 10.0 # Cauchy: C(10,10)
        else: 
            if distr in ['Normal','normal','NORMAL']:
                errors[:, -n_perturbs:] = rng.normal(loc=10.0, scale=10.0, size=perturb_shape) # Normal: N(10,10)
            elif distr in ['Lognormal','lognormal','LOGNORMAL', 'Log-normal']:
                errors[:, -n_perturbs:] = rng.lognormal(mean=10.0, sigma=10.0, size=perturb_shape) # Lognormal: LN(10,10)
            elif distr in ['Caucy','cauchy']:
                errors[:, -n_perturbs:] = rng.standard_cauchy(size=perturb_shape)*10.0 + 10.0 # Cauchy: C(10,10)

        return errors
    
    def contamination_distributions() -> tuple:
        
        # Call the standard error models first
        errors = standard_distributions()
        
        if replication == 1:
            if distr in ['Normal','normal','NORMAL']: 
                errors[-n_perturbs:] = rng.geometric(p=0.5, size=(perturb_shape)) # Student's t Contaminations: t(5)
            elif distr in ['Lognormal','lognormal','LOGNORMAL', 'Log-normal']:
                errors[-n_perturbs:] = rng.geometric(p=0.5, size=(perturb_shape)) # Student's t Contaminations: t(5)
            elif distr in ['Caucy','cauchy']:
                errors[-n_perturbs:] = rng.geometric(p=0.5, size=(perturb_shape)) # Student's t Contaminations: t(5)
        else:
            if distr in ['Normal','normal','NORMAL']: 
                errors[:, -n_perturbs:] = rng.geometric(p=0.5, size=(perturb_shape)) # Student's t Contaminations: t(5)
            elif distr in ['Lognormal','lognormal','LOGNORMAL', 'Log-normal']:
                errors[:, -n_perturbs:] = rng.geometric(p=0.5, size=(perturb_shape)) # Student's t Contaminations: t(5)
            elif distr in ['Caucy','cauchy']:
                errors[:, -n_perturbs:] = rng.geometric(p=0.5, size=(perturb_shape)) # Student's t Contaminations: t(5)

        return errors
    
    if cell in ['Standard', 'standard', 'STANDARD']:
        return standard_distributions()
    
    elif cell in ['Outliers', 'outliers', 'OUTLIERS']:
        return outlier_distributions()

    elif cell in ['Mixtures', 'mixtures', 'MIXTURES']:
        return mixture_distributions()

    elif cell in ['Contaminations', 'contaminations', 'CONTAMINATIONS']:
        return contamination_distributions()
    else:
        raise ValueError(f"Unsupported cell type: {cell}")
    
# This functions generate sequential design variable X and then constructs
# the response variable Y across the three distributions
data = {}
info = pd.DataFrame()
def sim_data(shape: tuple,
             cell: str, 
             trim_pct: float, 
             true_alpha: float, 
             true_beta: float, 
             distribution_name: str,
             distribution_plots: bool = False,
             samples: int = None
             ) -> tuple:

    global rng, data, info

    replication, n = shape

    # Generate X as a sequential model
    X = np.array(np.ones((replication, 1), dtype=int) * np.arange(1, n+1))

    # Set rng to reset for every new iterations over sample sizes
    rng = np.random.default_rng(n)  # For reproducibility
    state = rng.bit_generator.state['state']['state']
    #rng.bit_generator.state() = state

    # Write RNG states to file for the records
    info = pd.DataFrame({'Cell':[cell],
                         'Distribution':[distribution_name],
                         'Sample size':[n],
                         'Seed':[n],
                         'RNG State':[state]}).reset_index(drop=True)

    write_file(cell = cell, 
               dist_type = distribution_name,
               flag = 'random_state', 
               n=n,
               content = info,)

    # Generate random errors from distributions
    errors = sim_errors(n,
                        cell, 
                        distribution_name,
                        trim_pct, 
                        rng,
                        replication
                        )
    
    # Generate Y as Y = true_alpha + (true_beta * X) + error
    Y = np.array(true_alpha + (true_beta * X) + errors)

    # Plot error distributions for the current simulation cell and sample size
    # Convert to DataFrame (Each array becomes a column)
    if distribution_plots:
        
        new_fig = False
        if n == 10:
            # Set defaults
            new_fig = True
        
        data[distribution_name] = errors[0,:] # Enable easy unpacking into keyword argument
        
        if distribution_name in ['Cauchy','cauchy','CAUCHY']:
            kde_plot(cell=cell,
                    n=n, 
                    samples=samples,
                    new_fig=new_fig,
                    **data
                    )
            
            data = {}
            
            if cell == 'Contaminations' and n==1000:
                info = pd.DataFrame()
                # Write RNG final states to file for the records
                state = rng.bit_generator.state['state']['state']
                info = pd.DataFrame({'Cell':[cell],
                         'Distribution':[distribution_name],
                         'Sample size':[n],
                         'Seed':[n],
                         'RNG State':[state]}).reset_index(drop=True)

                write_file(cell = cell, 
                        dist_type = distribution_name,
                        flag = 'random_state', 
                        n = n,
                        content = info,)

    info = pd.DataFrame()
    
    return X.astype(np.float32), Y.astype(np.float32)