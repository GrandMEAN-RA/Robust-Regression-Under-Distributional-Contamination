
# Import needed libraries
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import terminal_manager as tm
from setup_tools import progress

# Initialize global variables for state-based indexing
figure = {}
axis = [None, None]
index = [None, None]
col_ind = 0
row_ind = 0


# This function plots Kernel Density Estimation curves
def kde_plot(cell: str,
             n: int = None, 
             samples: int = None,
             new_fig: bool = True,
             **plot_data
             )-> None:
    
    global figure, axis, col_ind, row_ind, index

    rows = samples // 3 + (samples % 3 > 0) if samples > 3 else 1
    cols = 3 if samples >= 3 else samples

    data = pd.DataFrame(plot_data)

    # Handle Plotting
    if new_fig == True:
        fig, ax = plt.subplots(rows, cols, figsize=( 6 * cols, 3 * rows))
        axis = [fig, ax]
        index = [row_ind, col_ind]
        figure['kde'] = [axis, index]  
    else:
        fig, ax = figure['kde'][0]
        row_ind, col_ind = figure['kde'][1]
        if col_ind < cols-1:
            col_ind += 1  
        else:
            col_ind = 0
            row_ind += 1 if row_ind < rows - 1 else 0
        index = [row_ind, col_ind]
        figure['kde'] = [axis, index]

    for arr in data.columns:
        sns.kdeplot(x=data[arr], ax=ax[row_ind,col_ind] if rows > 1 else ax[col_ind], label=arr)

    # Set x limits
    if (n >= 100 and cell in ['Standard', 'Mixtures', 'Contaminations']) or cell == 'Outliers':
        ax[row_ind,col_ind].set_xlim(-40, 40) if rows > 1 else ax[col_ind].set_xlim(-40, 40)
        if n == 500 and cell in ['Standard', 'Contaminations']:
            ax[row_ind,col_ind].set_xlim(-10, 10) if rows > 1 else ax[col_ind].set_xlim(-10, 10)
    if n <= 50 and cell == 'Mixtures':
        ax[row_ind,col_ind].set_xlim(-1, 1) if rows > 1 else ax[col_ind].set_xlim(-1, 1)
    if n == 50 and cell in ['Standard', 'Contaminations']:
        ax[row_ind,col_ind].set_xlim(-55, 55) if rows > 1 else ax[col_ind].set_xlim(-55, 55)
    if n == 30 and cell == 'Contaminations':
        ax[row_ind,col_ind].set_xlim(-25, 25) if rows > 1 else ax[col_ind].set_xlim(-25, 25)

    # Common plot settings
    ax[row_ind,col_ind].set_title(f'Sample Size = {n}') if rows > 1 else ax[col_ind].set_title(f'Sample Size = {n}')
    ax[row_ind,col_ind].set_xlabel('Error Values') if rows > 1 else ax[col_ind].set_xlabel('Error Values')
    ax[row_ind,col_ind].set_ylabel('Density') if rows > 1 else ax[col_ind].set_ylabel('Density')
    ax[row_ind,col_ind].legend() if rows > 1 else ax[col_ind].legend()
    
    # Save and display after accumulating all sample sizes
    if (row_ind == (rows-1)) and (col_ind == (cols-1)):
        fig.tight_layout()
        filename = f'error_distributions_{cell}_model'
        fig.suptitle(f'Error Distributions Across Sample Sizes: {cell} Model', fontsize=16, y=1.1)
        fig.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')

        plot_info = f'{tm.bold}{tm.red}{filename}{tm.reset}'
        
        progress_info = {'Status':'plots',
                    'PROGRESS':plot_info
                    }
        progress(**progress_info)
        fig.show()

        # Rset axis index to 0
        row_ind = col_ind = 0
        
# This function Plots scatter graph with trendlines
def scatter_plot(cell: str, 
                 data_files: list,
                 flag: str,
                 params: list = None,
                 new_fig: bool = True,
                 **plot_data
                 ) -> None:
    
    global figure, axis, col_ind, row_ind, index

    # Get plot dimensions
    rows = len(data_files) // 3 + (len(data_files) % 3 > 0) if len(data_files) > 3 else 1
    cols = 3 if len(data_files) >= 3 else len(data_files)

    data = pd.DataFrame(plot_data)

    # Handle Plotting
    if new_fig == True:
        fig, ax = plt.subplots(rows, cols, figsize=( 6 * cols, 3 * rows))
        axis = [fig, ax]
        index = [row_ind, col_ind]
        figure['scatter'] = [axis, index]  
    else:
        fig, ax = figure['scatter'][0]
        row_ind, col_ind = figure['scatter'][1]
        if col_ind < cols-1:
            col_ind += 1  
        else:
            col_ind = 0
            row_ind += 1 if row_ind < rows else 0
        index = [row_ind, col_ind]
        figure['scatter'] = [axis, index]

    # Plot scatter graph with trendlines for each method 
    colors = ['black', 'red', 'blue', 'green', 'yellow']
    
    ax.scatter(data[data.columns[0]], data[data.columns[1]], color='black', label='Data Points')
    
    if flag == 'empirical_residuals':
        current_ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='y=0 Axis')

    else:
        if len(data.columns) > 2:
            if params:
                for y, p, c in zip(data.columns[2:], params, colors):

                    if len(p) == 3:
                        label = f'${y} y = {round(p[0],3)} + {round(p[1],3)}x $\nR^2 = {round(p[2]),3}'  
                    else: 
                        label = f'${y} y = {round(p[0],3)} + {round(p[1],3)}x$'
                    
                    current_ax.plot(data[data.columns[0]], data[y], color=c, label= label)
            else:
                for y, c in zip(data.columns[2:], colors):   
                    current_ax.plot(data[data.columns[0]], data[y], color=c)

    if flag == 'exploratory': 
        current_ax.set_xlabel(data.columns[0])
        current_ax.set_ylabel(data.columns[1])
    else:
        current_ax.set_xlabel(data.columns[0])
        current_ax.set_ylabel(data.columns[1])
    current_ax.legend(fontsize=9)
    current_ax.set_title(f'Data with {cell}')
    current_ax.grid(False)

    if (row_ind == (rows-1)) and (col_ind == (cols-1)):
        filename = f'scatter_trends_raw {cell}'
        fig.suptitle(f'Scatter Graph of GDP vs GNI with Linear Trends' if flag == 'exploratory' else f'Scatter Graph of Residual vs Y-cap with Linear Trends')
        fig.savefig(f'{filename}.png' if flag == 'exploratory' else f'Residual vs Y.png', dpi=300, bbox_inches='tight')  # Save the figure
        
        plot_info = f'{tm.bold}{tm.red}{filename}{tm.reset}'
        
        progress_info = {'Status':'plots',
                    'PROGRESS':plot_info
                    }
        progress(**progress_info)
        fig.show()

        # Rset axis index to 0
        row_ind = col_ind = 0
        figure[tag] = None


def qq_plot(x: np.ndarray, 
            line_ang: str,
            cell: str, 
            data_files: list,
            new_fig: bool = True
            ) -> None:

    global figure, axis, col_ind, row_ind, index

    plot = 'qq'

    flag = 'qqplot'

    # Get plot dimensions
    rows, cols, tag = plot_dim(flag, 
                               cell, 
                               plot,
                               None,
                               None,
                               data_files
                               )

    # Handle Plotting
    if new_fig == True:
        fig, ax = create_axis(flag, rows, cols)
        axis = [fig, ax]
        index = [row_ind, col_ind]
        figure[tag] = [axis, index]  
    else:
        fig, ax = figure[tag][0]
        row_ind, col_ind = figure[tag][1]
        if col_ind < cols-1:
            col_ind += 1  
        else:
            col_ind = 0
            row_ind += 1 if row_ind < rows else 0

    current_ax = detect_axis(rows, cols, ax)

    # Setting line='45' adds a 45-degree reference line
    sm.qqplot(x, line=line_ang, ax=current_ax, fit=True)
    
    current_ax.set_title(f'Normal Q-Q Plot: Data with {cell}')
    current_ax.set_xlabel('Standardized Normal Quantiles')
    current_ax.set_ylabel('Data Quantiles')
    current_ax.legend()
    current_ax.grid(False)

    # 3. Display the plot
    if (row_ind == (rows-1)) and (col_ind == (cols-1)):
        filename = 'q-q_plot' 
        fig.suptitle('Normal Q-Q Plot')
        fig.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')  # Save the figure
        
        plot_info = f'{tm.bold}{tm.red}{filename}{tm.reset}'
        
        progress_info = {'Status':'plots',
                    'PROGRESS':plot_info
                    }
        progress(**progress_info)
        fig.show()

        # Rset axis index to 0
        row_ind = col_ind = 0

if __name__ == '__main__':

    x = np.random.standard_normal(100)
    y = np.random.lognormal(0,1,100)
    z = np.random.standard_cauchy(100)
    p = np.random.standard_t(5.0,100)
    q = np.random.standard_exponential(100)
    r = np.random.standard_gamma(1,100)

    data = {'x': x,
            'y': y,
            'z': z,
            'p': p,
            'r': r,
            'q': q}
    
    for i in range(len(data.keys())):
        new_fig = True if i == 0 else False
        kde_plot(cell='Normal',
                n = 100, 
                samples = 6,
                new_fig = new_fig,
                **data
                )