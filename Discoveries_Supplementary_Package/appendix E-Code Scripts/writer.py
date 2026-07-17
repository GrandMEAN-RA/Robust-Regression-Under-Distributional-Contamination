
# import needed libraries
import os
import pandas as pd
import terminal_manager as tm
from setup_tools import progress


# This function writes result dataframes into CSV files
def write_file(content: pd.DataFrame,
               method:str = None, 
               cell:str = None, 
               flag: str = None, 
               n: int = None,
               dist_type: str = None
               ) -> None:
    
    def pen(name: str)-> None:
        
        # Check if the file already exists and has data
        file_exists = os.path.exists(name) and os.stat(name).st_size > 0

        # content is your DataFrame
        content.to_csv(
            name, 
            mode='a',                  # Append mode
            header=not file_exists,    # Only write header if the file does NOT exist
            index=False                # Prevents writing row numbers (0, 1, 2...)
            )

    if flag == 'simulation':
        filenames = [f'{dist_type}.csv', 'General Simulation Results.csv']

        for name in filenames:
            
            pen(name)

            info = f'{tm.bold}{tm.cyan}Simulation results for n = {n} {cell} Model {dist_type} {method} has been written to file {name}{tm.reset}'
            progress_info = {'Status':'writer',
                            'PROGRESS':info
                            }
            progress(**progress_info)
    
    elif flag == 'sim data':
        filename = f'simulated data {cell} {dist_type} n = {n}.csv'
            
        pen(filename)

        info = f'{tm.bold}{tm.cyan}Simulated data for n = {n} {cell} Model {dist_type} {method} has been written to file {filename}{tm.reset}'
        progress_info = {'Status':'writer',
                        'PROGRESS':info
                        }
        progress(**progress_info)

    elif flag == 'random_state':
        filename = 'simulation states.csv'    

        pen(filename)

        info = f'{tm.bold}{tm.cyan}Random state for simulation cell {cell} n = {n} {dist_type} has been written to file {filename}{tm.reset}'
        progress_info = {'Status':'writer',
                        'PROGRESS':info
                        }
        progress(**progress_info)
    
    elif flag == 'system_specs':
        filename = 'system_specs.csv'    

        pen(filename)

        info = f'{tm.bold}{tm.cyan}System specifications have been written to file {filename}{tm.reset}'
        progress_info = {'Status':'writer',
                        'PROGRESS':info
                        }
        progress(**progress_info)
    
    elif flag == 'time track':
        filename = 'time_track.csv'    

        pen(filename)

        info = f'{tm.bold}{tm.cyan}Time tracking information has been written to file {filename}{tm.reset}'
        progress_info = {'Status':'writer',
                        'PROGRESS':info
                        }
        progress(**progress_info)

    else:
        filename = f'{cell}.csv'

        pen(filename)
        
        info = f'{tm.bold}{tm.cyan}Simulation results for n = {n} Data with {cell} {method} has been written to file {filename}{tm.reset}'
        progress_info = {'Status':'writer',
                        'PROGRESS':info
                        }
        progress(**progress_info)

    # Clear Content Dataframe
    content = None