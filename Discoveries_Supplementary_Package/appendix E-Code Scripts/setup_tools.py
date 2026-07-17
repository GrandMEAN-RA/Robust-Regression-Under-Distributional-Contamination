
import platform
import psutil
import GPUtil

from itertools import product
import time

import numpy as np
import pandas as pd

import terminal_manager as tm

sys_spec_anchor = None
batch_flag = None

# Detects and prints the system hardware specifications, including CPU, RAM, ROM, and GPU details. 
# This information is crucial for understanding the computational resources available for running simulations and analyses efficiently.
def get_system_specs():

    global sys_spec_anchor, batch_flag

    print("="*20 + " System Information " + "="*20)
    
    # 1. OS and Basic Info
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Processor Architecture: {platform.machine()}")

    # 2. CPU Information
    print("\n--- CPU ---")
    print(f"Processor: {platform.processor()}")
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"Total Cores (Logical): {psutil.cpu_count(logical=True)}")
    
    # CPU Speed
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        print(f"Max Frequency: {cpu_freq.max:.2f} Mhz")
        print(f"Current Frequency: {cpu_freq.current:.2f} Mhz")

    # 3. RAM (Memory) Information
    print("\n--- RAM ---")
    virtual_mem = psutil.virtual_memory()
    total_ram_gb = virtual_mem.total / (1024 ** 3)
    available_ram_gb = virtual_mem.available / (1024 ** 3)
    print(f"Total RAM: {total_ram_gb:.2f} GB")
    print(f"Available RAM: {available_ram_gb:.2f} GB")

    # 4. ROM / Disk Space Information (Main Drive)
    print("\n--- ROM / Storage ---")
    disk_info = psutil.disk_usage('/')
    total_disk_gb = disk_info.total / (1024 ** 3)
    free_disk_gb = disk_info.free / (1024 ** 3)
    print(f"Total Storage Space: {total_disk_gb:.2f} GB")
    print(f"Free Storage Space: {free_disk_gb:.2f} GB")
    
    # 5. Graphics / GPU Information
    print("\n--- Graphics / GPU ---")
    gpus = GPUtil.getGPUs()
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i+1}: {gpu.name}")
            print(f"  Total GPU Memory: {gpu.memoryTotal} MB")
            print(f"  Free GPU Memory: {gpu.memoryFree} MB")
            print(f"  GPU Temperature: {gpu.temperature} °C\n", end='\n')
    else:
        print("No dedicated GPU detected via GPUtil (or Integrated Graphics only).\n", end='\n')
    
    sys_spec_anchor = tm.get_current_position()

    system_specs =pd.DataFrame([{
        "Operating System": f"{platform.system()} {platform.release()}",
        "Processor Architecture": platform.machine(),
        "Processor": platform.processor(),
        "Physical Cores": psutil.cpu_count(logical=False),
        "Total Cores (Logical)": psutil.cpu_count(logical=True),
        "Max CPU Frequency (Mhz)": f'{cpu_freq.max:.2f}' if cpu_freq else None,
        "Current CPU Frequency (Mhz)": f'{cpu_freq.current:.2f}' if cpu_freq else None,
        "Total RAM (GB)": f'{total_ram_gb:.2f}',
        "Available RAM (GB)": f'{available_ram_gb:.2f}',
        "Total Storage Space (GB)": f'{total_disk_gb:.2f}',
        "Free Storage Space (GB)": f'{free_disk_gb:.2f}',  
                   "GPUs": [{gpu.name: {"Total Memory (MB)": gpu.memoryTotal,
                                        "Free Memory (MB)": gpu.memoryFree,
                                        "Temperature (°C)": gpu.temperature}} for gpu in gpus] if gpus else None
    }]).reset_index(drop=True)

    return available_ram_gb, system_specs

# Prints updates on the simulation progress
def progress(**information
             ):
    
    if information['Status'] == 'simulation setup':
        print(f'{tm.bold}{tm.cyan}\nSimulation Setup Details:{tm.reset}')
        print(f'{tm.bold}{"="*26}{tm.reset}')
        for key, value in information.items():
            if key != 'Status':
                print(f'{tm.bold}{tm.cyan}{key} {value}{tm.reset}')
                time.sleep(0.1)
        print(f'{tm.bold}{tm.cyan}Setting up simulation parameter grid >>>{tm.reset}\n', end='\n')
        sim_grid_anchor = tm.get_current_position()
        time.sleep(5)
    
    elif information['Status'] == 'parameter grid':
        print(f'{tm.bold}{tm.cyan}\nSimulation Parameter Grid Details:{tm.reset}')
        print(f'{tm.bold}{"="*36}{tm.reset}')
        for key, value in information.items():
            if key != 'Status':
                print(f'{tm.bold}{tm.cyan}{key} {value}{tm.reset}')
                time.sleep(0.1)
        print(f'{tm.bold}{tm.cyan}Starting simulation loop >>>{tm.reset}')
        time.sleep(5)
        print(f'{tm.bold}{"-"*26}{tm.reset}\n', end='\n')
        par_grid_anchor = tm.get_current_position()
    
    elif information['Status'] == 'batching':
        
        if information['Sample_Size'] >= 100:
            tm.refresh_terminal()
            print(f'{information["INFO"]}\n', end='\n', flush=True)
            
            if batch_flag == None:
                batch_anchor = tm.get_current_position()
                batch_flag = 1
            print(f'\r{tm.clear_below()}{information["PROGRESS"]}\n', end='\n', flush=True)
            
        else:
            print(f'\r{tm.clear_below()}{information["PROGRESS"]}\n', end='', flush=True)
        time.sleep(0.2)

    elif information['Status'] == 'estimation':
        tm.refresh_terminal()
        if information['Batch_count'] < information['Total_batches']:
            if information['Total_batches'] - information['Batch_count'] > 1:
                update = f'{tm.bold}{tm.cyan}PROGRESS: Batch {information['Batch_count']} of {information['Total_batches']} completed. {information['Total_batches'] - information['Batch_count']} batches remaining >>>{tm.reset}' 
            else:
                 update = f'{tm.bold}{tm.cyan}PROGRESS: Batch {information['Batch_count']} of {information['Total_batches']} completed. {information['Total_batches'] - information['Batch_count']} batch remaining >>>{tm.reset}'
        else:
            update = f'{tm.bold}{tm.cyan}PROGRESS: final batch {information['Batch_count']} completed.{tm.reset}'
    
        print(f'\r{tm.clear_below()}{update}', end='', flush=True)
        time.sleep(0.2)

    elif information['Status'] in ['simulation', 'analysis', 'writer', 'plots']:
        tm.refresh_terminal()
        print(f'\r{tm.clear_below()}{information["PROGRESS"]}', end='', flush=True)
        time.sleep(0.2)

def parameter_grid(**information
                   ):

    progress(**information)
    
    information['Status'] = 'parameter grid'

    parameter_grid = list(product(*information['Simulation Grid Parameters']))

    grid_size = len(parameter_grid)
    iterations = grid_size * information['Replications']

    grid_info = {'Status':information['Status'],
                'Total parameter combinations':grid_size,
                 'Replications size':information['Replications'],
                 'Total iteration count':iterations
                }
    
    progress(**grid_info)

    return parameter_grid

def batcher(replication: int,
            sample_size: int,
            available_ram: float,
            cell: str,
            distr: str,
            method: str
            ):
    
    if sample_size >= 100:
        if available_ram < 16.00:
            if sample_size == 100:
                batch_scale = sample_size if method == 'Theil' else 1
            else: 
                batch_scale = sample_size if method == 'Theil' else sample_size // 100 
        
        elif available_ram < 32.00: 
            batch_scale = sample_size // 2 if method == 'Theil' else sample_size // 50 
        
        elif available_ram < 64.00: 
            batch_scale = sample_size // 4 if method == 'Theil' else sample_size // 25 
        
        else:
            batch_scale = 1
    else:
        batch_scale = 1 # For smaller sample sizes, we can process all iterations in one batch without memory issues
        
    batch_size = replication // batch_scale
    total_batches = replication // batch_size
                
    alpha = np.zeros((replication,1), dtype=np.float64)
    beta = np.zeros((replication,1), dtype=np.float64)
    std_err_alpha = np.zeros((replication,1), dtype=np.float64)
    std_err_beta = np.zeros((replication,1), dtype=np.float64)

    batch_info = {'Status':'batching',
                  'Sample_Size': sample_size,
                  'INFO':f'{tm.bold}{tm.red}Due to the large sample size ({sample_size}) and replications ({replication}), the iterations will be run in {total_batches} batches of size {batch_size} each to ensure efficient running{tm.reset}',
                  'PROGRESS':f'{tm.bold}{tm.cyan}Simulation cell n = {sample_size} {cell} Model {distr} {method} is now in progress >>>{tm.reset}'}

    progress(**batch_info)

    return batch_size, total_batches, alpha.astype(np.float64), beta.astype(np.float64), std_err_alpha.astype(np.float64), std_err_beta.astype(np.float64)