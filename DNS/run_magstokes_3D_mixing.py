'''
Driver script for running 'magstokes_3D_mixing.py' with Dedalus v3.
See Section 5.2 of David et al. (2024).

Example usage:
    for i in {0..1}; do python run_magstokes_3D_mixing.py yoursurveyname $i; done

Reference
---------
C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
Magneto-Stokes Flow in a Shallow Free-Surface Annulus. J. Fluid Mech. (2024).
https://doi.org/10.1017/jfm.2024.674
'''

import dedalus.public as d3
import numpy as np
import file_tools as flt
import numpy as np
import pandas as pd
import logging
import sys
import os
import glob
import magstokes_3D_mixing as magmix
import magstokes_soln_minimal as magstokes
import sigfig as sf
root = logging.root
for h in root.handlers: h.setLevel("INFO") 
logger = logging.getLogger(__name__)
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

def create_dataframe(param_dic):
    """Convert dictionary of experiment parameters into multiindex of params for each experiment.
    Parameters paired in a tuple will be paired in the multiindex.
    E.g. {'A':[1,2], ('B','C'):([3,4],[-3,-4]),'D':[0]} ->
    A   B   C   D
    1   3  -3   0
    1   4  -4   0
    2   3  -3   0
    2   4  -4   0
    """
    tuples = []
    param_lists = {}
    for key in param_dic:
        if isinstance(key, str):
            param_lists[key] = param_dic[key]
        elif isinstance(key, tuple):
            tuples.append(key)
            param_lists[key[0]] = list(range(len(param_dic[key][0])))
            for keyi in key[1:]:
                param_lists[keyi] = [pd.NA]

    params = pd.MultiIndex.from_product(param_lists.values(), names=param_lists.keys())
    params = pd.DataFrame(index=params).reset_index()

    for tup in tuples:
        for column in tup[1:]:
            params[column] = params[tup[0]]
        for ind, column in enumerate(tup):
            params[column] = params[column].apply(lambda j: param_dic[tup][ind][j])

    return params

series = sys.argv[1]
index = int(sys.argv[2])
save_dir = f'data/{series}'

def dt_CFL(params):
    Rarr,Harr,Nθarr,Carr = params['R'], params['H'], params['Nθ'], params['CFL_number']
    dtarr = np.zeros(len(Rarr))
    for i in range(len(Rarr)):
        R,H,Nθ,Cour = Rarr[i],Harr[i],Nθarr[i],Carr[i]
        rarr,zarr = np.linspace(R,1,1000),np.linspace(0,1,1000)
        rr,zz = np.meshgrid(rarr,zarr)
        l = params['N_analyt'][i]
        uu = magstokes.uSteadyND(rr,zz,H,R,l=l)
        umax = np.max(uu)
        rmax = rr[uu==umax][0]
        dtarr[i] = sf.round(1/umax * (2*np.pi*rmax)/Nθ * Cour,sigfigs=2)
    return dtarr

def compute_K(params):
    Pearr,PeDeeparr,Rarr,Harr, = params['Pe'],params['PeDeep'],params['R'],params['H']
    Karr = np.zeros(len(Pearr))
    for i in range(len(Pearr)):
        Pe,PeDeep,R,H = Pearr[i],PeDeeparr[i],Rarr[i],Harr[i]
        if (~np.isnan(Pe))&(np.isnan(PeDeep)):
            Karr[i]=1/Pe
        elif (np.isnan(Pe))&(~np.isnan(PeDeep)):
            Pe = PeDeep * 4*H**4/(1-R)**4
            Karr[i] = 1/Pe
        else:
            print(f'Pe={Pe}, PeDeep={PeDeep}')
            raise ValueError('Missing or conflicting values of Pe and PeDeep')
    return Karr

def compute_stop_sim_time(params):
    stop_T_arr,stop_T_deep_arr,Rarr,Harr, = params['stop_T'],params['stop_T_deep'],params['R'],params['H']
    stop_sim_time_arr = np.zeros(len(Rarr))
    for i in range(len(Rarr)):
        stop_T,stop_T_deep,R,H = stop_T_arr[i],stop_T_deep_arr[i],Rarr[i],Harr[i]
        if (~np.isnan(stop_T))&(np.isnan(stop_T_deep)):
            stop_sim_time_arr[i]=stop_T
        elif (np.isnan(stop_T))&(~np.isnan(stop_T_deep)):
            stop_sim_time_arr[i]=stop_T_deep * 4*H**2/(1-R)**2
        else:
            raise ValueError('Missing or conflicting values of stop_T and stop_T_deep')
    return stop_sim_time_arr

param_list = {
    ('Nr', 'Nθ', 'Nz'): ([64],[256],[128]),
#     'Nθ': [32],
#     'Nz': [16],
    'N_analyt':[46],
    'R': [0.9], # radius ratio
    'H': [0.6], # aspect ratio
    'Pe': [10000., 14000., 19000., 27000., 37000., 52000., 72000.,
           100000., 140000., 190000., 270000., 370000., 520000., 720000.,
           1000000.,1400000., 1900000., 2700000.],# Peclet number, shallow. Pe = (H^2/κ)/((ri + ro)/U)
    'PeDeep': [None], # Peclet number, deep. Pe_deep = ((ro-ri)^2/κ)/((ri + ro)/U_deep)
    'stop_T':[50*30*np.pi],
    'stop_T_deep': [None], # Stop time in deep time units, Tdeep
    'streak_halfwidth': [np.pi/2],
    'N_streak_edge': [12],
    'dealias': [3/2],
    'mesh':[None],
    'CFL_number': [0.25],
    'timestep': [None],
    'timestepper':['RK443'],
    'save_step_3D': [50*0.25*np.pi],
    'save_step_2D': [50*0.25*np.pi],
    'save_step_1D': [50*0.2],
    'save_step_0D': [50*0.02],
    'save_step_spectra':[50*1],
    'max_writes': [10],
    'print_freq': [1],
    # 'sim_name':['test-0'],
    'save_dir': [save_dir],
}

# Create directories
flt.makedir('DNS-data/3DMixing/data/')
flt.makedir('DNS-data/3DMixing/parameters/')

# Make parameter dataframe
params = create_dataframe(param_list)

# Use CFL condition to compute timestep
params['timestep']= dt_CFL(params)

# Compute inverse Peclet number, K
params['K'] = compute_K(params)

# Compute stop time in units of T (shallow scale)
params['stop_sim_time'] = compute_stop_sim_time(params)

params['sim_name'] = ['-'.join([series,f'{i:0>3d}']) for i in params.index]
# series_restart = 'ch-3D-comparison-1'
# params['restart_file'] = [last_save_file(f'{series_restart}-{i:0>3d}') for i in range(len(params))]

params.to_csv(f'DNS-data/3DMixing/parameters/parameters-{series}.csv')

magmix.run_magnetocouette_advection(params.loc[index])
magmix.plot_magnetocouette_advection(params.loc[index])
