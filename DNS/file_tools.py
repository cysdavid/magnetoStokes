# Author: Eric W. Hester

import numpy as np
import os
import h5py
from os.path import join
import pickle

def save_data(filename, dset, dname, group='/',overwrite=False):
    """Save array and name to a group in an hdf5 file.
    
    Parameters
    ----------
    
    dsets: numpy array
        dataset
    dnames: string
        dataset name
    filename: string
        file name
    group: string, optional
        subgroup of hdf5 file to write to
    overwrite: boolean, optional
    
    Returns
    -------
    None
    """
    with h5py.File(filename,'a') as f:
        if group not in f.keys(): f.create_group(group)
        g = f[group]
        if dname in g.keys(): 
            if overwrite: 
                g[dname][...] = dset
                return
            else: del g[dname]
        g[dname] = dset
    return

def make_group(filename,name,group='/'):
    """Make a group in an hdf5 file."""
    with h5py.File(filename,'a') as f:
        g = f[group]
        if name not in g.keys():
            g.create_group(name)
    return


def load_data(filename, *dnames, group='/',show=False,flatten=True,sel=None,asscalar=True,checkint=True):
    """Load list of arrays given names of group in an hdf5 file.
    
    Parameters
    ----------
    dnames: list
        strings of dataset names
    filename: string
        file name
    group: string, optional
        subgroup of hdf5 file to write to
    overwrite: boolean, optional
    show: boolean, optional
    flatten: boolean, optional
        return 1D array if empty dimensions
    asscalar: boolean, optional
        return scalar if size 1 array
    sel: slice object, optional
        return slice of data array
    checkint: return integer if integer scalar
    Returns
    -------
    List of numpy arrays

    """

    with h5py.File(filename,'r') as f:
        arrs = []
        g = f[group]
        for dname in dnames:
            if show: print(dname)    
            if not sel: sel = Ellipsis
            arr = g[dname][sel]
            if asscalar and arr.size==1: 
                arr = arr.item()
                if checkint and isinstance(arr,float) and arr.is_integer(): 
                    arr = int(arr)
            elif flatten:
                if np.prod(arr.shape) == max(arr.shape): 
                    arr = arr.flatten()                
                elif arr.shape[0] == 1: 
                    arr = arr[0,Ellipsis]
            arrs.append(arr)
    return arrs

def makedir(path):
    """Simple wrapper to make a path if it doesn't exist
    
    Parameters
    ----------
    path : string
    """
    if not os.path.isdir(path): os.makedirs(path)

def get_keys(filename, group='/'):
    """ Helper to get keys of an hdf5 file/group.

    Parameters
    ----------
    filename : string
        The hdf5 file name
    group : the subgroup of the file, optional
    """
    with h5py.File(filename, 'r') as f:
        g = f[group]
        keys = sorted(list(g.keys()))
    return keys

def print_keys(filename, group='/', formatted=False):
    """Wrapper to print keys of an hdf5 file/group.

    Parameters
    ----------
    filename : string
        The hdf5 file name
    group : the subgroup of the file, optional
    """
    keys = get_keys(filename, group=group)
    if formatted:
        for key in keys: print(key)
    else:
        print(keys)
    return

def delete(filename,*names,group='/'):
    """Delete file or group in an hdf5 file."""
    with h5py.File(filename,'a') as f:
        g = f[group]
        for name in names:
            if name in g.keys(): del g[name]
    return

def move(filename,origin,destination):
    """Move file or group in an hdf5 file."""
    with h5py.File(filename,'a') as f:
        f[destination] = f[origin]
    return

# Save individual bases
# Load domain stitches together from bases information
def save_basis(path,basis,order):
    """Save basis object info."""
    typ = type(basis).__name__
    name = basis.name
    size = basis.base_grid_size
    interval = basis.interval
    for arr, name in zip([typ,name,size,interval],
                         ['type','name','size','interval']): 
        save_data(path,arr,name,group='{}'.format(order))
        
    if typ == 'Compound':
        for suborder, subbasis in enumerate(basis.subbases):
                save_basis(path,subbasis,order='{}/{}'.format(order,suborder))

def save_domain(path,domain):
    """Save domain object bases info."""
    for i, basis in enumerate(domain.bases):
        save_basis(path,basis,order='{}'.format(i))

def load_basis(path,order,dealias=1):
    """Load basis from file."""
    from dedalus import public as de
    classes = {'Fourier':de.Fourier,'Chebyshev':de.Chebyshev,'SinCos':de.SinCos,'Compound':de.Compound}
    typ, name, size, interval = load_data(path, 'type','name','size','interval',group=order)
    typ, name = typ.decode("utf-8"), name.decode("utf-8")

    if typ == 'Compound':     # Recursively load compound basis
        suborders = sorted([key for key in get_keys(path,group=order) if key.isdigit()])
        subbases = [load_basis(path,'{}/{}'.format(order,suborder),dealias=dealias) for suborder in suborders]
        return classes[typ](name,subbases,dealias=dealias)
    
    return classes[typ](name,size,interval=interval,dealias=dealias)

def load_domain(path,dealias=1,comm=None):
    """Load domain object."""
    from dedalus import public as de
    if comm:
        from mpi4py import MPI
        if comm=='world': comm = MPI.COMM_WORLD
        elif comm=='self': comm = MPI.COMM_SELF
            
    orders = sorted(get_keys(path))
    bases = [load_basis(path,order,dealias=dealias) for order in orders]
    return de.Domain(bases, grid_dtype=np.float64, comm=comm)

def pickle_save(obj,name,dr=''):
    """Save python object with pickle."""
    if dr: makedir(dr)
    with open(join(dr,name+'.pickle'),'wb') as f:
        pickle.dump(obj,f)

def pickle_load(name,dr=''):
    """Load python object with pickle."""
    with open(join(dr,name+'.pickle'),'rb') as f:
        return pickle.load(f)
