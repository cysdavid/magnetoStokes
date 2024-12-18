'''
Script (Dedalus v3) for 3D mixing simulations in Section 5.2 of David et al. (2024).
Run using the driver script 'run_magstokes_3D_mixing.py'

Reference
---------
C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
Magneto-Stokes Flow in a Shallow Free-Surface Annulus. J. Fluid Mech. (2024).
https://doi.org/10.1017/jfm.2024.674
'''

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import glob
import h5py
import file_tools as flt
import magstokes_soln_minimal as mhdstokes
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

alt_dim = {'r':'z','z':'r'}

def create_comp_domain(params,dtype=np.float64):
    domain = {}
    domain['coords'] = coords = d3.CartesianCoordinates('θ','r','z')
    domain['dist'] = dist = d3.Distributor(coords, dtype=dtype)
    domain['θb'] = θbasis = d3.RealFourier(coords['θ'], size=params['Nθ'], bounds=(0, 2*np.pi), dealias=params['dealias'])
    domain['rb'] = rbasis = d3.ChebyshevT(coords['r'], size=params['Nr'], bounds=(params['R'], 1), dealias=params['dealias'])
    domain['zb'] = zbasis = d3.ChebyshevT(coords['z'], size=params['Nz'], bounds=(0, 1), dealias=params['dealias'])
    bases = (θbasis,rbasis,zbasis)
    domain['θa'], domain['ra'], domain['za'] = θa, ra, za = dist.local_grid(θbasis,scale=params['dealias']), dist.local_grid(rbasis,scale=params['dealias']), dist.local_grid(zbasis,scale=params['dealias'])
    
    domain['θ'] = θf = dist.Field(name='θ', bases=domain['θb'])
    domain['r'] = rf = dist.Field(name='r', bases=domain['rb'])
    domain['z'] = zf = dist.Field(name='z', bases=domain['zb'])
    domain['θfull'] = θfull = dist.Field(name='θθ', bases=bases)
    domain['rfull'] = rfull = dist.Field(name='rr', bases=bases)
    domain['zfull'] = zfull = dist.Field(name='zz', bases=bases)
    for dim in 'θrz':
        domain[dim].change_scales(params['dealias'])
        domain[dim+'full'].change_scales(params['dealias'])
        domain[dim]['g'] = domain[dim+'a']
        domain[dim+'full']['g'] = domain[dim+'a']
    return domain

def create_tau_bases(domain,k=2):
    # Tau parameters
    domain['tau_params'] = ks = {'tau_int':k, 'lift_int':k, 'lift_edge':k}
    domain['tau_bases'] = {(dim,typ): domain[dim+'b'].derivative_basis(k) for dim in 'rz' for typ, k in ks.items()}
    return domain

def create_tau_terms(domain,vector=False,name=''):
    if vector:
        τs = {(dim,i): domain['dist'].VectorField(name=f'τ{name}{dim}{i}', bases=(domain['θb'],domain['tau_bases'][dim,'tau_int']), coordsys=domain['coords']) for dim in 'rz' for i in range(2)}
        for i in range(4): τs[i] = domain['dist'].VectorField(name=f'τ{name}{i}', coordsys=domain['coords'], bases=(domain['θb']))
    else:  # scalar variables
        τs = {(dim,i): domain['dist'].Field(name=f'τ{name}{dim}{i}', bases=(domain['θb'],domain['tau_bases'][dim,'tau_int'])) for dim in 'rz' for i in range(2)}
        for i in range(4): τs[i] = domain['dist'].Field(name=f'τ{name}{i}',bases=(domain['θb']))

    k_int, k_edge = domain['tau_params']['lift_int'], domain['tau_params']['lift_edge']
    τbs = domain['tau_bases']
    taus = {}
    taus['int'] = sum(d3.Lift(τs[dim,i],τbs[alt_dim[dim],'lift_int'],j) for dim in 'rz' for i,j in zip([0,1],[-1,-2]))
    taus['t'] = d3.Lift(τs[2], τbs['r','lift_edge'], -1) + d3.Lift(τs[0], τbs['r','lift_edge'], -2)
    taus['b'] = d3.Lift(τs[3], τbs['r','lift_edge'], -1) + d3.Lift(τs[1], τbs['r','lift_edge'], -2)
    return τs, taus
    

def run_magnetocouette_advection(params):
    domain = create_comp_domain(params)
    domain = create_tau_bases(domain)

    τs, taus = create_tau_terms(domain,vector=False)
    t = domain['dist'].Field(name='t')
    c = domain['dist'].Field(name='c',bases=(domain['θb'],domain['rb'],domain['zb']))
    uθ = domain['dist'].Field(name='uθ',bases=(domain['θb'],domain['rb'],domain['zb']))

    θ,r,z = domain['θ'], domain['r'], domain['z']
    for f in θ,r,z: f.change_scales(params['dealias'])
    dθ,dr,dz = [lambda f, c=c: d3.Differentiate(f,domain['coords'][c]) for c in 'θrz']
    K, R, H = params['K'], params['R'], params['H']

    advection = (1+R)*r*uθ*dθ(c)
    evolution = r**2*d3.dt(c)
    diffusion = -K*(r**2*dz(dz(c)) + H**2*(r*dr(c) + r**2*dr(dr(c)) + dθ(dθ(c))))
    interior = (evolution + taus['int'] + diffusion, -advection)
    top = (dz(c)(z=1.) + taus['t'],0)
    bot = (dz(c)(z=0.) + taus['b'],0)
    lft = (dr(c)(r=R),0)
    rgt = (dr(c)(r=1.),0)
    bcs = [top,bot,lft,rgt]
    taubcs = [
        (τs['r',0](r=R),0),
        (τs['r',0](r=1),0),
        (τs['r',1](r=R),0),
        (τs['r',1](r=1),0),
    ]
    
    problem = d3.IVP([c] + list(τs.values()), namespace=locals(),time=t)
    problem.add_equation(interior)
    for bc in bcs + taubcs:
        problem.add_equation(bc)

    solver = problem.build_solver(getattr(d3,params['timestepper']), ncc_cutoff=1e-10)

    # Solver
    solver.stop_sim_time = params['stop_sim_time']
    
    # Initial conditions
    uθ.change_scales(params['dealias'])
    c.change_scales(params['dealias'])

    rr, θθ, zz = domain['rfull']['g'].copy(), domain['θfull']['g'].copy(), domain['zfull']['g'].copy()
    uθ['g'] = mhdstokes.uSteadyND(rr,zz,H,R,l=params['N_analyt'])
    
    def ramp(x,hw,ew):
        return 1/(1 + np.exp(-(ew*(2*hw + ew)*(hw**2 - 2*x**2 + (hw + ew)**2))/((hw**2 - x**2)*(x**2 - (hw + ew)**2))))

    halfwidth = params['streak_halfwidth']
    edgewidth = 2*np.pi/params['Nθ']*params['N_streak_edge']

    c['g'] = ramp(θθ-np.pi,halfwidth-edgewidth/(2*rr),edgewidth/rr)
    c['g'][(θθ-np.pi)**2 <= (halfwidth - edgewidth/(2*rr))**2] = 1
    c['g'][(θθ-np.pi)**2 >= (halfwidth + edgewidth/(2*rr))**2] = 0

    # Mixing norm
    c0 = domain['dist'].Field(name='c0',bases=(domain['θb'], domain['rb'],domain['zb']))
    c0.change_scales(params['dealias'])
    c0['g'] = c['g'].copy()

    integrate = lambda f: d3.integ(r*f)
    cmean = integrate(c)/(np.pi*(1-R**2))
    cmean0 = cmean.evaluate()
    mix_norm = (integrate((c-cmean)**2)/integrate((c0-cmean0)**2))**(1/2)

    # L2 norms
    l2_θ = lambda f: d3.integ(f**2,'θ')/(2*np.pi)
    l2_r = lambda f: d3.integ(r*f**2,'r') * 2/(1-R**2)
    l2_z = lambda f: d3.integ(f**2,'z')
    l2_θz = lambda f: d3.integ(f**2,('θ','z'))/(2*np.pi)
    l2_rz = lambda f: d3.integ(r*f**2,('r','z')) * 2/(1-R**2)
    l2_rθ = lambda f: d3.integ(r*f**2,('r','θ'))/(np.pi*(1-R**2))
    l2_rθz = lambda f: d3.integ(r*f**2)/(np.pi*(1-R**2))

    # Flow-normal average
    mean_rz = lambda f: d3.integ(r*f,('r','z')) * 2/(1-R**2)

    # Analysis                                                                                      
    starting = solver.evaluator.add_file_handler(f'DNS-data/3DMixing/data/starting-{params["sim_name"]}', sim_dt=-1, max_writes=50)
    starting.add_tasks(solver.state)
    starting.add_task(uθ)
    starting.add_task(mean_rz(uθ/r),name="omega_avg")

    snapshots_3D = solver.evaluator.add_file_handler(f'DNS-data/3DMixing/data/snapshots-3D-{params["sim_name"]}', sim_dt=params['save_step_3D'], max_writes=50)
    snapshots_3D.add_tasks(solver.state)
    snapshots_3D.add_task(uθ)
    
    snapshots_2D = solver.evaluator.add_file_handler(f'DNS-data/3DMixing/data/snapshots-2D-{params["sim_name"]}', sim_dt=params['save_step_2D'], max_writes=200)
    for field in [uθ,c]:
        for i, θi in enumerate(np.linspace(0,2*np.pi,9)[:-1]):
            snapshots_2D.add_task(field(θ=θi),name=field.name+f"_θ{i:d}")
        for i, ri in enumerate(np.linspace(R,1,5)):
            snapshots_2D.add_task(field(r=ri),name=field.name+f"_r{i:d}")
        for i, zi in enumerate(np.linspace(0,1,5)):
            snapshots_2D.add_task(field(z=zi),name=field.name+f"_z{i:d}")
        snapshots_2D.add_task(d3.integ(field,domain['coords']['θ']),name=field.name+f"_θ_int")
        snapshots_2D.add_task(d3.integ(field,domain['coords']['r']),name=field.name+f"_r_int")
        snapshots_2D.add_task(d3.integ(field,domain['coords']['z']),name=field.name+f"_z_int")

    snapshots_spectra = solver.evaluator.add_file_handler(f'DNS-data/3DMixing/data/snapshots-spectra-{params["sim_name"]}', sim_dt=params['save_step_spectra'], max_writes=200)
    snapshots_spectra.add_task(c,name='ccoeff',layout='c')
    snapshots_spectra.add_task(l2_θz(c),name="cl2_θz_coeff",layout='c')
    snapshots_spectra.add_task(l2_rz(c),name="cl2_rz_coeff",layout='c')
    snapshots_spectra.add_task(l2_rθ(c),name="cl2_rθ_coeff",layout='c')
    snapshots_spectra.add_task(l2_θ(c),name="cl2_θ_coeff",layout='c')
    snapshots_spectra.add_task(l2_r(c),name="cl2_r_coeff",layout='c')
    snapshots_spectra.add_task(l2_z(c),name="cl2_z_coeff",layout='c')

    snapshots_1D = solver.evaluator.add_file_handler(f'DNS-data/3DMixing/data/snapshots-1D-{params["sim_name"]}', sim_dt=params['save_step_1D'], max_writes=50)
    snapshots_1D.add_task(mean_rz(c), name="cmean_rz")

    snapshots_0D = solver.evaluator.add_file_handler(f'DNS-data/3DMixing/data/snapshots-0D-{params["sim_name"]}', sim_dt=params['save_step_0D'], max_writes=50)
    snapshots_0D.add_task(mix_norm, name="mix_norm")
    snapshots_0D.add_task(cmean, name="cmean")
    snapshots_0D.add_task(l2_rθz(c), name="cl2_rθz")
    snapshots_0D.add_task(mean_rz(uθ/r),name="omega_avg")

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(uθ**2), name='speed')
    flow.add_property(c, name='conc')
    flow.add_property(mix_norm, name="mix_norm")
        
    solver.step(params['timestep'])
    start_time = time.time()
    
    uθ.change_scales(params['dealias'])
    uθ['g'] = mhdstokes.uSteadyND(rr,zz,H,R,l=params['N_analyt'])
    try:
        while solver.proceed:
            # uθ.change_scales(params['dealias'])
            # uθ['g'] = mhdstokes.uND(rr,zz,solver.sim_time,H,R,1/params['N'])
            if solver.iteration % params['print_freq'] == 0:

                log = [f'it {solver.iteration:d}',
                       f'sim time {solver.sim_time:.2f}',
                       f'wall time {(time.time() - start_time):.1f} s',
                       f'max u {flow.max("speed"):.3f}',
                       f'max c {flow.max("conc"):.3f}',
                       f'mix norm {flow.max("mix_norm"):.3f}',
                       ]
                logger.info(', '.join(log))
                if np.isnan(flow.max("conc")):
                    logger.info('NaN!')
                    break
            solver.step(params['timestep'])
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise     
    return
    

def list_saves(files):
    sorted_files = sorted(files, key=lambda s: int(s.split('_s')[1].split('.')[0]))
    saves = []
    for i, f in enumerate(sorted_files):
        t, it = flt.load_data(f, 'sim_time','iteration',group='scales',asscalar=False)
        for j, (tj, itj) in enumerate(zip(t,it)):
            saves.append({
                'file_index':i, 
                'file':f, 
                'save_index':j, 
                'sim_time':tj, 
                'sim_it':itj})
    for i, save in enumerate(saves):
        save['index'] = i
    return saves    
    
    
def plot_magnetocouette_advection(params):
    # init_file = sorted(glob.glob(f'DNS-data/3DMixing/data/starting-{params["sim_name"]}/*.h5'))[0]
    data_files_3D = sorted(glob.glob(f'DNS-data/3DMixing/data/snapshots-3D-{params["sim_name"]}/*.h5'))
    data_files_2D = sorted(glob.glob(f'DNS-data/3DMixing/data/snapshots-2D-{params["sim_name"]}/*.h5'))
    data_files_spectra = sorted(glob.glob(f'DNS-data/3DMixing/data/snapshots-spectra-{params["sim_name"]}/*.h5'))

    saves_2D = list_saves(data_files_2D)
    saves_spectra = list_saves(data_files_spectra)

    plot_dir = f'DNS-data/3DMixing/plots/{params["sim_name"][:-4]}/{params["sim_name"]}'
    if rank == 0:
        flt.makedir(plot_dir)
        flt.makedir(plot_dir+'/slices')
        flt.makedir(plot_dir+'/spectra')

    with h5py.File(data_files_3D[0], 'r') as f:
        θ, r, z = [f['tasks']['c'].dims[i][n][:] for i, n in [(1,'θ'),(2,'r'),(3,'z')]]
        umax, cmax = f['tasks']['uθ'][:].max(), f['tasks']['c'][:].max()

    θθθ, rrr, zzz = np.meshgrid(θ, r, z, indexing='ij')
    grids = {}
    grids['z'] = (θz, rz) = np.meshgrid(θ, r, indexing='ij')
    grids['r'] = (θr, zr) = np.meshgrid(θ, z, indexing='ij')
    grids['θ'] = (rθ, zθ) = np.meshgrid(r, z, indexing='ij')
    
    cmaps = {'uθ':'RdBu_r', 'c':'Purples'}
    ranges = {'uθ': [0., umax], 'c': [0., cmax]}
    labels = {'r': {'xlabel': '$θ$', 'ylabel': '$z$'},
              'θ': {'xlabel': '$r$', 'ylabel': '$z$'},
              'z': {'xlabel': '$θ$', 'ylabel': '$r$'}}
    projection = {'r': None, 'z': 'polar', 'θ': None}
    slices = {'r': ['0','1','2','3','4','_int'],
              'z': ['0','1','2','3','4','_int'],
              'θ': ['0','1','2','3','4','5','6','7','_int']}

    for save_2D in saves_2D[rank::size]:
        f, it, findex = save_2D['file'], save_2D['index'], save_2D['save_index']
        fields_2D = {key: flt.load_data(f,key,group='tasks',sel=np.s_[findex,...])[0] for key in flt.get_keys(f,group='tasks')}
        for coord_index, coord in enumerate(['θ','r','z']):
            fig, ax = plt.subplots(2, len(slices[coord]), figsize=(12,12), subplot_kw={'projection':projection[coord]})
            ps = {}
            for field_index, field in enumerate(['c','uθ']):
                ax_row_index = field_index
                for slice_index, slce in enumerate(slices[coord]):
                    ax_col_index = slice_index
                    axes_index = (ax_row_index, ax_col_index)
                    arr = fields_2D[f'{field}_{coord}{slce}']
                    grid = grids[coord]
                    ps[axes_index] = ax[axes_index].pcolormesh(*grid, np.squeeze(arr), vmin=ranges[field], cmap=cmaps[field])
                # plt.colorbar(ps[axes_index],ax=ax[ax_row_index,:])#,orientation='horizontal')
            plt.savefig(f'{plot_dir}/slices/{params["sim_name"]}-slices-{coord}-slices-{it:0>3d}.png',bbox_inches='tight',dpi=600)
            plt.close()

    for save_spectra in saves_spectra[rank::size]:
        f, it, findex = save_spectra['file'], save_spectra['index'], save_spectra['save_index']
        fields_spectra = {key: flt.load_data(f,key,group='tasks',sel=np.s_[findex,...])[0] for key in flt.get_keys(f,group='tasks')}
        fields_spectra['cl2_r_coeff']=fields_spectra['cl2_r_coeff'][:,0,:]
        fields_spectra['cl2_z_coeff']=fields_spectra['cl2_z_coeff'][:,:,0]
        
        cl2_rz_coeff,cl2_rθ_coeff,cl2_θz_coeff=[fields_spectra['cl2_rz_coeff'],fields_spectra['cl2_rθ_coeff'],fields_spectra['cl2_θz_coeff']]
        kθ,kr,kz = np.arange(len(cl2_rz_coeff)),np.arange(len(cl2_θz_coeff)),np.arange(len(cl2_rθ_coeff))
        kr_mesh,kθ_mesh,kz_mesh = np.meshgrid(kr,kθ,kz)
        l2norm_k = np.sqrt(kr_mesh**2 + kθ_mesh**2 + kz_mesh**2)

        with np.errstate(divide='ignore'):
            axes = ['z','r','θ']
            ax_spec = []
            for ax in axes:
                ax_spec.append(ax)
                ax_spec.append(ax)
            fig,axs = plt.subplot_mosaic([ax_spec,['3D','3D','3D','1D','1D','1D']],figsize=(15,8))
            for e in axes:
                Z = np.abs(fields_spectra[f'cl2_{e}_coeff'])
                im = axs[e].pcolor(Z,norm=colors.LogNorm(vmin=np.min(Z[Z!=0]), vmax=Z.max()))
                fig.colorbar(im,ax=axs[e],label=f'$\\text{{abs}}(\\widehat{{\\Vert c \\Vert_{e}^2}}\\;)$')
                rem_axes = np.array(axes)[np.array(axes)!=e]
                axs[e].set_xlabel(f'{rem_axes[0]} mode number')
                axs[e].set_ylabel(f'{rem_axes[1]} mode number')
            axs['3D'].scatter(l2norm_k.flatten(),np.abs(fields_spectra['ccoeff'].flatten()),alpha=0.2,s=0.5)
            axs['3D'].set_yscale('log')
            axs['3D'].set_xlabel('$\\Vert(k_r,k_\\theta,k_z)\\Vert$')
            axs['3D'].set_ylabel('$\\text{abs}(\\widehat{c})$')

            axs['1D'].plot(kθ,np.abs(fields_spectra['cl2_rz_coeff']),label='$\\text{abs}(\\widehat{\\Vert c \\Vert_{r,z}}^2\\;)\\;(k_\\theta)$')
            axs['1D'].plot(kr,np.abs(fields_spectra['cl2_θz_coeff']),label='$\\text{abs}(\\widehat{\\Vert c \\Vert_{\\theta,z}}^2\\;)\\;(k_r)$')
            axs['1D'].plot(kz,np.abs(fields_spectra['cl2_rθ_coeff']),label='$\\text{abs}(\\widehat{\\Vert c \\Vert_{r,\\theta}}^2\\;)\\;(k_z)$')
            axs['1D'].set_yscale('log')
            axs['1D'].set_xlabel('mode number')
            axs['1D'].legend()

            plt.suptitle(f'$(N_r,N_\\theta,N_z)=({len(kr)},{len(kθ)},{len(kz)}),\\;\\;t={save_spectra['sim_time']:.2f}$',y=0.95,fontsize=15)
            plt.subplots_adjust(wspace=1,hspace=0.2)
            plt.savefig(f'{plot_dir}/spectra/{params["sim_name"]}-spectra-{it:0>3d}.png',bbox_inches='tight',dpi=300)
            plt.close()

        print(rank, it)
