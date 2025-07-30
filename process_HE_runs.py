import argparse
import glob
import numpy as np
import pandas as pd
import tables as tb
from   typing      import Callable
from   typing      import Optional
from   typing      import List
from   pandas      import DataFrame
from   pandas      import Series

from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata

from invisible_cities.reco.corrections import read_maps, apply_all_correction
from invisible_cities.types.symbols import NormStrategy

def get_args():
    parser = argparse.ArgumentParser(
        description="Process a NEXT100 run and build summaries")
    parser.add_argument(
        "-r", "-run", "--run",
        dest="run_number",
        type=int,
        required=True,
        help="Run number to analyse, e.g. 15107")
    parser.add_argument(
        "-m", "--map",
        dest="map_name",
        type=str,
        default='map_MC_4bar_15063.h5',
        help="Name of the correction map")
    parser.add_argument(
        "-d", "--dim",
        dest="dimension",
        type=int,
        default=3,
        help="Number of dimensions for clustering algorithm")
    parser.add_argument(
        "-w", "--wf",
        dest="wf_selection",
        action='store_true',
        help="Pass if you want the script to perform cuts + select WF. If not passed, only saves summary")
    parser.add_argument(
        "-s", "--save",
        dest="save_name",
        type=str,
        default="run_summary.h5",
        help="Name of the file to save")
    parser.add_argument(
        "-l", "--ldc",
        dest="ldc_name",
        type=str,
        default="*",
        help="Name of the ldc to run over")
    parser.add_argument(
        "-n", "--nhit",
        dest="drop_nhits",
        type=int,
        default=3,
        help="Number of hits to drop a cluster")
    parser.add_argument(
        "-c", "--corr",
        dest="corr_type",
        type=str,
        default="2D",
        help="Type of map, 2D or 3D")
    return parser.parse_args()

args        = get_args()
run_number  = args.run_number 
map_name    = args.map_name
save_name   = args.save_name
ldc_name    = args.ldc_name
drop_nhits  = args.drop_nhits
corr_type   = args.corr_type

#just add ldc tag to saved file if ldc is not all
if ldc_name != '*':
    save_name = ldc_name + save_name

drop_cluster_dim = args.dimension
wf_selection = args.wf_selection

energy_windows = [(0.47, 0.67), (1.55, 1.8), (2.6, 2.9)]

source_path = '/mnt/netapp1/Store_next_data/NEXT100/data/{run_n}/hdf5/prod/*/*/sophronia/trigger2/'.format(run_n = run_number)
data_path = source_path + '{}/*'.format(ldc_name)

store_path  = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/HE_runs/'
save_path_summary = store_path + '/{run_n}/'.format(run_n = run_number) + save_name
map_path = store_path + map_name

def lineal(x, a, b):
    return a * x + b

def in_range(x, min_, max_):
    return (x > min_) & (x < max_)

def get_fname_info(path):
    file = path.split('/')[-1]
    split_name = file.split('_')
    run_n, file_n, ldc_n = split_name[1], split_name[2], split_name[3]
    return run_n, file_n, ldc_n

def drop_isolated_clusters(distance: List[float] = [16., 16., 4.], nhit: int = 3,
                           variables: List[str] = []) -> Callable:
    '''
    If len(distance) == 2, it will perform on X, Y
    If len(distance) == 3, it will perform on X, Y, Z
    '''
    ndim = len(distance)
    dist = np.sqrt(ndim)

    def drop(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df
        coords = []

        coords.append(df.X.values / distance[0])
        coords.append(df.Y.values / distance[1])

        if ndim == 3:
            coords.append(df.Z.values / distance[2])

        coords = np.column_stack(coords)

        try:
            nbrs = NearestNeighbors(radius=dist, algorithm='ball_tree').fit(coords)
            neighbors = nbrs.radius_neighbors(coords, return_distance=False)
            mask = np.array([len(neigh) > nhit for neigh in neighbors])
        except Exception as e:
            print(f"Error in NearestNeighbors: {{e}}")
            return df.iloc[:0]  # fallback: return empty

        pass_df = df.loc[mask].copy()

        if not pass_df.empty and variables:
            with np.errstate(divide='ignore', invalid='ignore'):
                columns = pass_df.loc[:, variables]
                scale = df[variables].sum().values / columns.sum().values
                columns *= scale
                pass_df.loc[:, variables] = columns

        return pass_df

    return drop


def get_corr(filename, n_norm=1000):
    krmap = pd.read_hdf(filename, "/krmap")
    meta  = pd.read_hdf(filename, "/mapmeta")
    dtxy_map = krmap.loc[:, list("zxy")].values
    e0_map   = krmap.e0.values
    
    dt_norm   = np.random.rand(  n_norm)*1*meta.iloc[0].bin_size_z
    xy_norm   = np.random.rand(2,n_norm)*5*meta.iloc[0].bin_size_x
    dtxy_norm = np.stack([dt_norm, *xy_norm], axis=1)
    norm      = griddata(dtxy_map, e0_map, dtxy_norm, method="linear").mean()
    def corr(x, y, dt, t, method="nearest"): #t is not used but added just to be equal to the other corr function
        dtxy_data = np.stack([dt, x, y], axis=1)
        e_data   = griddata(dtxy_map, e0_map, dtxy_data, method=method)
        return norm / e_data
    return corr


def hits_summary(group, fname, coords = ['X', 'Y', 'Z'], ener = 'Ec'):
    def dcoord(group, coords, i):
        c = coords[i]
        return group[c].max() - group[c].min()
    def barycenter(group, coords, i):
        return (group[coords[i]] * group['Q']).sum() / group['Q'].sum() #changed to Q because makes more sense
    
    run_n, file_n, ldc_n = get_fname_info(fname)
    
    return pd.Series({
        'run_n': str(run_n),
        'file_n': str(file_n),
        'ldc_n': str(ldc_n),
        'time': group['time'].unique()[0],
        'dX': dcoord(group, coords, 0),
        'dY': dcoord(group, coords, 1),
        'dZ': dcoord(group, coords, 2),
        'Xmin': group[coords[0]].min(),
        'Ymin': group[coords[1]].min(),
        'Zmin': group[coords[2]].min(), 
        'Rmax': np.sqrt(group['X']**2 + group['Y']**2).max(),
        'Xmax': group[coords[0]].max(),
        'Ymax': group[coords[1]].max(),
        'Zmax': group[coords[2]].max(),
        'X_bary': barycenter(group, coords, 0),
        'Y_bary': barycenter(group, coords, 1),
        'Z_bary': barycenter(group, coords, 2),
        'total_E': group['E'].sum(),
        'total_energy': group[ener].sum(),
        'num_hits': int(len(group)),
    })

def get_dropped_hits_info(df, df_drop):
    '''
    Takes the original df and the df with dropped hits to extract all this info
    '''
    # from the original DF, select the dropped and non dropped hits
    dropped_hits = df.reset_index(drop=True).merge(df_drop.drop(columns = 'Ec').reset_index(drop=True), how='outer', indicator=True).query('_merge == "left_only"').drop(columns='_merge')
    nondrop_hits = df.reset_index(drop=True).merge(df_drop.drop(columns = 'Ec').reset_index(drop=True), how='outer', indicator=True).query('_merge == "both"').drop(columns='_merge')
    assert len(dropped_hits) + len(nondrop_hits) == len(df)

    # pick per non dropped event the zrange to select the dropped hits, merge and create the mask 
    z_ranges = nondrop_hits.groupby(['event', 'npeak']).agg(Zmin = ('Z', 'min'), Zmax = ('Z', 'max'))
    dropped_hits = dropped_hits.merge(z_ranges, on=['event', 'npeak'], how='left')
    inZrange_mask = dropped_hits['Z'].between(dropped_hits['Zmin'], dropped_hits['Zmax'])

    # select both dropped hits (in and out of the event)
    reco_drop_in = dropped_hits[inZrange_mask]
    reco_drop_out = dropped_hits[~inZrange_mask]
    #summarize their info
    info_in = reco_drop_in.groupby(['event', 'npeak']).agg(Ec_drop_in = ('Ec', 'sum'), nhits_drop_in = ('Ec', 'count')).reset_index()
    info_out = reco_drop_out.groupby(['event', 'npeak']).agg(Ec_drop_out = ('Ec', 'sum'), nhits_drop_out = ('Ec', 'count')).reset_index()
    return info_in, info_out

def create_hits_summary(reco, corr_fun, dropper, fname):
    # Correct energy
    factor = corr_fun(reco.X, reco.Y, reco.Z, reco.time)
    #correct energy in the borders (for NaN hits)
    factor_border = corr_fun([479],[0],[0],[1])

    reco['Ec'] = reco.E * factor
    reco["Ec_border"] = reco.E * factor_border
    # add the latter energy
    reco["Ec"] = reco["Ec"].fillna(reco["Ec_border"])

    reco_full = reco.copy()
    
    # drop isolated clusters
    reco = reco.groupby(['event', 'npeak'], group_keys=False).apply(dropper)

    # get dropped clusters info
    info_in, info_out = get_dropped_hits_info(reco_full, reco)

    # do summary
    reco_summary = reco.groupby(['event', 'npeak']).apply(lambda group: hits_summary(group, fname)).reset_index()

    # add dropped clusters info to summary
    reco_summary = reco_summary.merge(info_in, on = ['event', 'npeak'], how = 'outer').fillna({'nhits_drop_in': 0})
    reco_summary = reco_summary.merge(info_out, on = ['event', 'npeak'], how = 'outer').fillna({'nhits_drop_out': 0})
    reco_summary = reco_summary.dropna(thresh=5).astype({'num_hits':'int', 'nhits_drop_out': 'int', 'nhits_drop_in': 'int'})
    # quick fix but not the best solution...
    
    return reco_summary

def apply_dst_cuts(dst, s1_DT_cut = [0.32, 500]):
    # Had to decouple cuts 
    #1S1
    dst_S1 = dst[dst.nS1 == 1]
    nev_S1 = len(dst_S1.event.unique())
    #1S2
    dst_S2 = dst_S1[dst_S1.nS2 == 1]
    nev_S2 = len(dst_S2.event.unique())
    #DT > 0
    dst_DT = dst_S2[dst_S2.DT > 0].copy()
    nev_DT = len(dst_DT.event.unique())
    #alphas
    dst_DT['limS1e'] = lineal(dst_DT.DT, s1_DT_cut[0], s1_DT_cut[1])
    dst_al = dst_DT[dst_DT.S1e < dst_DT.limS1e]
    nev_al = len(dst_al.event.unique())
    return dst_DT, [nev_S1, nev_S2, nev_DT, nev_al]

def apply_fid_cuts(reco):
    # R
    reco_r = reco[in_range(reco.Rmax , 0, 450)]
    nev_r = len(reco_r.event.unique())
    # Z
    reco_fid = reco_r[in_range(reco_r.Zmin , 20, 1369*0.865) & in_range(reco_r.Zmax , 20, 1369*0.865)]
    nev_f = len(reco_fid.event.unique())
    return reco_fid, [nev_r, nev_f]

def apply_window_cuts(reco, windows = [(0.45, 0.65), (1.4, 1.8), (2.2, 3)]):
    combined_mask = np.zeros_like(reco.total_energy, dtype=bool)
    nev_win = []
    for wd in windows:
        mask = in_range(reco.total_energy, wd[0], wd[1])
        nev_win.append(len(reco[mask].event.unique()))
        combined_mask |= mask
    return reco[combined_mask], nev_win



files = sorted(glob.glob(data_path + '*'), key=lambda x: (x.split('/')[-2], int(x.split('/')[-1].split('_')[2])))

# Energy correction
if corr_type == "2D":
    maps = read_maps(map_path)

    get_coef  = apply_all_correction(maps
                                     , apply_temp = False
                                     , norm_strat = NormStrategy.kr)
if corr_type == "3D":
    get_coef = get_corr(map_path)


# Cluster dropping
if drop_cluster_dim == 2:
    dist = [16., 16.]
if drop_cluster_dim == 3:
    dist = [16., 16., 4.]

dropper = drop_isolated_clusters(distance = dist, nhit = drop_nhits, variables = ['Ec'])

# time_to_Z = get_df_to_z_converter(maps) if maps.t_evol is not None else identity

eff = np.array([0] * 10) # to compute efficiencies

for i, f in enumerate(files):
    nev = []
    try:
        dst = pd.read_hdf(f, 'DST/Events')
        reco = pd.read_hdf(f, 'RECO/Events')
    except Exception as e:
        print(f"Skipping corrupted/invalid file: {f}")
        print(f"Error: {e}")
        continue

    if reco.empty:
        print(f"Skipping empty file: {f}")
        continue

    if wf_selection:
        # For wf selection, we apply first some cuts in the DST so we do drop_clusters over less events
        nev_initial = len(dst.event.unique())
        nev.extend([nev_initial])
        # apply dst cuts
        dst, nev_dst = apply_dst_cuts(dst)
        nev.extend(nev_dst)
        reco = reco[np.isin(reco.event, dst.event.unique())]

        # check no dst is empty after cuts
        if dst.empty:
            print(f"DST empty after initial cuts: {f}")
            continue

    # create summary
    reco_summary = create_hits_summary(reco, get_coef, dropper, f)

    if wf_selection:
        # and also we directly deliver everything with cuts
        #apply fid cuts
        reco_summary, nev_reco = apply_fid_cuts(reco_summary)
        nev.extend(nev_reco)

        # apply energy window cuts
        reco_summary, nev_win = apply_window_cuts(reco_summary, windows=energy_windows)
        nev.extend(nev_win)
        dst = dst[np.isin(dst.event, reco_summary.event.unique())] # not saved but no problem

        #create df for wf selection (creo que me vale directamente el reco, porque ya tiene toda la info)
        selected_events = reco_summary.drop(columns = ['time', 'npeak',
                                                    'dX', 'dY', 'dZ', 
                                                    'Xmin', 'Ymin', 'Zmin', 
                                                    'Rmax', 'Xmax','Ymax','Zmax',
                                                    'X_bary','Y_bary','Z_bary',
                                                    'total_E', 'num_hits'])
        selected_events.to_hdf(save_path_summary, key = 'RUN/Selected_events', mode = 'a', append = True, complib="zlib", complevel=4)
        eff += np.array(nev)
        
    else: #decided not to save this for the WF selector
        dst.to_hdf (save_path_summary, key = 'DST/Events', mode = 'a', append= True, complib="zlib", complevel=4)
        reco_summary.to_hdf(save_path_summary, key = 'RECO/Events_summary', mode = 'a', append= True, complib="zlib", complevel=4)

    print(i)

if wf_selection:
    eff_df = pd.DataFrame([eff],columns = ['total', '1S1', '1S2', 'DT', 'alphas', 'rad', 'z', 'SP', 'DEP', 'PP'])
    eff_df.to_hdf(save_path_summary, 'RUN/eff', complib="zlib", complevel=4)
