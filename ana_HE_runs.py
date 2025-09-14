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
from invisible_cities.types.symbols    import NormStrategy
from invisible_cities.types.ic_types   import NN

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
run_number  = args.run_number #15589 
map_name    = args.map_name #'combined_15546_15557.map3d' 
save_name   = args.save_name #'' 
ldc_name    = args.ldc_name #'ldc1' 
drop_nhits  = args.drop_nhits #3 #
corr_type   = args.corr_type #"3D" #

#just add ldc tag to saved file if ldc is not all
if ldc_name != '*':
    save_name = ldc_name + save_name

drop_cluster_dim = args.dimension #3 


source_path = '/mnt/netapp1/Store_next_data/NEXT100/data/{run_n}/hdf5/prod/*/*/sophronia/trigger2/'.format(run_n = run_number)
data_path = source_path + '{}/*'.format(ldc_name)

store_path  = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/HE_ana_runs/' 
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

def redistribute_energy(group: pd.DataFrame) -> pd.DataFrame:
    """
    Funtion that redistributes energy per slice given the Q of the SiPMs
    """
    tot_E = group.E.sum()
    mask = group['drop'].values
    drp = group[mask]
    srv = group[~mask]

    if drp.empty:
        # no hits to distribute
        return srv

    if srv.empty:
        # hits turned NN
        drp = drp.copy()
        drp[['X', 'Y', 'Q']] = NN  # or NN if defined
        return drp

    # redistribute
    srv = srv.copy()
    srv['E'] = (srv.Q / srv.Q.sum()) * tot_E
    return srv

def merge_NN_hits(hits: pd.DataFrame, same_peak: bool = True) -> pd.DataFrame:
    # quickly split NN vs normal
    sel = hits.Q.eq(NN)
    if not sel.any():
        return hits

    normal = hits[~sel].copy()
    nn     = hits[sel]

    if normal.empty:
        # nothing to receive: drop all NN
        return normal

    # For each NN, find candidate receivers and closest distance
    # Build a mapping of receiver index -> energy/energy_correction
    corr = pd.DataFrame(0.0, index=normal.index, columns=["E", "Ec"])

    # precompute distances matrix only once if same_peak=False
    if same_peak:
        # group normal hits by npeak to avoid repeated filtering
        normal_groups = {p: g for p, g in normal.groupby("npeak")}
    else:
        z_normal = normal.Z.values
        idx_normal = normal.index.values

    for _, row in nn.iterrows():   # still a loop over NN hits, but usually few
        if same_peak:
            cand = normal_groups.get(row.npeak)
            if cand is None or cand.empty:
                continue
            dz = (cand.Z - row.Z).abs()
            closest = cand.loc[np.isclose(dz, dz.min())]
        else:
            dz = np.abs(z_normal - row.Z)
            m  = np.isclose(dz, dz.min())
            closest = normal.loc[idx_normal[m]]

        wE  = closest.E / closest.E.sum()
        wEc = closest.Ec / closest.Ec.sum()
        corr.loc[closest.index, "E"]  += row.E  * wE
        corr.loc[closest.index, "Ec"] += row.Ec * wEc

    normal[["E","Ec"]] += corr
    return normal

def drop_isolated_clusters(distance: List[float] = [16., 16., 4.], nhit: int = 3) -> Callable:
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
        drop_df = df.loc[~mask].copy()
        # mask them to redistribute their energy after
        pass_df['drop'] = False
        drop_df['drop'] = True

        # recover the hits that are inside the Z range, because we want their energy to be taken into account
        if not pass_df.empty:
            zmin, zmax = pass_df.Z.min(), pass_df.Z.max()
            inside_mask = (drop_df.Z >= zmin) & (drop_df.Z <= zmax)
            drop_inrange = drop_df.loc[inside_mask]
            if not drop_inrange.empty:
                pass_df = pd.concat([pass_df, drop_inrange], axis=0)

        # at this point I have the df with the energy I should use (ener outsize Z range is deleted)
        # now we redistribute the energy of the dropped in range hits per slice using the charge
        pass_df = (pass_df.groupby(['Z'], group_keys=False)
                        .apply(redistribute_energy)
                        .reset_index(drop=True))
        # and finally redistribute the energy of the hits that were alone in a slice
        pass_df = merge_NN_hits(pass_df)
        return pass_df

    return drop

def get_corr(filename):
    krmap = pd.read_hdf(filename, "/krmap")
    meta  = pd.read_hdf(filename, "/mapmeta")
    dtxy_map   = krmap.loc[:, list("zxy")].values
    factor_map = krmap.factor.values
    def corr(x, y, dt, t, method="nearest"):
        dtxy_data   = np.stack([dt, x, y], axis=1)
        factor_data = griddata(dtxy_map, factor_map, dtxy_data, method=method)
        return factor_data
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


def create_hits_summary(reco, corr_fun, fname):
    # Correct energy
    factor = corr_fun(reco.X, reco.Y, reco.Z, reco.time)
    #correct energy in the borders (for NaN hits)
    factor_border = corr_fun([479],[0],[0],[1])
    reco['Ec'] = reco.E * factor
    reco["Ec_border"] = reco.E * factor_border
    # add the latter energy
    reco["Ec"] = reco["Ec"].fillna(reco["Ec_border"])
    # do summary
    reco_summary = reco.groupby(['event', 'npeak']).apply(lambda group: hits_summary(group, fname)).reset_index()

    return reco_summary

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

dropper = drop_isolated_clusters(distance = dist, nhit = drop_nhits)

# time_to_Z = get_df_to_z_converter(maps) if maps.t_evol is not None else identity

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
    
    # drop hits and redistribute energy
    reco = reco.groupby(['event', 'npeak'], group_keys=False).apply(dropper)
    # correct energy and create summary
    reco_summary = create_hits_summary(reco, get_coef, f)
        
    dst.to_hdf (save_path_summary, key = 'DST/Events', mode = 'a', append= True, complib="zlib", complevel=4)
    reco_summary.to_hdf(save_path_summary, key = 'RECO/Events_summary', mode = 'a', append= True, complib="zlib", complevel=4)

    print(i)
