import os
import sys
import prody as pr
import numpy as np
#You can either add the python package path.
#sys.path.append(r'/mnt/e/GitHub_Design/Metalprot')
from metalprot.search import search_selfcenter
from metalprot.basic import filter
import pickle

'''
python /mnt/e/GitHub_Design/Metalprot/scrips/position_ligand/belinostat/run_selfcenter_search.py local

'''

class Para():
     
    # resnums = [3, 7, 10, 14, 17, 18, 21, 24, 25, 
    #     51, 54, 58, 61, 65, 68, 69, 72, 77, 81, 84, 88, 91, 92, 95, 99, 
    #     125, 128, 132, 135, 139, 142, 146]
    #resnums = [3, 7, 72, 77]
    resnums = [x for x in range(2, 146)]

    win_filter = [('A', x) for x in resnums]

    geometry_path = None
    #geometry_path = '/mnt/e/DesignData/ligands/LigandBB/_lig_fe/fe_geo.pdb'

    metal_metal_dist = 0.6

    num_contact_vdms = [3]

    #allowed_aa_combinations = [['H', 'H', 'D'], ['H', 'H', 'E'], ['H', 'H', 'H']] 
    allowed_aa_combinations = [['H', 'H', 'H']]



def run(target_path, query_dir, outdir, win_filter, para, path_to_database):

    with open(query_dir + 'all_metal_vdm.pkl', 'rb') as f:
        query_all_metal = pickle.load(f)

    with open(query_dir + 'AAMetalPhiPsi.pkl', 'rb') as f:
        all_querys = pickle.load(f)

    with open(query_dir + 'cluster_centroid_dict.pkl', 'rb') as f:
        cluster_centroid_dict = pickle.load(f)

    print(len(all_querys))

    ### run Search_struct

    geometry_path = para.geometry_path
    metal_metal_dist = para.metal_metal_dist
    num_contact_vdms = para.num_contact_vdms
    allowed_aa_combinations = para.allowed_aa_combinations


    _filter = filter.Search_filter(filter_abple = False, filter_phipsi = False, max_phipsi_val = 30, 
        filter_vdm_score = False, min_vdm_score = 0, filter_vdm_count = False, min_vdm_clu_num = 20,
        after_search_filter_geometry = True, filter_based_geometry_structure = True, angle_tol = 35, aa_aa_tol = 0.35, aa_metal_tol = 0.25,
        pair_angle_range = [92, 116], pair_aa_aa_dist_range = [3.0, 3.5], pair_metal_aa_dist_range = None,
        after_search_filter_qt_clash = True, vdm_vdm_clash_dist = 2.7, vdm_bb_clash_dist = 2.2, 
        write_filtered_result = False, selfcenter_filter_member_phipsi=True)

    ss =  search_selfcenter.Search_selfcenter(target_path,  outdir, all_querys, cluster_centroid_dict, query_all_metal, 
        num_contact_vdms, metal_metal_dist, win_filter, validateOriginStruct = False, 
        search_filter= _filter, geometry_path = geometry_path, density_radius = 0.6, 
        search_2ndshell = True, secondshell_vdms = path_to_database, rmsd_2ndshell = 0.75,
        allowed_aa_combinations = allowed_aa_combinations)

    search_selfcenter.run_search_selfcenter(ss)
    return 


def run_local():

    query_dir = '/mnt/e/DesignData/ligands_metal/ZN_rcsb_datesplit/20211013/20211013_selfcenter/pickle_noCYS/'
    #query_dir = '/mnt/e/DesignData/ligands/all/20220116_FE_MN_CO/20220116_selfcenter/pickle_noCYS/'

    workdir = '/mnt/e/DesignData/Metalloprotein/ParametricBundles2/param_ala/'
    
    path_to_database='/mnt/e/DesignData/Combs/Combs2_database/vdMs/'

    pdb_file = '00010.pdb'

    para = Para()

    target_path = workdir + pdb_file
    win_filter = para.win_filter
    outdir = workdir + 'output_' + pdb_file.split('.')[0] + '_/'
    
    run(target_path, query_dir, outdir, win_filter, para, path_to_database)


def run_wynton():
    #query_dir = '/wynton/home/degradolab/lonelu/GitHub_Design/Metalprot/data/database/pickle_all_fe_220119/'
    #query_dir = '/wynton/home/degradolab/lonelu/GitHub_Design/Metalprot/data/database/pickle_noCYS_mn_fe_co_220119/'
    query_dir = '/wynton/home/degradolab/lonelu/GitHub_Design/Metalprot/data/20211013_selfcenter/pickle_noCYS/'
    
    #workdir = '/wynton/home/degradolab/lonelu/GitHub_Design/Metalprot/data/ntf2_fe/family_3vsy/'
    workdir = '/wynton/home/degradolab/lonelu/DesignData/Metalloenzyme/HelixZn/'

    path_to_database='/wynton/home/degradolab/lonelu/DesignData/Database/vdMs/'

    para = Para()

    pdb_files = sorted([fp for fp in os.listdir(workdir) if fp[0] != '.' and '.pdb' in fp])

    ind = int(sys.argv[2]) -1
    if ind > len(pdb_files) -1:
        return
    target_path = workdir + pdb_files[ind]
    win_filter = para.win_filter
    outdir = workdir + 'output_' + pdb_files[ind].split('.')[0] + '_/'
    
    run(target_path, query_dir, outdir, win_filter, para, path_to_database)
    return


if __name__=='__main__':
    if sys.argv[1] == 'wynton': 
        run_wynton()
    elif sys.argv[1] == 'local':
        run_local()