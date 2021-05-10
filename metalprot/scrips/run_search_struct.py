import os
import sys
import prody as pr

#You can either add the python package path.
sys.path.append(r'/mnt/e/GitHub_Design/Metalprot')
from metalprot import search_struct, extract_vdm

# Generate queryss
queryss = []

query_dir = '/mnt/e/DesignData/ligands/ZN_rcsb/'

print(query_dir)

#Get query pdbs 
querys = extract_vdm.extract_all_centroid(query_dir, summary_name = '_summary.txt', file_name_includes = ['cluster', '7_'], score_cut = 0, clu_num_cut = 10)

queryss.append(querys)

#Get query_2nd pdbs 
# TO DO: currently, cannot superimpose to cluster with phipsi angle, database starting with '5_'.

query_2nds = extract_vdm.extract_all_centroid(query_dir, summary_name = '_summary.txt', file_name_includes = ['cluster', '6_'], score_cut = 1, clu_num_cut = 50)

queryss.append(query_2nds)

print(len(queryss[0]))
print(len(queryss[1]))

# run Search_struct

workdir = '/mnt/e/DesignData/ligands/Design_Sam/'

outdir = workdir + 'output_test/'

target_path = workdir + '3into4_helix_assembly_renum.pdb'

rmsd_cuts = [0.5, 0.5]

dist_cuts = [0.75, 0.75]

num_iter = 2

clash_query_query = 2.3

clash_query_target = 2.8

use_sep_aas = [False, False]

tolerance = 0.5

ss = search_struct.Search_struct(target_path, outdir, queryss, rmsd_cuts, dist_cuts, num_iter, clash_query_query, clash_query_target, use_sep_aas, tolerance)

ss.run_search_struct()