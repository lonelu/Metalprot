import os
import sys
import prody as pr
#sys.path.append(r'/mnt/e/GitHub_Design/Metalprot')
from metalprot import ligand_database as ldb
import itertools

### set up parameters

workdir = "/mnt/e/DesignData/ligands/ZN_rcsb/20210608/"

metal_sel = 'ion or name NI MN ZN CO CU MG FE' 
align_sel_backbone = 'name C CA N O NI MN ZN CO CU MG FE or ion'

cores = ldb.load_cores(workdir + '_Seq_cores_reps/')

# Align 2 separate aa core

for c in cores:   
    c.generate_AAcAA_Metal(filter_aas = True, aas = ['HIS', 'HIS'], key = 'AAcAAMetal_HIS_HIS')
    c.write_vdM(workdir + 'M5_AAcAAMetal_HIS_HIS_reps/',  key = 'AAcAAMetal_HIS_HIS')

_pdbs = ldb.get_all_pbd_prody(workdir + 'M5_AAcAAMetal_HIS_HIS_reps/')
ldb.run_cluster(_pdbs, workdir, 'M5_AAcAAMetal_HIS_HIS_clusters05/', rmsd = 0.5, metal_sel = metal_sel, len_sel = 2*4 + 1, align_sel = align_sel_backbone, min_cluster_size = 2, tag = 'm5_bb_')
ldb.run_cluster(_pdbs, workdir, 'M5_AAcAAMetal_HIS_HIS_clusters02/', rmsd = 0.2, metal_sel = metal_sel, len_sel = 2*4 + 1, align_sel = align_sel_backbone, min_cluster_size = 2, tag = 'm5_bb_')


ldb.run_cluster(_pdbs, workdir, 'M5_AAcAAMetalSc_HIS_HIS_clusters05/', rmsd = 0.5, metal_sel = metal_sel, len_sel = 2*10 + 1, align_sel = 'heavy', min_cluster_size = 2, tag = 'm5_sc_')
ldb.run_cluster(_pdbs, workdir, 'M5_AAcAAMetalSc_HIS_HIS_clusters02/', rmsd = 0.2, metal_sel = metal_sel, len_sel = 2*10 + 1, align_sel = 'heavy', min_cluster_size = 2, tag = 'm5_sc_')



# Align 2 separate aa core

for c in cores:   
    c.generate_AAdAA_Metal(filter_aas = False, aas = ['HIS', 'HIS'], key = 'AAdAAMetal_aa_aa')
    c.write_vdM(workdir + 'AAdAAMetal_aa_aa_reps/',  key = 'AAdAAMetal_aa_aa')

_pdbs = ldb.get_all_pbd_prody(workdir + 'M5_AAdAAMetal_HIS_HIS_reps/')
ldb.run_cluster(_pdbs, workdir, 'M5_AAdAAMetal_HIS_HIS_clusters05/', rmsd = 0.5, metal_sel = metal_sel, len_sel = 2*4 + 1, align_sel = align_sel_backbone, min_cluster_size = 2, tag = 'm5_bb_')
ldb.run_cluster(_pdbs, workdir, 'M5_AAdAAMetal_HIS_HIS_clusters02/', rmsd = 0.2, metal_sel = metal_sel, len_sel = 2*4 + 1, align_sel = align_sel_backbone, min_cluster_size = 2, tag = 'm5_bb_')


ldb.run_cluster(_pdbs, workdir, 'M5_AAdAAMetalSc_HIS_HIS_clusters05/', rmsd = 0.5, metal_sel = metal_sel, len_sel = 2*10 + 1, align_sel = 'heavy', min_cluster_size = 2, tag = 'm5_sc_')
ldb.run_cluster(_pdbs, workdir, 'M5_AAdAAMetalSc_HIS_HIS_clusters02/', rmsd = 0.2, metal_sel = metal_sel, len_sel = 2*10 + 1, align_sel = 'heavy', min_cluster_size = 2, tag = 'm5_sc_')



