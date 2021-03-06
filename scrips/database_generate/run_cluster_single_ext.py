import os
import sys
import prody as pr
#sys.path.append(r'/mnt/e/GitHub_Design/Metalprot')
from metalprot.database import database_extract as ldb
from metalprot.database import database_cluster as ldb_clu
from metalprot.database import database_vdMAtomOrder

'''
python /mnt/e/GitHub_Design/Metalprot/scrips/database_generate/run_cluster_single.py
'''

def extract_single_vdm(cores, outdir, AA, key, basic = True, extention = None, n = None, key_out = None, phipsi = False):
    for c in cores:   
        if basic:
            c.generate_AA_Metal(AA, key)
        elif extention:
            c.generate_AA_ext_Metal(AA, extention, key)          
        elif n:
            c.generate_AA_kMetal(AA, n, key, key_out) 
        elif phipsi:
            c.generate_AA_phipsi_Metal(AA, key)
        c.write_vdM(outdir, key = key)
    return

### set up parameters

workdir = "/mnt/e/DesignData/ligands/Zn_rcsb_datesplit/20211013/"

metal_sel = 'ion or name NI MN ZN CO CU MG FE' 
align_sel_backbone = 'name C CA N O NI MN ZN CO CU MG FE or ion'

cores = ldb.load_cores(workdir + '_Seq_core_date_reps/')


AA = 'GLU'

len_sel = 9

#Change len_sel according to aa sidechain number. His:6, Glu: 5, Asp: 4, Cys: 2
if AA == 'HIS':
    len_sel_sc = len_sel + 6 
elif AA == 'GLU':
    len_sel_sc = len_sel + 5
elif AA == 'ASP':
    len_sel_sc = len_sel + 4 
elif AA == 'CYS':
    len_sel_sc = len_sel + 2 


# AAExtMetal_HIS_ext1
extract_single_vdm(cores, workdir + 'AAextMetal_' + AA + '_reps/', AA = AA, key = 'AAextMetal_' + AA + '_ext3', basic = False, extention=3, n = None, key_out=None)

#_pdbs = ldb.get_all_pbd_prody(workdir + 'M3-1_AAextMetal_' + AA + '_reps/')

#ldb.run_cluster(_pdbs, workdir, 'M3-1_AAextMetal_' + AA + '_cluster02/', rmsd = 0.2, metal_sel = metal_sel, len_sel = 13, align_sel = align_sel_backbone, min_cluster_size = 2, tag = 'm3-1_')



