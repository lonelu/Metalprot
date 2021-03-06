'''
The class here is for searching metal-ligand.
First search the metal, 
then generate potential ligand positions.
the search the chemical groups of the ligands via Combs. 
'''

import os
import numpy as np
import prody as pr
from . import position_ligand
from ..basic import filter

class Search_ligand:
    '''
    
    '''
    def __init__(self, workdir, all_ala_path, all_gly_path, ideal_geo_path, ideal_geo_o, common_title = None):
        self.workdir = workdir
        self.all_ala = pr.parsePDB(self.workdir + all_ala_path)
        self.all_gly = pr.parsePDB(self.workdir + all_gly_path)
        self.ideal_geo = pr.parsePDB(self.workdir + ideal_geo_path)
        self.ideal_geo_o = ideal_geo_o
        self.common_title = common_title
        self.outdir = self.workdir + self.common_title + '/'

        self.min_geo_struct = None
        self.all_ligands = None
        self.filtered_ligands = None


    def generate_ligands(self, all_ligs, target, lig_connects, geo_sels = ['OE2 OK1 FE', 'OE2 OK2 FE', 'OK1 OE2 FE', 'OK1 OK2 FE', 'OK2 OE2 FE', 'OK2 OK1 FE'], clash_dist = 2.5):
        '''
        Generate all potential artificial ligand positions.
        '''
        self.all_ligands = []
        
        min_geo_struct, min_rmsd = filter.Search_filter.get_min_geo(self.ideal_geo, self.ideal_geo_o) 
        self.min_geo_struct = min_geo_struct

        for i in range(len(geo_sels)):
            geo_sel = geo_sels[i]
            _ligs = [l.copy() for l in all_ligs]
            [l.setTitle('Geo_' + str(i) + '_' + l.getTitle() ) for l in _ligs]
            position_ligand.lig_2_ideageo(_ligs, lig_connects, min_geo_struct, geo_sel = geo_sel)
            self.all_ligands.extend(_ligs)

        filtered_ligs = position_ligand.ligand_clashing_filter(self.all_ligands, target, dist = clash_dist)

        if len(filtered_ligs) <= 0:
            print('The position could not support the ligand.')
            return 

        self.filtered_ligands = filtered_ligs
        return

    

def prepare_search_ligand(workdir, ideal_geo_o_path):
    '''
    The inputs are from Search metal results.
    '''

    pdb_set = {}
    for f in os.listdir(workdir):
        if '.pdb' in f and 'allala' in f:
            pdb_set[f.split('allala')[0]] = {}

    for f in os.listdir(workdir):
        if '.pdb' in f and 'idealgeo' in f:
            pdb_set[f.split('idealgeo')[0]]['idealgeo'] = f

        if '.pdb' in f and 'allala' in f:
            pdb_set[f.split('allala')[0]]['allala'] = f

        if '.pdb' in f and 'allgly' in f:
            pdb_set[f.split('allgly')[0]]['allgly'] = f        
        

    ideal_geo_o = pr.parsePDB(ideal_geo_o_path)

    search_ligands = []
    for k in pdb_set.keys():
        sl = Search_ligand(workdir, pdb_set[k]['allala'], pdb_set[k]['allgly'], pdb_set[k]['idealgeo'], ideal_geo_o, k)
        search_ligands.append(sl)

    return search_ligands

    