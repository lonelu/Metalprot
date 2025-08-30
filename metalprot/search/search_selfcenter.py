'''
The database used for the search should be the selfcentered database.

The pairwise neighbor method will be removed in the future.

'''
import os
import itertools
import ssl
import prody as pr
import math
import numpy as np
from scipy.fftpack import ss_diff

from ..basic import utils
from ..basic import prody_ext

import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

from .search import Search_vdM, supperimpose_target_bb, calc_pairwise_neighbor
from .graph import Graph
from .comb_info import CombInfo
from .find_path_by_matrix_fixed import neighbor_generate_nngraph, calc_adj_matrix_paths
from ..basic.filter import Search_filter
from . import search_2ndshell

class Search_selfcenter(Search_vdM):
    '''
    Inheritated from Search_vdM. 
    Except here is searching the selfcenter vdM database. 
    '''

    def selfcenter_calc_density(self, comb_dict, radius):
        '''
        For each path, calc the density. 
        The density here is defined: 1. volume, a ball with the center of centroid metal and a radius 0.5. 
                                     2. Number, all members from each centroid vdM in this ball.
        This method is similar to overlap but is less computational heavy.
        '''
        print('selfcenter-calc-density')
        
        for key in comb_dict.keys():
            
            if not self.search_filter.write_filtered_result and comb_dict[key].after_search_filtered:
                continue

            win_comb = key[0]

            ### get all metal coords
            coords = []
            w_ind_s = []
 
            for w in win_comb:               
                member_metal_coords = comb_dict[key].centroid_dict[w].get_metal_mem_coords()
                coords.extend(member_metal_coords)
                member_metal_ids = comb_dict[key].centroid_dict[w].clu_member_ids
                for ind in range(len(member_metal_ids)):                    
                    w_ind_s.append((w, ind))

            ### Get the center point. 
            '''
            Here the center can be:
            1. The geometry center, which in some cases the geometry center is not the metal center of the vdM's all member.
            2. The all coords center, which bias toward the large clusters.
            3. The center of vdM members center.
            '''
            metal_coords = []  
            for _query in comb_dict[key].centroid_dict.values():                                
                metal_coords.append(_query.get_metal_coord())   
            center = [pr.calcCenter(prody_ext.transfer2pdb(metal_coords))]
            # center = [pr.calcCenter(prody_ext.transfer2pdb(coords))]


            ### get the overlap. 
            coord_in_center, coord_has_center = calc_pairwise_neighbor(coords, center, radius)

            overlap_ind_dict = {} 
            overlap_query_id_dict = {} # {win:[overlap ids]} need to get {win:[query ids]} 

            for i in range(len(w_ind_s)):
                w, ind = w_ind_s[i]

                if len(coord_in_center[i]) <= 0:
                    continue

                value = comb_dict[key].centroid_dict[w].clu_member_ids[ind]
                if self.search_filter.selfcenter_filter_member_phipsi:
                    phix, psix = self.phipsi[w]
                    if (not utils.filter_phipsi(phix, self.vdms[value].phi, self.search_filter.max_phipsi_val)) or (not utils.filter_phipsi(psix, self.vdms[value].psi, self.search_filter.max_phipsi_val)):
                        continue
                
                if w in overlap_query_id_dict.keys():
                    overlap_ind_dict[w].add(ind)
                    overlap_query_id_dict[w].add(value)
                else:
                    overlap_ind_dict[w] = set()
                    overlap_ind_dict[w].add(ind)
                    overlap_query_id_dict[w] = set()
                    overlap_query_id_dict[w].add(value)

            if len(overlap_ind_dict.keys()) != len(win_comb):
                continue
            
            comb_dict[key].overlap_ind_dict = overlap_ind_dict
            comb_dict[key].overlap_query_id_dict = overlap_query_id_dict 


        if self.eval_density:         
            #write into log
            x = str(key) + '\t'
            x += str(radius) + '\t'
            volume = 4.0/3.0 * math.pi * (radius*radius*radius)

            overlaps = [len(comb_dict[key].overlap_ind_dict[w]) for w in win_comb]     
            total = sum(overlaps)
            
            x += str(total) + '\t'
            x += str(round(volume, 2)) + '\t'
            x += str(round(total/volume, 2)) + '\t'
            x += '\t'.join([str(len(comb_dict[key].overlap_ind_dict[w])) for w in win_comb]) + '\t'

            clus = [comb_dict[key].centroid_dict[w].clu_num for w in win_comb]
            clu_total = sum(clus)
            x += str(clu_total) + '\t'
            x += '\t'.join([str(comb_dict[key].centroid_dict[w].clu_num) for w in win_comb]) + '\t'
            
            aas = [a[0] for a in key[1]]
            
            f_total = np.prod(overlaps)/np.prod([self.aa_vdm_info_dict[a][0] for a in aas])*clu_total/volume
            x += str(round(f_total * 10000, 2))  + '\t'
            f_max = np.prod(overlaps)/np.prod([self.aa_vdm_info_dict[a][1] for a in aas])*clu_total/volume
            x += str(round(f_max * 100, 2))  + '\t'
            f_avg = np.prod(overlaps)/np.prod([self.aa_vdm_info_dict[a][2] for a in aas])*clu_total/volume
            x += str(round(f_avg, 2))  + '\t'
            f_median = np.prod(overlaps)/np.prod([self.aa_vdm_info_dict[a][3] for a in aas])*clu_total/volume
            x += str(round(f_median, 2))  + '\t'

            x += '\n'
            self.log += x
   
        return      


    def selfcenter_analysis_comb(self, win_comb, comb_dict):
        '''
        For each win_comb, extract_vdMs, calc geometrys, and filter by genometry.
        '''
        print('selfcenter-run at: ' + ','.join([str(w) for w in win_comb]))
        if len([comb_dict.keys()]) <= 0:
            return comb_dict
            
        _target = self.target.copy()
        
        self.neighbor_extract_query(_target, comb_dict)

        self.neighbor_calc_geometry(comb_dict)
        self.neighbor_aftersearch_filt(_target, comb_dict) 

        self.selfcenter_calc_density(comb_dict, self.density_radius)
        self.neighbor_calc_comb_score(comb_dict)
            
        return comb_dict

    
    def selfcenter_write(self, win_comb, comb_dict):
        if len([comb_dict.keys()]) <= 0:
            return

        self.selfcenter_get_write_represents(comb_dict)
    
        self.selfcenter_write_win(self.best_aa_comb_dict)

        outpath = 'win_' + '-'.join([self.target_ind2chidres[w][0] + '-' + str(self.target_ind2chidres[w][1]) for w in win_comb]) + '/'
        outdir = self.workdir + outpath  
        
        if not self.search_filter.write_filtered_result:
            if len([key for key in comb_dict.keys() if not comb_dict[key].after_search_filtered]) > 0:
                self.neighbor_write_summary(outdir, self.best_aa_comb_dict, name = '_summary_' + '_'.join([self.target_ind2chidres[w][0] + '-' + str(self.target_ind2chidres[w][1]) for w in win_comb]) + '_' + self.target.getTitle() + '.tsv')
        else:
            if len(comb_dict)>0:
                self.neighbor_write_summary(outdir, self.best_aa_comb_dict, name = '_summary_' + '_'.join([self.target_ind2chidres[w][0] + '-' + str(self.target_ind2chidres[w][1]) for w in win_comb]) + '_' + self.target.getTitle() + '.tsv')
        return

    
    def run_search_selfcenter_depre20220622(self):
        '''
        ss: Search_selfcenter.
        First, Find paths with the matrix method in 'find_path_by_matrix'.

        Then calc the geometry and clashing.
        
        For any pass the filters, calc the density etc.
        '''
        self.neighbor_generate_query_dict()
        m_adj_matrix, win_labels, vdm_inds = neighbor_generate_nngraph(self)

        paths = []
        for _num_contact in self.num_contact_vdms:
            _paths = calc_adj_matrix_paths(m_adj_matrix, _num_contact)
   
            if not self.validateOriginStruct and len(self.allowed_aa_combinations_sorted) > 0:
                for _path in _paths:
                    aas = tuple(sorted([self.vdms[vdm_inds[p]].aa_type for p in _path]))
                    if aas in self.allowed_aa_combinations_sorted:
                        paths.append(_path)
            else:
                paths.extend(_paths)
        print('Find {} possible solutions before aftersearch filter'.format(len(paths)))

        if len(paths) <= 0:
            self.neighbor_write_log()
            return


        # TO DO: The geometry is not working for geo other than tetrahydral.
        win_comb_dict = {}
        for path in paths:
            win_comb = tuple([win_labels[p] for p in path])
            clu_key = tuple([self.vdms[vdm_inds[p]].get_cluster_key() for p in path])
            comb = dict()
            for i in range(len(win_comb)):
                comb[win_comb[i]] = [vdm_inds[path[i]]]

            combinfo = CombInfo()
            combinfo.comb = comb 
            if win_comb not in win_comb_dict.keys():
                win_comb_dict[win_comb] = {}
            win_comb_dict[win_comb][(win_comb, clu_key)] = combinfo

        for win_comb in win_comb_dict.keys():
            comb_dict = win_comb_dict[win_comb]
            _comb_dict = self.selfcenter_analysis_comb(win_comb, comb_dict)
            #self.neighbor_comb_dict.update(_comb_dict)
            self.neighbor_comb_dict[win_comb] = _comb_dict
        #self.neighbor_write_summary(self.workdir, self.best_aa_comb_dict, name = '_summary_' + self.target.getTitle() + '_' + self.time_tag + '.tsv')

        #self.neighbor_write_log()

        return

#>>> Writing Functions.

    def selfcenter_write_info(self, key, info, path_tag = '', end_tag = ''):
        '''
        Write output.
        Too many output, May need optimization. 
        '''
        if not self.search_filter.write_filtered_result and info.after_search_filtered:
            return
            
        outpath = path_tag + 'win_' + '-'.join([self.target_ind2chidres[k][0] + '-' + str(self.target_ind2chidres[k][1]) for k in key[0]]) + '/'
        outdir = self.workdir + outpath
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        tag = 'W_' + '-'.join([self.target_ind2chidres[k][0] + '-' + str(self.target_ind2chidres[k][1]) for k in key[0]]) + '_' + '-'.join([k[0] for k in key[1]]) + '_' + '-'.join([str(k[1]) for k in key[1]])
        # Write geometry       
        pr.writePDB(outdir + tag +'_geo.pdb', info.geometry) 

        #Write Centroid and all metal coords in the cluster
        for w in key[0]:
            centroid = info.centroid_dict[w]
            pdb_path = outdir + tag + '_w_' + str(w)  + end_tag + '.pdb'
            pr.writePDB(pdb_path, centroid.query)

            if self.output_wincomb_overlap:
                clu_allmetal_coords = centroid.get_metal_mem_coords()
                prody_ext.write2pymol(clu_allmetal_coords, outdir, tag + '_w_' + self.target_ind2chidres[w][0] + '-' + str(self.target_ind2chidres[w][1]) +'_points.pdb')  
        
        #Write overlap
        if not info.overlap_ind_dict:
            return

        if self.output_wincomb_overlap:
            for w in key[0]:
                metal_coords = []

                candidate_ids = info.overlap_ind_dict[w]
                #print(len(candidate_ids))
                centroid = info.centroid_dict[w]
                clu_allmetal_coords = centroid.get_metal_mem_coords()
                for cid in candidate_ids:
                    #print(len(clu_allmetal_coords[cid]))
                    metal_coords.append(clu_allmetal_coords[cid])

                prody_ext.write2pymol(metal_coords, outdir, tag + '_w_' + str(w) +'_overlaps.pdb')  
                
        volume = 4.0/3.0 * math.pi * (self.density_radius**3)
        total = sum([len(info.overlap_ind_dict[w]) for w in key[0]])
        info.volume = volume
        info.volPerMetal = total/volume
        return 


    def selfcenter_write_win(self, comb_dict, path_tag = ''):
        '''
        Write output.
        Too many output, May need optimization. 
        '''
        print('selfcenter search neighbor_write')
        for key in comb_dict.keys(): 
            info = comb_dict[key]
            self.selfcenter_write_info(key, info, path_tag = path_tag, end_tag = '_' + info.tag)
        return   


    def selfcenter_get_write_represents(self, comb_dict):
        '''
        Here we write the 'best' comb in each win_comb with same aa_comb.
        '''
        print('selfcenter-write-represents.')
        win_clu_2_win_aas = {}

        for key in comb_dict.keys():  
            info = comb_dict[key]
            if not self.search_filter.write_filtered_result and info.after_search_filtered:
                continue

            #>>> Here is where is 2ndshell infomation is extracted.
            if self.search_2ndshell:
                comb_dict[key].calc_2ndshell()

            wins = key[0]
            aas = tuple([c[0] for c in key[1]])
            if (wins, aas) in win_clu_2_win_aas.keys():
                win_clu_keys = win_clu_2_win_aas[(wins, aas)]
                if comb_dict[win_clu_keys['BestOPscore']].overlapScore < comb_dict[key].overlapScore:
                    win_clu_2_win_aas[(wins, aas)]['BestOPscore'] = key
                if comb_dict[win_clu_keys['BestGeo']].geo_rmsd > comb_dict[key].geo_rmsd:
                    win_clu_2_win_aas[(wins, aas)]['BestGeo'] = key
                if comb_dict[win_clu_keys['BestCluscore']].cluScore < comb_dict[key].cluScore:
                    win_clu_2_win_aas[(wins, aas)]['BestCluscore'] = key
                if self.search_2ndshell and comb_dict[win_clu_keys['Best2ShScore']].secondshell_score < comb_dict[key].secondshell_score:
                    win_clu_2_win_aas[(wins, aas)]['Best2ShScore'] = key
                if self.search_2ndshell and comb_dict[win_clu_keys['Best2ShRmsd']].secondshell_rmsd > comb_dict[key].secondshell_rmsd:
                    win_clu_2_win_aas[(wins, aas)]['Best2ShRmsd'] = key

            else:
                #Here, trying to find the best one with overlap score, rmsd and cluster score.
                win_clu_2_win_aas[(wins, aas)] = {}
                win_clu_2_win_aas[(wins, aas)]['BestOPscore'] = key
                win_clu_2_win_aas[(wins, aas)]['BestGeo'] = key
                win_clu_2_win_aas[(wins, aas)]['BestCluscore'] = key
                win_clu_2_win_aas[(wins, aas)]['Best2ShScore'] = key
                win_clu_2_win_aas[(wins, aas)]['Best2ShRmsd'] = key

        for key in win_clu_2_win_aas.keys():
            for k in win_clu_2_win_aas[key].keys():
                win_clu_key = win_clu_2_win_aas[key][k]
                # Here the 'self.best_aa_comb_dict' is used to write summary.tsv file.
                if win_clu_key in self.best_aa_comb_dict.keys():
                    self.best_aa_comb_dict[win_clu_key].tag += '_' + k
                else:
                    self.best_aa_comb_dict[win_clu_key] = comb_dict[win_clu_key]            
                    self.best_aa_comb_dict[win_clu_key].tag = k

        for key in self.best_aa_comb_dict.keys():
            if not ('BestOPscore' in self.best_aa_comb_dict[key].tag or 'Best2ShScore' in self.best_aa_comb_dict[key].tag or 'Best2ShRmsd' in self.best_aa_comb_dict[key].tag):
                continue

            outpath = 'win_' + '-'.join([self.target_ind2chidres[k][0] + '-' + str(self.target_ind2chidres[k][1]) for k in key[0]]) + '/'
            outdir = self.workdir + outpath
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            tag = 'W_' + '-'.join([self.target_ind2chidres[k][0] + '-' + str(self.target_ind2chidres[k][1]) for k in key[0]]) + '_' + '-'.join([k[0] for k in key[1]]) + '_' + '-'.join([str(k[1]) for k in key[1]])
            ag = prody_ext.combine_vdm_into_ag(self.best_aa_comb_dict[key].centroid_dict.values(), tag, self.best_aa_comb_dict[key].geometry, self.best_aa_comb_dict[key].overlapScore, self.best_aa_comb_dict[key].cluScore)
            pdb_path = self.outdir_represent + tag + '_' + self.best_aa_comb_dict[key].tag
            pr.writePDB(pdb_path + '.pdb', ag) 
            pdb_path = outdir + tag + '_' + self.best_aa_comb_dict[key].tag
            pr.writePDB(pdb_path + '.pdb', ag)               
            # # If the ideal geometry is not used as a filter before. 
            # if self.best_aa_comb_dict[key].geo_rmsd < 0: 
            #     ideal_geometry, rmsd = Search_filter.get_min_geo(self.best_aa_comb_dict[key].geometry, self.geo_struct)
            #     self.best_aa_comb_dict[key].geo_rmsd = rmsd
            #     self.best_aa_comb_dict[key].ideal_geo = ideal_geometry
            if self.best_aa_comb_dict[key].ideal_geo:
                pdb_path_idealgeo = self.outdir_represent + tag + '_idealgeo_' + str(round(self.best_aa_comb_dict[key].geo_rmsd, 2)) + '.pdb'
                pr.writePDB(pdb_path_idealgeo, self.best_aa_comb_dict[key].ideal_geo)   
                pdb_path_idealgeo = outdir + tag + '_idealgeo_' + str(round(self.best_aa_comb_dict[key].geo_rmsd, 2)) + '.pdb'
                pr.writePDB(pdb_path_idealgeo, self.best_aa_comb_dict[key].ideal_geo)  
           
            #>>> Write secondshell
            for lig_key in self.best_aa_comb_dict[key].secondshell_dict.keys():
                for ag, _infos in self.best_aa_comb_dict[key].secondshell_dict[lig_key]:
                    _chidres = self.target_ind2chidres[lig_key[0]] 
                    _2ShTag =  _chidres[0] + str(_chidres[1]) + '_' + '_'.join([_infos[0], _infos[1], _infos[2][0], str(_infos[2][1]), str(round(_infos[5], 2)), str(round(_infos[7], 2))])
                    pr.writePDB(outdir + tag + '_' + _2ShTag, ag)
        return


    def write_for_combs(self):
        '''
        The function here is to write structures for COMBS search.
        '''
        for key in self.best_aa_comb_dict.keys():
            if 'BestOPscore' not in self.best_aa_comb_dict[key].tag:
                continue
            tag = 'W_' + '-'.join([self.target_ind2chidres[k][0] + '-' + str(self.target_ind2chidres[k][1]) for k in key[0]]) + '_' + '-'.join([k[0] for k in key[1]]) + '_' + '-'.join([str(k[1]) for k in key[1]])

            pdb_path = self.workdir + 'represents_combs/' + tag 
            os.makedirs(self.workdir + 'represents_combs/', exist_ok = True)
            if self.best_aa_comb_dict[key].ideal_geo:
                pdb_path_idealgeo = pdb_path + '_idealgeo_' + str(round(self.best_aa_comb_dict[key].geo_rmsd, 2)) + '.pdb'
                pr.writePDB(pdb_path_idealgeo, self.best_aa_comb_dict[key].ideal_geo)                     
            
            ag_new = prody_ext.combine_vdm_target_into_ag(self.target, self.best_aa_comb_dict[key].centroid_dict, True, self.best_aa_comb_dict[key].geometry, tag, aa = '')
            pr.writePDB(pdb_path + '_all_vdms.pdb', ag_new)

            #ag_ala = prody_ext.target_to_all_gly_ala(self.target, pdb_path + 'all_ala.pdb', [], aa = 'ALA')
            #pr.writePDB(pdb_path + '_allala.pdb', ag_ala)

            ag_gly = prody_ext.combine_vdm_target_into_ag(self.target, self.best_aa_comb_dict[key].centroid_dict, True, self.best_aa_comb_dict[key].geometry, tag, aa = 'GLY')      
            #ag_gly = prody_ext.target_to_all_gly_ala(self.target, pdb_path + 'all_gly.pdb', [], aa = 'GLY')
            pr.writePDB(pdb_path + '_allgly.pdb', ag_gly)    
        return 


#>>> run search.

def run_search_selfcenter(ss):
    '''
    ss: Search_selfcenter.
    First, Find paths with the matrix method in 'find_path_by_matrix'.

    Then calc the geometry and clashing.
    
    For any pass the filters, calc the density etc.
    '''
    ss.neighbor_generate_query_dict()
    m_adj_matrix, win_labels, vdm_inds = neighbor_generate_nngraph(ss)

    paths = []

    for _num_contact in ss.num_contact_vdms:
        _paths = calc_adj_matrix_paths(m_adj_matrix, _num_contact)
        print('adj_matrix_paths: '.format(len(_paths)))
        if not ss.validateOriginStruct and len(ss.allowed_aa_combinations_sorted) > 0:
            for _path in _paths:
                aas = tuple(sorted([ss.vdms[vdm_inds[p]].aa_type for p in _path]))
                if aas in ss.allowed_aa_combinations_sorted:
                    paths.append(_path)
        else:
            paths.extend(_paths)
    print('Find {} possible solutions before aftersearch filter'.format(len(paths)))

    if len(paths) <= 0:
        ss.neighbor_write_log()
        return


    # TO DO: The geometry is not working for geo other than tetrahydral.
    win_comb_dict = {}
    for path in paths:
        win_comb = tuple([win_labels[p] for p in path])
        clu_key = tuple([ss.vdms[vdm_inds[p]].get_cluster_key() for p in path])
        comb = dict()
        for i in range(len(win_comb)):
            comb[win_comb[i]] = [vdm_inds[path[i]]]

        combinfo = CombInfo()
        combinfo.comb = comb 
        if win_comb not in win_comb_dict.keys():
            win_comb_dict[win_comb] = {}
        win_comb_dict[win_comb][(win_comb, clu_key)] = combinfo

    for win_comb in win_comb_dict.keys():
        comb_dict = win_comb_dict[win_comb]
        _comb_dict = ss.selfcenter_analysis_comb(win_comb, comb_dict)
        #ss.neighbor_comb_dict.update(_comb_dict)
        ss.neighbor_comb_dict[win_comb] = _comb_dict

    if ss.search_2ndshell:
        para = search_2ndshell.Para()
        para.rmsd = ss.rmsd_2ndshell
        search_2ndshell.run_search_2ndshell(ss.target, ss.neighbor_comb_dict, ss.target_ind2chidres, ss._resnum_filtered, ss.secondshell_vdms, para)


    for win_comb in ss.neighbor_comb_dict.keys():
        comb_dict = ss.neighbor_comb_dict[win_comb]
        ss.selfcenter_write(win_comb, comb_dict)
    
    ss.neighbor_write_summary(ss.workdir, ss.best_aa_comb_dict, name = '_summary_' + ss.target.getTitle() + '_' + ss.time_tag + '.tsv')

    ss.neighbor_write_log()

    return
