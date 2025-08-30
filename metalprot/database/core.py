
import itertools
import os
from numpy.core.fromnumeric import argmin
import prody as pr
import numpy as np
from ..basic import constant, prody_ext

# According to the prody atom flag (http://prody.csb.pitt.edu/manual/reference/atomic/flags.html#flags), NI MN ZN CO CU MG FE CA are not all flag as ion. 
# Note that calcium is read as CA, which is same as alpha carbon in prody selection. 
# So to select calcium, we need to add ion before it.

metal_sel = 'name NI MN ZN CO CU MG FE' 


def connectivity_filter(pdb_prody, ind, ext_ind):
    res1 = pdb_prody.select('protein and resindex ' + str(ind))
    res2 = pdb_prody.select('protein and resindex ' + str(ext_ind))
    if not res2:
        return False
    if res1[0].getResnum() - res2[0].getResnum() == ind - ext_ind and res1[0].getChid() == res2[0].getChid() and res1[0].getSegname() == res2[0].getSegname():
        return True
    return False

def extend_res_indices(inds_near_res, pdb_prody, extend = 4):
    extend_inds = []
    inds = set()
    for ind in inds_near_res:
        for i in range(-extend, extend + 1):
            the_ind = ind + i
            if the_ind not in inds and the_ind>= 0 and connectivity_filter(pdb_prody, ind, the_ind):
                extend_inds.append(the_ind)
                inds.add(the_ind)
    return extend_inds



def get_contact(pdbs):
    metal_coords = []
    contact_aas = []

    for pdb in pdbs:
        metal = pdb.select(metal_sel)[0]
        metal_coords.append(metal.getCoords())
        _contact_aas = pdb.select('protein and not carbon and not hydrogen and within 2.83 of resindex ' + str(metal.getResindex()))
        if len(_contact_aas) > 1:
            dists = [None]*len(_contact_aas)
            for i in range(len(_contact_aas)):
                dist = pr.calcDistance(metal, _contact_aas[i])
                dists[i] = dist
            contact_aa = _contact_aas[argmin(dists)]
        else:
            contact_aa = _contact_aas[0]
        contact_aas.append(contact_aa)

    names = [c.getName() for c in contact_aas]
    names.append(metal.getName())

    contact_aas_coords = []
    metal_coord = np.sum(metal_coords, 0)/(len(metal_coords))
    for c in contact_aas:
        contact_aas_coords.append(c.getCoords())
    contact_aas_coords.append(metal_coord)

    coords = np.array(contact_aas_coords)
    atom_contact_pdb = pr.AtomGroup('atom_contact')
    atom_contact_pdb.setCoords(coords)
    atom_contact_pdb.setNames(names)
        
    return atom_contact_pdb


class Core:
    def __init__(self, full_pdb):
        self.full_pdb = full_pdb

        self.metal = full_pdb.select(metal_sel)[0]
        self.metal_resind = self.metal.getResindex()

        self.contact_aas = self.full_pdb.select('protein and not carbon and not hydrogen and within 2.83 of resindex ' + str(self.metal.getResindex()))
        self.contact_aa_resinds = np.unique(self.contact_aas.getResindices())    

        # Inside the atomGroupDict. We will generate different type of pdb for clustering into vdM.
        self.atomGroupDict = dict()

    # EDITED BY 5.8
    def get_2ndshell_indices(inds, pdb_prody, ni_index, contact_aa_resinds =[], only_bb_2ndshell = False, _2nd_extend = 0):
        '''
        The method is to extract 2nd shell contact by applying heavy atom distance cut off 3.4.
        '''
        _2nd_resindices = []
        for ind in inds:
            if pdb_prody.select('resindex ' + str(ind)).getResnames()[0] == 'HIS':
                dist1 = pr.calcDistance(pdb_prody.select('index ' + str(ni_index))[0], pdb_prody.select('resindex ' + str(ind) + ' name ND1')[0])
                dist2 = pr.calcDistance(pdb_prody.select('index ' + str(ni_index))[0], pdb_prody.select('resindex ' + str(ind) + ' name NE2')[0])
                if dist1 < dist2:
                    index = pdb_prody.select('resindex ' + str(ind) + ' name NE2')[0].getIndex()
                    resindex = pdb_prody.select('resindex ' + str(ind) + ' name NE2')[0].getResindex()
                else:
                    index = pdb_prody.select('resindex ' + str(ind) + ' name ND1')[0].getIndex()
                    resindex = pdb_prody.select('resindex ' + str(ind) + ' name ND1')[0].getResindex()                         
            elif pdb_prody.select('resindex ' + str(ind)).getResnames()[0] == 'ASP':
                dist1 = pr.calcDistance(pdb_prody.select('index ' + str(ni_index))[0], pdb_prody.select('resindex ' + str(ind) + ' name OD1')[0])
                dist2 = pr.calcDistance(pdb_prody.select('index ' + str(ni_index))[0], pdb_prody.select('resindex ' + str(ind) + ' name OD2')[0])
                if dist1 < dist2:
                    index = pdb_prody.select('resindex ' + str(ind) + ' name OD2')[0].getIndex()
                    resindex = pdb_prody.select('resindex ' + str(ind) + ' name OD2')[0].getResindex()
                else:
                    index = pdb_prody.select('resindex ' + str(ind) + ' name OD1')[0].getIndex()
                    resindex = pdb_prody.select('resindex ' + str(ind) + ' name OD1')[0].getResindex()               
            elif pdb_prody.select('resindex ' + str(ind)).getResnames()[0] == 'GLU':
                dist1 = pr.calcDistance(pdb_prody.select('index ' + str(ni_index))[0], pdb_prody.select('resindex ' + str(ind) + ' name OE1')[0])
                dist2 = pr.calcDistance(pdb_prody.select('index ' + str(ni_index))[0], pdb_prody.select('resindex ' + str(ind) + ' name OE2')[0])
                if dist1 < dist2:
                    index = pdb_prody.select('resindex ' + str(ind) + ' name OE2')[0].getIndex()
                    resindex = pdb_prody.select('resindex ' + str(ind) + ' name OE2')[0].getResindex()
                else:
                    index = pdb_prody.select('resindex ' + str(ind) + ' name OE1')[0].getIndex()
                    resindex = pdb_prody.select('resindex ' + str(ind) + ' name OE1')[0].getResindex()
            else:
                continue     
            if only_bb_2ndshell:
                all_near = pdb_prody.select('protein and heavy and backbone and within 3.4 of index ' + str(index) + ' and not resindex ' + str(resindex))
            else:
                all_near = pdb_prody.select('protein and heavy and within 3.4 of index ' + str(index) + ' and not resindex ' + str(resindex))

            if not all_near or not all_near.select('nitrogen or oxygen or sulfur'):
                continue
            inds_2nshell = all_near.select('nitrogen or oxygen or sulfur').getResindices()
            inds_2nshell = [x for x in inds_2nshell if x not in contact_aa_resinds] #Remove from contact_aa_resinds
            if _2nd_extend > 0:
                #Here we extend the inds_2nshell, so later we can extract more bb infomation.
                inds_2nshell = extend_res_indices(inds_2nshell, pdb_prody, _2nd_extend)

            _2nd_resindices.extend(np.unique(inds_2nshell)) 

        return _2nd_resindices

    
    def add2atomGroupDict(self, key, sel_pdb_prody):
        if not key in self.atomGroupDict.keys():
            self.atomGroupDict[key] = []
        self.atomGroupDict[key].append(sel_pdb_prody)


    def write_vdM(self, outdir, key):
        if not key in self.atomGroupDict.keys():
            return

        os.makedirs(outdir, exist_ok=True)

        for ag in self.atomGroupDict[key]:
            #ag[1] here is supposed to be tag in the title.
            pr.writePDB(outdir + self.full_pdb.getTitle() + '_' + ag[1] + '.pdb', ag[0])
            
        
    def generate_AA_Metal(self, AA = 'HIS', key = 'AAMetal_HIS'):
        aa_sel = 'resname ' + AA

        for resind in self.contact_aa_resinds:
            if not self.full_pdb.select(aa_sel + ' and resindex ' + str(resind)):
                continue
            sel_pdb_prody = self.full_pdb.select('resindex ' + str(resind) + ' '+ str(self.metal_resind))
            tag = AA + '_' + self.full_pdb.select('resindex ' + str(resind)).getChids()[0] + '_' + str(self.full_pdb.select('resindex ' + str(resind)).getResnums()[0]) 
            
            self.add2atomGroupDict(key, (sel_pdb_prody, tag))            
        return 
        
    
    def generate_AA_ext_Metal(self, AA = 'HIS', extention = 3, key = 'AAExtMetal_HIS_ext3'):
        aa_sel = 'resname ' + AA

        for resind in self.contact_aa_resinds:
            if not self.full_pdb.select(aa_sel + ' and resindex ' + str(resind)):
                continue

            ext_inds = extend_res_indices([resind], self.full_pdb, extend =extention)
            if len(ext_inds) != 2*extention + 1:
                continue
            #print(self.full_pdb.getTitle() + '+' + '-'.join([str(x) for x in ext_inds]))
            sel_pdb_prody = self.full_pdb.select('resindex ' + ' '.join([str(ind) for ind in ext_inds]) + ' '+ str(self.metal_resind))
            tag =  AA + '_' + self.full_pdb.select('resindex ' + str(resind)).getChids()[0] + '_' + str(self.full_pdb.select('resindex ' + str(resind)).getResnums()[0]) 

            self.add2atomGroupDict(key, (sel_pdb_prody, tag))
        return


    def _generate_AA_phipsi_Metal(self, resind):
        ext_inds = extend_res_indices([resind], self.full_pdb, extend =1)
        if len(ext_inds) != 3:
            return
        #print(self.full_pdb.getTitle() + '+' + '-'.join([str(x) for x in ext_inds]))
        atom_inds = []
        atom_inds.extend(self.full_pdb.select('resindex ' + str(resind-1)).select('name C O').getIndices())
        atom_inds.extend(self.full_pdb.select('resindex ' + str(resind)).getIndices())
        atom_inds.extend(self.full_pdb.select('resindex ' + str(resind+1)).select('name N CA').getIndices())
        sel_pdb_prody = self.full_pdb.select('index ' + ' '.join([str(x) for x in atom_inds]) + ' '+ str(self.metal.getIndex()))
        return sel_pdb_prody

    
    def generate_AA_phipsi_Metal(self, AA = 'HIS', key = 'AAMetalPhiPsi_HIS'):
        aa_sel = 'resname ' + AA

        for resind in self.contact_aa_resinds:
            if not self.full_pdb.select(aa_sel + ' and resindex ' + str(resind)):
                continue
            
            sel_pdb_prody = self._generate_AA_phipsi_Metal(resind)
            if not sel_pdb_prody:
                continue
            tag = AA + '_' + self.full_pdb.select('resindex ' + str(resind)).getChids()[0] + '_' + str(self.full_pdb.select('resindex ' + str(resind)).getResnums()[0]) 
            self.add2atomGroupDict(key, (sel_pdb_prody, tag))
        return


    def generate_AAcAA_Metal(self, filter_aas = False, aas = ['HIS', 'HIS'], extention = 3, extention_out = 0, key = 'AAcAA_Metal_ext0'):
        exts = []
        for resind in self.contact_aa_resinds:
            ext_inds = extend_res_indices([resind], self.full_pdb, extend =extention)
            exts.append(ext_inds)
        
        pairs = []
        pairs_ind = []
        for i in range(len(self.contact_aa_resinds) - 1):
            for j in range(i+1, len(self.contact_aa_resinds)):
                overlap = list(set(exts[i]) & set(exts[j]))
                if len(overlap) ==0: continue

                if filter_aas:
                    filtered = True
                    if aas and len(aas)==2:
                        if self.full_pdb.select('resindex ' + str(self.contact_aa_resinds[i])).getResnames()[0] == aas[0] and self.full_pdb.select('resindex ' + str(self.contact_aa_resinds[j])).getResnames()[0] == aas[1]:
                            filtered= False
                        elif self.full_pdb.select('resindex ' + str(self.contact_aa_resinds[i])).getResnames()[0] == aas[1] and self.full_pdb.select('resindex ' + str(self.contact_aa_resinds[j])).getResnames()[0] == aas[0]:
                            filtered= False
                    if filtered: continue

                pairs.append((i, j))
                if self.contact_aa_resinds[i] < self.contact_aa_resinds[j]:
                    pairs_ind.append((self.contact_aa_resinds[i], self.contact_aa_resinds[j]))
                else:
                    pairs_ind.append((self.contact_aa_resinds[j], self.contact_aa_resinds[i]))
        
        for v in range(len(pairs)):
            i = pairs[v][0]
            j = pairs[v][1]
            ext_inds = list(set(exts[i]) | set(exts[j]))
            ext_inds = [x for x in ext_inds if x >= pairs_ind[v][0]-extention_out and x <= pairs_ind[v][1]+extention_out]
            sel_pdb_prody = self.full_pdb.select('resindex ' + ' '.join([str(ind) for ind in ext_inds]) + ' '+ str(self.metal_resind))
            
            self.add2atomGroupDict(key, (sel_pdb_prody, str(i) + '_' +str(j)))
        return


    def generate_AAdAA_Metal(self, filter_aas = False, aas = ['HIS', 'HIS'], key = 'AAdAA_Metal'):
        pairs_ind = []
        for i in range(len(self.contact_aa_resinds) - 1):
            for j in range(i+1, len(self.contact_aa_resinds)):

                if filter_aas:
                    filtered = True
                    if aas and len(aas)==2:
                        if self.full_pdb.select('resindex ' + str(self.contact_aa_resinds[i])).getResnames()[0] == aas[0] and self.full_pdb.select('resindex ' + str(self.contact_aa_resinds[j])).getResnames()[0] == aas[1]:
                            filtered= False
                        elif self.full_pdb.select('resindex ' + str(self.contact_aa_resinds[i])).getResnames()[0] == aas[1] and self.full_pdb.select('resindex ' + str(self.contact_aa_resinds[j])).getResnames()[0] == aas[0]:
                            filtered= False
                    if filtered: continue

                pairs_ind.append((self.contact_aa_resinds[i], self.contact_aa_resinds[j]))
                #pairs_ind.append((self.contact_aa_resinds[j], self.contact_aa_resinds[i]))  # In the database, we don't know the order of the two aa.
        
        for p in pairs_ind:
            sel_pdb_prody = self.full_pdb.select('resindex ' + str(p[0]) + ' ' +  str(p[1]) + ' '+ str(self.metal_resind))
            
            self.add2atomGroupDict(key, (sel_pdb_prody, ''))
        return


    def generate_AAdAAdAA_Metal(self, key = 'AAdAAdAA_Metal'):
        _ags = []
        for i in self.contact_aa_resinds:
            #if self.full_pdb.select('resindex ' + str(i)).getResnames()[0] not in ['HIS', 'GLU', 'ASP', 'CYS']:
            #    continue
            _ags.append(self.full_pdb.copy().select('resindex ' + str(i)).toAtomGroup())

        for ps in itertools.combinations(range(len(_ags)), 3):
            title = ''.join([constant.one_letter_code[_ags[p].getResnames()[0]] + str(p) for p in ps])
            sel_ags = [_ags[p] for p in ps]
            sel_ags.append(self.metal.copy())

            sel_pdb_prody = prody_ext.combine_ags(sel_ags, title, ABCchids=['A', 'B', 'C', 'M'])
            self.add2atomGroupDict(key, (sel_pdb_prody, title))
        return


    def get_inds_from_resind(self, pdb_prody, resind, aa = 'resname HIS'):
        if aa == 'resname HIS':
            inds = pdb_prody.select('name ND1 NE2 and resindex ' + str(resind)).getIndices()
        elif aa == 'resname ASP':
            inds = pdb_prody.select('name OD1 OD2 and resindex ' + str(resind)).getIndices()
        elif aa == 'resname GLU':
            inds = pdb_prody.select('name OE1 OE2 and resindex ' + str(resind)).getIndices()
        else:
            inds = []
        return inds


    def generate_AA_2ndShell_Metal(self, key = 'AA2ndShellMetal', filter_AA = False, AA = 'HIS', only_bb_2ndshell = False):
        for resind in self.contact_aa_resinds:
            if filter_AA and not self.full_pdb.select('resname ' + AA + ' and resindex ' + str(resind)):
                continue      
            #inds = get_inds_from_resind(pdb_prody, resind, aa) edited by 5.8
            _2nshell_resinds = Core.get_2ndshell_indices([resind],  self.full_pdb, self.metal.getIndex(), self.contact_aa_resinds, only_bb_2ndshell = only_bb_2ndshell)
            if len(_2nshell_resinds) > 0:
                for _2resind in _2nshell_resinds:      
                    #print(self.full_pdb.getTitle() + '+' + '-'.join([str(x) for x in _2nshell_resinds]))
                    sel_pdb_prody = self.full_pdb.select('resindex ' + str(resind) + ' ' + str(_2resind) + ' ' + str(self.metal_resind))
                  
                    self.add2atomGroupDict(key, (sel_pdb_prody, '_aa' + str(resind) + '_aa' + str(_2resind) + '_'))
        return


    def generate_AA_2ndShell_connect_Metal(self, key = 'AA2ndShellConnectMetal', filter_AA = False, AA = 'HIS', only_bb_2ndshell = False, extend = 4 ,_2nd_extend = 4, generate_sse = False):
        '''
        The purpose of the function is to find 2nd shell that are in the same secondary structure with the 1st shell.
        At the same time, the function could be used to find bb only 2ndshell vdMs. 
        '''
        for resind in self.contact_aa_resinds:
            if filter_AA and not self.full_pdb.select('resname ' + AA + ' and resindex ' + str(resind)):
                continue      
            _resinds = extend_res_indices([resind], self.full_pdb, extend)
            _resnums = self.full_pdb.select('resindex ' + ' '.join([str(r) for r in _resinds])).getResnums()
            #inds = get_inds_from_resind(pdb_prody, resind, aa)
            _2nshell_resinds = get_2ndshell_indices([resind],  self.full_pdb, self.metal.getIndex(), self.contact_aa_resinds, only_bb_2ndshell = only_bb_2ndshell)
            if len(_2nshell_resinds) > 0:
                for _2resind in _2nshell_resinds:      
                    #print(self.full_pdb.getTitle() + '+' + '-'.join([str(x) for x in _2nshell_resinds]))
                    inds_2ndshell = extend_res_indices([_2resind], self.full_pdb, _2nd_extend)
                    #Note that the resnums could come from different chain or segments. The chances are low though.
                    _resnums_2ndshell = self.full_pdb.select('resindex ' + ' '.join([str(r) for r in inds_2ndshell])).getResnums()

                    if len(set(_resinds).intersection(inds_2ndshell)) <= 0 or len(set(_resnums).intersection(_resnums_2ndshell)) <= 0:
                        continue
                    if generate_sse:
                        #Here try to generate the secondary structure by get [resind:_2resind].
                        #all_inds = list(set(_resinds) | set(inds_2ndshell))
                        if resind < _2resind:
                            all_inds = [x for x in range(resind, _2resind + 1, 1)]
                        else:
                            all_inds = [x for x in range(_2resind, resind + 1, 1)]
                        sel_pdb_prody = self.full_pdb.select('resindex ' + ' '.join([str(x) for x in all_inds]) +  ' ' + str(self.metal_resind))                       
                    else:
                        sel_pdb_prody = self.full_pdb.select('resindex ' + str(resind) + ' ' + str(_2resind) + ' ' + str(self.metal_resind))
                    self.add2atomGroupDict(key, (sel_pdb_prody, '_aa' + str(resind) + '_aa' + str(_2resind) + '_'))
        return


    def generate_AA_kMetal(self, AA = 'HIS', k = 5, key = 'AAMetal_HIS', key_out = 'AA5Metal_HIS'):        
        if not key in self.atomGroupDict:
            return
        for ag in self.atomGroupDict[key]:
            sel_pdb_prody = ag.select('protein').toAtomGroup() 
            for i in range(k):
                sel_pdb_prody = sel_pdb_prody + self.metal.toAtomGroup()

            if not key_out in self.atomGroupDict.keys():
                self.atomGroupDict[key_out] = []
            self.atomGroupDict[key_out].append(sel_pdb_prody)
        return


    def generate_binary_contact(self, key = 'BinaryAtomContact'):
        '''
        # For geometry purpose.
        binary contact is defined as: atom-Metal-atom. 
        '''
        atom_inds = np.unique(self.contact_aas.getIndices())
        for xs in itertools.combinations(atom_inds, 3):
            sel_pdb_prody = self.full_pdb.select('index ' + ' '.join([str(x) for x in xs])+ ' ' + str(self.metal.getIndex()))

            self.add2atomGroupDict(key, (sel_pdb_prody, '_'.join([str(x) for x in xs])))            
        return 


    def generate_atom_contact(self, key = 'AtomContact'):
        '''
        # For geometry purpose.
        '''
        atom_inds = np.unique(self.contact_aas.getIndices())

        sel_pdb_prody = self.full_pdb.select('index ' + ' '.join([str(x) for x in atom_inds])+ ' ' + str(self.metal.getIndex()))
        _key = key + str(len(sel_pdb_prody))
        self.add2atomGroupDict(_key, (sel_pdb_prody, ''))            
        return 