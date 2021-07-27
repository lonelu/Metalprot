import prody as pr
import itertools
import numpy as np
from prody.atomic import pointer
from .hull import transfer2pdb, write2pymol

metal_sel = 'ion or name NI MN ZN CO CU MG FE' 


class Query:
    def __init__(self, query, score = 0, clu_num = 0, clu_total_num = 0, is_bivalent = False, win = None, path = None, ag = None, hull_ag = None, cluster = None):
        self.query = query
        self.score = score
        self.clu_num = clu_num
        self.clu_total_num = clu_total_num

        #Get contact_resind
        cen_inds = np.unique(self.query.getResindices())
        self.contact_resind = cen_inds[int(cen_inds.shape[0]/2)-1]

        #Bivalent vdm usage.
        self.is_bivalent = is_bivalent

        #Extra properties for special usage.
        self.win = win
        self.path = path
        self.ag = ag
        self._2nd_shells = []
        if self.win is not None and '_win_' not in self.query.getTitle():
            self.query.setTitle(self.query.getTitle().split('.pdb')[0] + '_win_' + '_'.join([str(w) for w in self.win]) + '.pdb')

        #Extra properties for hull usage. 
        self.hull_ag = hull_ag
        self.cluster = cluster  #The cluster is used as a reference. Many Query will share the same cluster to save memory. Be careful about the change of alignment.
        self.candidates = None
        self.candidates_points = None

        #Extra properties for phipsi.
        self.phi = None
        self.psi = None

    def get_phi_psi(self):
        indices = self.query.select('name N C CA O').getIndices() 
        atoms = [self.query.select('index ' + str(i)) for i in indices]
        self.phi = pr.calcDihedral(atoms[0], atoms[1], atoms[2], atoms[3])
        self.psi = pr.calcDihedral(atoms[4], atoms[5], atoms[6], atoms[7])

        # for p in self.query.iterResidues():
        #     try:
        #         self.phi = pr.calcPhi(p)
        #     except:
        #         pass

        #     try:
        #         self.psi= pr.calcPsi(p)
        #     except:
        #         pass
        return 

    def realign_by_CCAN_candidates(self, cand_ids, align_sel = 'name N CA C'):
        '''
        realign certain members of the clusters to the target. 
        '''
        candidates = []
        for i in cand_ids:
            q = self.cluster.querys[i].copy()
            inds = np.unique(q.query.getResindices())
            ind = inds[int(inds.shape[0]/2)-1]

            transform = pr.calcTransformation(q.query.select(align_sel + ' and resindex ' + str(ind)), self.query.select(align_sel + ' and resindex ' + str(self.contact_resind)))
            transform.apply(q.query)
        self.candidates = candidates
        return

    def extract_mem_metal_point(self):
        '''
        Only use when load the data to generate hull_ag. As cluster is used as a reference and shared by other Query at different position of the protein bb.
        '''
        points = []
        for q in self.cluster.querys:
            points.append(q.query.select(metal_sel)[0].getCoords())
        self.hull_ag = transfer2pdb(points)
        return


    def get_hull_points(self):
        if not self.hull_ag:
            return None
        return self.hull_ag.getCoords()


    def write_points(self, outdir):
        points = self.get_hull_points()
        write2pymol(points, outdir, self.query.getTitle())


    def set_win(self, win):
        self.win = win
        if self.win is not None and '_win_' not in self.query.getTitle():
            self.query.setTitle(self.query.getTitle().split('.pdb')[0] + '_win_' + '_'.join([str(w) for w in self.win]) + '.pdb')
        

    def to_tab_string(self):
        query_info = self.query.getTitle() + '\t' + str(round(self.score, 2)) + '\t' + str(self.clu_num)  + '\t'+ str(self.clu_total_num)
        return query_info

    def copy(self):
        hull_ag = None
        if self.hull_ag:
            hull_ag = self.hull_ag.copy()
        
        return Query(self.query.copy(), self.score, self.clu_num, self.clu_total_num, self.is_bivalent, self.win, self.path, self.ag, hull_ag, self.cluster)
    
    def win_str(self):
        return '-'.join([str(w) for w in self.win])

    def write(self, outpath):
        pr.writePDB(outpath, self.query)


class Comb:
    def __init__(self, querys, min_contact_query = None, min_contact_rmsd = None):
        self.querys = querys       
        self.total_score = sum([q.score for q in querys])
        self.total_clu_number= sum([q.clu_num for q in querys])        
        self.scores = [q.score for q in querys]
        self.clu_nums= [q.clu_num for q in querys]
        self.min_contact_query = min_contact_query
        self.min_contact_rmsd = min_contact_rmsd

        self.pair_dists = None
        self.pair_angles = None

    def to_tab_string(self):
        query_names = '||'.join([q.query.getTitle() for q in self.querys])
        query_scores = '||'.join([str(round(n, 2)) for n in self.scores])
        query_clu_nums = '||'.join([str(s) for s in self.clu_nums])
        wins = '||'.join([q.win_str() for q in self.querys])
        vdm_info = str(round(self.total_score, 2)) + '\t' + str(self.total_clu_number) + '\t' + query_names + '\t' + query_scores + '\t' + query_clu_nums + '\t' + wins
        if self.min_contact_query:
            vdm_info += '\t' + self.min_contact_query.query.getTitle() + '\t' + str(round(self.min_contact_query.score, 2)) + '\t' + str(round(self.min_contact_rmsd, 2))
            
        if self.pair_dists:
            vdm_info += '\t' + '||'.join([str(round(d, 2)) for d in self.pair_dists]) + '\t' + '||'.join([str(round(a, 2)) for a in self.pair_angles])
        return vdm_info


    def calc_pair_geometry(self):
        '''
        Calc paired query angle and distance. 
        '''
        #Get metal, binding atom for each binding atom
        mts = []
        cts = []
        for q in self.querys:
            metal = q.query.select(metal_sel)[0]
            _contact_aas = q.query.select('protein and not carbon and not hydrogen and within 2.83 of resindex ' + str(metal.getResindex()))
            #For each aa, only select one contact atom. 
            resindices = _contact_aas.getResindices()
            for rid in resindices:
                #TO DO: Not the first one, but the closest one for the  such ASP contains two contact atoms.
                ct = _contact_aas.select('resindex ' + str(rid))[0]
                mts.append(metal)
                cts.append(ct)

        ct_len = len(cts)

        dist_pair = []
        angle_pair = []
        for i, j in itertools.combinations(range(ct_len), 2):   
            dist = pr.calcDistance(cts[i], cts[j])
            dist_pair.append(dist)
            angle = pr.calcAngle(cts[i], mts[i], cts[j])
            angle2 = pr.calcAngle(cts[i], mts[j], cts[j])
            angle_pair.append((angle + angle2)/2)
        self.pair_dists = dist_pair
        self.pair_angles = angle_pair   


class Cluster:
    def __init__(self, querys):      
        self.querys = querys
        self.total_score = sum([q.score for q in querys])
        self.total_clu_number= sum([q.clu_num for q in querys])        
        self.scores = [q.score for q in querys]
        self.clu_nums= [q.clu_num for q in querys]
        self.centroid = [q for q in querys if 'centroid' in q.query.getTitle()][0]

    def realign_by_CCAN(self, target, align_sel = 'name N CA C'):
        for q in self.querys:
            inds = np.unique(q.query.getResindices())
            ind = inds[int(inds.shape[0]/2)-1]
            transform = pr.calcTransformation(q.query.select(align_sel + ' and resindex ' + str(ind)), target.query.select(align_sel + ' and resindex ' + str(target.contact_resind)))
            transform.apply(q.query)
        return


    def write(self, outpath):
        for q in self.querys:
            q.write(outpath)
        return



