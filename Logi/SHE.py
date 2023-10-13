#@title Sequential Block Elimination
from itertools import product
import numpy as np
from numpy.random import Generator, PCG64
from collections import defaultdict
import itertools
from operator import itemgetter
import numpy.lib.recfunctions as rf
#from math import log
from pathlib import Path
import logging
from statistics import mean

class SHE(object):
    def __init__(self, n_rounds, bounds):
        self.n_rounds = n_rounds
        self.rg = Generator(PCG64(12345))
        self.cnt1 = 16
        self.cnt2 = 16

        ar1 = np.linspace(bounds[0,0], bounds[0,1], self.cnt1, dtype='float16')
        ar2 = np.linspace(bounds[1,0], bounds[1,1], self.cnt2, dtype='float16')
        self.ln1 = ar1.shape[0]
        self.ln2 = ar2.shape[0]
        no_actions = self.cnt1*self.cnt2

        self.full_actions = np.array([*product(ar1,ar2)]).reshape(self.cnt1*self.cnt2,2) # set of actions tuples
        self.incl_actions = np.ones(no_actions,dtype=bool)
        self.actions_mean = np.zeros(no_actions, dtype='float32')
        self.actions_var = np.zeros(no_actions, dtype='float32')
        self.actions_wards = np.zeros(no_actions, dtype='uint16')

        self.revenues = defaultdict(list)
        self.reset()

        log_file = Path('./logs/%s.log' %('she'))  #logging as SuccessiveBlockElimination (sbe)
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        self.log = logging.getLogger('she')
        logging.info("Running Successive Block Elimination algorithm")

        self.M = self.flt_inds_actv_actions.size # no of blocks
        self.n_r = [3, 3, 3, 3, 2, 2, 2, 2]
        self.n_r_ind = 0
        self.n_r_cnt = 0
        self.round_count = 0

    def reset(self):
        self.max = None # maximum value observed thus far
        self.argmax = None # argument at which maximum value was observed

    def act(self,t):
        pl_arm = None
        pl_arm_ind = None
        if (self.n_r_ind == len(self.n_r)):
            self.n_r_ind -= 1
        pl_cnt = self.n_r[self.n_r_ind]
        if (self.round_count == self.M):
            self.round_count = 0    #reset round count
            self.n_r_cnt += 1
        if (self.n_r_cnt == self.n_r[self.n_r_ind]):
            logging.info("running action elimination algorithm")
            self.eliminate_least_rewarding_half()   #this function should update active indices and M
            self.n_r_ind += 1
            self.M = self.flt_inds_actv_actions.size # no of blocks

        ravel_pl_arm_ind = self.flt_inds_actv_actions[self.round_count]
        pl_arm_ind = np.unravel_index(ravel_pl_arm_ind, (self.ln1,self.ln2))
        pl_arm = self.full_actions[pl_arm_ind]
        self.round_count+=1
        return list(pl_arm), (self.excl_actions[np.nonzero(self.excl_actions)]).tolist(), ravel_pl_arm_ind, pl_arm_ind

    def eliminate_least_rewarding_half(self):
        logging.info("==== Eliminating the worst half ====")
        if (self.flt_inds_actv_actions.size == 1):
            logging.info("only one arm remaining... returning")
            return 0
        median = np.median(self.means[self.flt_inds_actv_actions])
        logging.info("Median Revenue:{}".format(median))
        eliminated_inds = np.unravel_index(self.flt_inds_actv_actions[self.means[self.flt_inds_actv_actions]<= median],self.full_actions.shape)
        logging.info("Eliminated actions are: " +','.join('{}'.format(el) for el in zip(*eliminated_inds)))
        self.excl_actions[eliminated_inds] = self.full_actions[eliminated_inds]

        Tp = ~np.isin(self.full_actions, self.excl_actions)
        if Tp.sum() == 0:
            logging.info("All actions are eliminated. Check the code")
            exit(-1)
        logging.info("Number of active arms: {}".format(Tp.sum()))
        self.inds_actv_actions = np.nonzero(Tp)    #indices of active actions
        self.flt_inds_actv_actions = np.ravel_multi_index(self.inds_actv_actions, dims=(self.ln1,self.ln2))    # flattened active actions index


    def update(self, state, revenue, r_ind, ind):
        p1p2 = tuple(self.full_actions[ind])
        #arm_sem = 0
        self.revenues[p1p2].append(revenue)

        arm_mean = mean(self.revenues[p1p2])
        self.actions_mean[ind] = arm_mean

        self.means[r_ind] = arm_mean

        #if len(self.revenues[p1p2]) > 1:
        #    arm_sem = sem(self.revenues[p1p2])

        #self.actions_sem[ind] = arm_sem
        reward = 0

        logging.info("arm:{}, state:{}, revenues:{}, meanr:{}, liner ind:{}, ind:{}".format(p1p2, state, revenue, arm_mean, r_ind, ind))

        max_ind = np.argmax(self.actions_mean[self.inds_actv_actions])
        argmaxx = [self.inds_actv_actions[0][max_ind],self.inds_actv_actions[1][max_ind]]
        self.argmax = self.full_actions[self.inds_actv_actions[0][max_ind],self.inds_actv_actions[1][max_ind]].tolist()
        self.max = [self.actions_mean[tuple(argmaxx)]]
        logging.info("optimizer solution:{}, corresponding revenue:{}, index:{}, linear index:{}".format(self.argmax, self.max, argmaxx, max_ind))
        return reward
