#@title Recursive Block Elimination with variance dependent arm selection
from itertools import product
import numpy as np
from numpy.random import Generator, PCG64, beta, lognormal, choice
from collections import defaultdict
import itertools
from operator import itemgetter
import numpy.lib.recfunctions as rf
from math import log,sqrt
from pathlib import Path
import logging
from statistics import mean,fmean,stdev, variance
from scipy.stats import sem

class TS_BTTVE(object):
    def __init__(self, bounds):
        self.cnt1 = 16
        self.cnt2 = 16
        self.rg = Generator(PCG64(12345))
        self.lower_bounds = None
        self.upper_bounds = None
        self.round = 0
        self.excl_cnt = 2
        self.block_arr = []

        self.block_mean = np.array([])
        self.block_var = np.array([])

        self.block_pmean = np.array([])
        self.block_pvar = np.array([])

        self.block_switch = 1
        self.block_incl_inds = np.array([])
        self.arm_elm = 0

        ar1 = np.linspace(bounds[0,0], bounds[0,1], self.cnt1, dtype='float16')
        ar2 = np.linspace(bounds[1,0], bounds[1,1], self.cnt2, dtype='float16')
        self.ln1 = ar1.shape[0]
        self.ln2 = ar2.shape[0]
        no_actions = self.ln1*self.ln2

        self.gamma = 2
        self.prior_samp_sz = 5
        self.deg_fd = 1
        self.p_mean = 1
        self.p_var = 0.5

        self.full_actions = np.array([*product(ar1, ar2)], dtype=('float16,float16')).reshape(self.ln1,self.ln2) # set of actions tuples
        self.excl_actions = np.zeros_like(self.full_actions)
        self.actions_mean = np.zeros_like(self.full_actions, dtype='float32')
        self.actions_var = np.zeros_like(self.full_actions, dtype='float32')
        self.actions_sem = np.zeros_like(self.full_actions, dtype='float32')
        self.actions_wards = np.zeros_like(self.full_actions, dtype='uint16')

        self.actions_pmean = np.zeros_like(self.full_actions, dtype='float32')
        self.actions_pvar = np.zeros_like(self.full_actions, dtype='float32')

        self.pmeans = np.ones(no_actions, dtype='float32')*self.p_mean
        self.pvars = np.ones(no_actions, dtype='float32')*self.p_var

        self.means = np.zeros(no_actions, dtype='float16')
        self.vars = np.zeros(no_actions, dtype='float16')

        self.revenues = defaultdict(list)
        Tp = ~np.isin(self.full_actions, self.excl_actions)
        self.inds_actv_actions = np.nonzero(Tp)    #indices of active actions
        self.flt_inds_actv_actions = np.ravel_multi_index(self.inds_actv_actions, dims=(self.ln1,self.ln2))    # flattened active actions index

        self.reset()

        log_file = Path('./logs/%s.log' %('bttve'))  #logging as SuccessiveBlockElimination (sbe)
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        self.log = logging.getLogger('bttve')
        logging.info("Running Successive Block Elimination algorithm")
        #logging.info("Set of Full Arms in num_list are: {}".format(self.full_actions))
        '''self.samp_mean = 0
        self.samp_var = 0'''

    def reset(self):
        self.max = None # maximum value observed thus far
        self.argmax = None # argument at which maximum value was observed

    def act(self,t):
        self.round = t

        self.pmeans[:] = self.p_mean
        self.pvars[:] = self.p_var
        self.pmeans[np.ravel_multi_index(np.nonzero(self.block_pmean), dims = self.block_pmean.shape)] = self.block_pmean[np.nonzero(self.block_pmean)]
        self.pvars[np.ravel_multi_index(np.nonzero(self.block_pvar), dims = self.block_pvar.shape)] = self.block_pvar[np.nonzero(self.block_pvar)]
        self.pmeans[np.ravel_multi_index(np.nonzero(self.actions_pmean), dims = self.actions_pmean.shape)] = self.actions_pmean[np.nonzero(self.actions_pmean)]
        self.pvars[np.ravel_multi_index(np.nonzero(self.actions_pvar), dims = self.actions_pvar.shape)] = self.actions_pvar[np.nonzero(self.actions_pvar)]


        self.means[np.ravel_multi_index(np.where(self.actions_mean == 0), dims=(self.ln1, self.ln2))] = 0.0
        self.vars[np.ravel_multi_index(np.where(self.actions_var == 0), dims=(self.ln1, self.ln2))] = 0.0
        self.means[np.ravel_multi_index(np.nonzero(self.block_mean), dims = self.block_mean.shape)] = self.block_mean[np.nonzero(self.block_mean)]
        self.vars[np.ravel_multi_index(np.nonzero(self.block_mean), dims = self.block_mean.shape)] = self.block_var[np.nonzero(self.block_mean)]
        self.means[np.ravel_multi_index(np.nonzero(self.actions_mean), dims = self.actions_mean.shape)] = self.actions_mean[np.nonzero(self.actions_mean)]
        self.vars[np.ravel_multi_index(np.nonzero(self.actions_var), dims = self.actions_var.shape)] = self.actions_var[np.nonzero(self.actions_var)]

        p_var = self.rg.gamma(self.deg_fd/2, (self.deg_fd*self.pvars)/2)
        p_mean = self.rg.normal(self.pmeans, 1/(self.prior_samp_sz*self.pvars))

        samples = lognormal(p_mean[self.flt_inds_actv_actions], p_var[self.flt_inds_actv_actions])
        ravel_pl_arm_ind = self.flt_inds_actv_actions[np.random.choice(lognormal(self.means[self.flt_inds_actv_actions],self.vars[self.flt_inds_actv_actions]).argsort()[::-1][:2])]
        #ravel_pl_arm_ind = self.flt_inds_actv_actions[choice(np.flatnonzero(samples == np.max(samples)))]
        #ravel_pl_arm_ind = self.flt_inds_actv_actions[np.argmax(lognormal(self.means[self.flt_inds_actv_actions],self.vars[self.flt_inds_actv_actions]))]
        pl_arm_ind = np.unravel_index(ravel_pl_arm_ind, (self.ln1,self.ln2))
        pl_arm = self.full_actions[pl_arm_ind]
        return list(pl_arm), (self.excl_actions[np.nonzero(self.excl_actions)]).tolist(), ravel_pl_arm_ind, pl_arm_ind

    def eliminate_arms(self):
        logging.info("==== Runnung elimination algorithms ====")
        def update_bounds_get_indcs():    #update the bounds of the blocks
            if (self.block_incl_inds.size == 0) and (self.block_switch == 1):
                self.block_incl_inds = np.arange(self.block_arr.shape[0])
                self.block_excl_inds = np.array([],dtype=np.uint16)
                self.block_switch = 1
            '''logging.info("Blocking arms for revenue calculation")
            logging.info("Number of blocks to update the bounds:{}".format(self.block_incl_inds.shape[0]))
            logging.info("Indices for blocking are: {}".format(self.block_incl_inds))
            logging.info("===== Included blocks =====")
            for j in self.block_incl_inds:
                logging.info("{} st block is: {}".format(j,self.block_arr[j]))'''
            block_revenues = []
            for grp in rf.structured_to_unstructured(self.block_arr[self.block_incl_inds]):
                vals = list(x for key in grp.reshape(-1, grp.shape[-1]) for x in self.revenues[tuple(key)])
                if not vals:
                    vals = [0]
                if len(vals) >= 4:
                  #block_revenues.append([min(vals),max(vals),mean(vals),sem(vals),vals])
                  block_revenues.append([min(vals),max(vals),fmean(vals),fmean(vals),sem(vals),variance(vals),vals])
                else:
                  if len(vals) < 2:
                    block_revenues.append([0,1000,0,fmean(vals),0,0,vals])
                  else:
                    block_revenues.append([0,1000,0,fmean(vals),0,variance(vals),vals])

            '''logging.info("==Revenues from the blocks==")
            for i in range(len(block_revenues)):
                logging.info("Block: {} min:{} max:{} pseudo-mean:{} real-mean:{} sem:{} var:{} full:{}".format(self.block_incl_inds[i],block_revenues[i][0],block_revenues[i][1],block_revenues[i][2],block_revenues[i][3],block_revenues[i][4],block_revenues[i][5],block_revenues[i][6]))
            '''
            for i in range(len(block_revenues)):
                if len(block_revenues[i][6]) >=2:
                    self.block_pmean[i] = fmean(np.log(block_revenues[i][6]))
                    self.block_pvar[i] = variance(np.log(block_revenues[i][6]))
                else:
                    self.block_pvar[i] = self.p_var
                    self.block_pmean[i] = fmean(np.log(block_revenues[i][6])) if block_revenues[i][6][0] != 0 else block_revenues[i][6][0]

                self.block_mean[i] = block_revenues[i][3]
                self.block_var[i] = block_revenues[i][5]

            unexpl_block_indcs = [i for i,x in enumerate(block_revenues) if x[1] == 1000]
            #expl_block_indcs = [j for j in range(len(block_revenues)) if j not in unexpl_block_indcs ]
            logging.info("not sufficiently enough explored block indices are: {}".format(unexpl_block_indcs))
            #LBs = [ block_revenues[i][2]- self.gamma*block_revenues[i][3] for i,x in enumerate(block_revenues) ]
            LBs = [ x[2] - self.gamma*x[4] for x in block_revenues ]
            logging.info("Lower bounds are: {}".format(LBs))
            max_LB_ind = np.argmax(LBs)
            logging.info("Max lower bound is:{}".format(block_revenues[max_LB_ind][2]-self.gamma*block_revenues[max_LB_ind][4]))
            for i in unexpl_block_indcs:
              block_revenues[i][2] = block_revenues[max_LB_ind][2]
              block_revenues[i][4] = block_revenues[max_LB_ind][4]

            excl_inds = [i for i,x in enumerate(block_revenues) if (x[2]+self.gamma * x[4] < LBs[max_LB_ind]) ]
            #logging.info("excluded block indices are: {}".format(self.block_incl_inds[excl_inds]))
            self.block_excl_inds = np.union1d(self.block_excl_inds,self.block_incl_inds[excl_inds])
            Tp_excl_inds = np.isin(self.full_actions,self.block_arr[self.block_excl_inds])

            self.excl_actions[Tp_excl_inds] = self.full_actions[Tp_excl_inds]
            '''
            for j in range(len(block_revenues)):    #excl_inds:
                Tp_excl_inds = np.isin(self.full_actions,self.block_arr[self.block_incl_inds[j]])
                ex_blocks = self.full_actions[Tp_excl_inds]
                for k in ex_blocks:
                    logging.info("revenues for {} is: {}".format(k,self.revenues[tuple(k)]))
            '''
            self.block_incl_inds = np.array(list(set(self.block_incl_inds).difference(self.block_incl_inds[excl_inds])))
            if (self.block_incl_inds.shape[0]/self.block_arr.shape[0] <= 0.45):
                logging.info("Enough blocks are removed.. switching to lower block size")
                self.block_switch = 1
                self.excl_cnt-=1
            else:
                logging.info("Enough blocks are not removed ... Continuing with current block size itself")
                self.block_switch = 0


        def get_block(arr, block_dim):
            logging.info("Running block elimination")
            bdm0, bdm1 = block_dim
            blk_sz = bdm0*bdm1
            (r_inds, c_inds) = self.inds_actv_actions
            logging.info("Block size is: {} x {}".format(bdm0,bdm1))
            spl_arrs = np.split(arr[r_inds,c_inds].reshape(r_inds.shape[0]//bdm0,bdm1),r_inds.shape[0]//(bdm0*bdm1))
            block_arr = []
            for i in range(0,len(spl_arrs),bdm0):
                block_arr.extend(swap_order(spl_arrs[i:i+bdm0]))

            return np.array(block_arr)

        def swap_order(A):
            tmp = np.transpose(np.stack(A), axes=(1, 0, 2))
            return np.split(tmp.reshape(-1, len(A)), indices_or_sections = len(A))

        if self.excl_cnt > 1:
            if (self.block_switch == 1):
                block_shape = (2**(self.excl_cnt), 2**(self.excl_cnt) )
                self.block_arr = get_block(self.full_actions, block_shape)
                self.block_mean = np.zeros_like(self.block_arr, dtype='float16')
                self.block_var = np.zeros_like(self.block_arr, dtype='float16')
                self.block_pmean = np.zeros_like(self.block_arr, dtype='float16')
                self.block_pvar = np.zeros_like(self.block_arr, dtype='float16')

            block_revenues = update_bounds_get_indcs() #update lower and upper bounds
        else:
            self.arm_elm = 1
            logging.info("Running individual arm elimination")
            arm_revenues = []
            for key in self.full_actions[self.inds_actv_actions]:
                vals = self.revenues[tuple(key)]
                if len(vals) >= 3:
                    arm_revenues.append([min(vals), max(vals), fmean(vals), sem(vals), vals, key])
                else:
                    arm_revenues.append([0, 1000, 0, 0, vals, key])

            unexpl_arm_indcs = [i for i,x in enumerate(arm_revenues) if x[2] == 0]
            logging.info("Unexplored arm indices are: {}".format(unexpl_arm_indcs))
            LBs = [ arm_revenues[i][2]- self.gamma*arm_revenues[i][3] for i,x in enumerate(arm_revenues) ]
            logging.info("Lower bounds are: {}".format(LBs))
            max_LB_ind = np.argmax(LBs)
            logging.info("Max lower bound is: {}".format(arm_revenues[max_LB_ind][2]-self.gamma*arm_revenues[max_LB_ind][3]))
            for i in unexpl_arm_indcs:
              arm_revenues[i][2] = arm_revenues[max_LB_ind][2]
              arm_revenues[i][3] = arm_revenues[max_LB_ind][3]

            excl_inds = [i for i,x  in enumerate(arm_revenues) if ( x[2]+self.gamma * x[3] < LBs[max_LB_ind]) ]

            excl_arms = [ arm_revenues[i][5] for i in excl_inds ]
            logging.info("Excluded arms are: " + ', '.join('({}:{})'.format(*el) for el in excl_arms))
            Tp_excl_inds = np.isin(self.full_actions,excl_arms)
            self.excl_actions[Tp_excl_inds] = self.full_actions[Tp_excl_inds]

        Tp = ~np.isin(self.full_actions, self.excl_actions)
        if Tp.sum() == 0:
            logging.info("All actions are eliminated. Check the code")
            exit(-1)

        logging.info("Number of active arms: {}".format(Tp.sum()))
        self.inds_actv_actions = np.nonzero(Tp)    #indices of active actions
        self.flt_inds_actv_actions = np.ravel_multi_index(self.inds_actv_actions, dims=(self.ln1,self.ln2))    # flattened active actions index

        return 0

    def update(self, state, revenue, r_ind, ind):
        state_weights = np.array([0.075, 1.12, 1.18])
        elm_rnd = 40
        p1p2 = tuple(self.full_actions[ind])

        self.revenues[p1p2].append(revenue)
        #arm_sem = 2
        self.p_mean = fmean(np.log(list(itertools.chain(*self.revenues.values()))))
        if self.round > 1:
            self.p_var = variance(np.log(list(itertools.chain(*self.revenues.values()))))
            #arm_sem = sem(list(itertools.chain(*self.revenues.values())))

        arm_sem = 7 #we keep this high to 
        arm_var = 2
        arm_pvar = self.p_var
        #arm_pvar = self.p_var
        if len(self.revenues[p1p2]) > 1:
            arm_sem = sem(self.revenues[p1p2])
            arm_var = variance(self.revenues[p1p2])
            arm_pvar = variance(np.log(self.revenues[p1p2]))

        arm_mean = fmean(self.revenues[p1p2])
        arm_pmean = fmean(np.log(self.revenues[p1p2]))
        self.actions_mean[ind] = arm_mean
        self.actions_sem[ind] = arm_sem
        self.actions_var[ind] = arm_var

        self.actions_pmean[ind] = arm_pmean
        self.actions_pvar[ind] = arm_pvar

        self.means[r_ind ] = arm_mean
        self.vars[r_ind] = arm_var
        self.actions_wards[ind] += 1
        reward = 0
        #weighted_sum = np.dot(state_weights,state)

        logging.info("round:{}, arm:{}, state:{}, revenues:{}, meanr:{}, sem:{}, liner ind:{}, ind:{}, count:{}".format(self.round, p1p2, state, revenue, arm_mean, arm_sem, r_ind, ind, self.actions_wards[ind]))
        logging.info("sample mean:{} sample var:{}".format(self.p_mean,self.p_var))

        #logging.info("current bounds: {}".format((self.actions_mean-self.gamma*self.actions_sem).flatten()[self.flt_inds_actv_actions]))
        maxx_lin_ind = (self.actions_mean - self.gamma*self.actions_sem).flatten()[self.flt_inds_actv_actions].argmax()
        #maxx_lin_ind = (self.actions_mean - self.gamma*self.actions_var).flatten()[self.flt_inds_actv_actions].argmax()
        maxx_ind = np.unravel_index(self.flt_inds_actv_actions[maxx_lin_ind], dims=(self.ln1,self.ln2))
        self.argmax = self.full_actions[maxx_ind].tolist()
        self.max = [self.actions_mean[maxx_ind]]
        bnd = [self.actions_mean[maxx_ind] - self.gamma*self.actions_sem[maxx_ind]]
        logging.info("optimizer solution:{}, revenue:{}, bound:{}, index:{}, linear index:{}, count:{}".format(self.argmax, self.max, bnd, maxx_ind, maxx_lin_ind, self.actions_wards[maxx_ind]))

        if not (self.round%elm_rnd):
            self.eliminate_arms()
        return reward
