#@title Recursive Block Elimination with variance dependent arm selection
from itertools import product
import numpy as np
from numpy.random import Generator, PCG64, beta, lognormal, choice
from collections import defaultdict
import itertools
from operator import itemgetter
import numpy.lib.recfunctions as rf
from math import log,sqrt,floor
from pathlib import Path
import logging
from statistics import mean,fmean,stdev, variance
from scipy.stats import sem
from sklearn.cluster import KMeans

class TS_BTHE(object):
    def __init__(self, bounds):
        self.rg = Generator(PCG64(12345))
        self.cnt1 = 16
        self.cnt2 = 16
        self.init_clus_sz = 4
        self.no_clus = self.init_clus_sz
        self.round = 0

        self.block_array = []
        self.block_mean = []
        self.block_var = []
        self.block_pmean = []
        self.block_pvar = []

        self.block_switch = 1
        self.block_inc = 1

        ar1 = np.linspace(bounds[0,0], bounds[0,1], self.cnt1, dtype='float16')
        ar2 = np.linspace(bounds[1,0], bounds[1,1], self.cnt2, dtype='float16')
        self.ln1 = ar1.shape[0]
        self.ln2 = ar2.shape[0]
        no_actions = self.cnt1*self.cnt2

        self.gamma = 2
        self.prior_samp_sz = 25
        self.deg_fd = 1
        self.p_mean = 1
        self.p_var = 0.5

        self.full_actions = np.array([*product(ar1,ar2)]).reshape(self.cnt1*self.cnt2,2) # set of actions tuples
        self.incl_actions = np.ones(no_actions,dtype=bool)
        self.actions_mean = np.zeros(no_actions, dtype='float32')
        self.actions_var = np.zeros(no_actions, dtype='float32')
        self.actions_sem = np.zeros(no_actions, dtype='float32')
        self.actions_wards = np.zeros(no_actions, dtype='uint16')

        self.actions_pmean = np.zeros(no_actions, dtype='float32')
        self.actions_pvar = np.zeros(no_actions, dtype='float32')

        self.pmeans = np.ones(no_actions, dtype='float32')*self.p_mean
        self.pvars = np.ones(no_actions, dtype='float32')*self.p_var

        self.means = np.zeros(no_actions, dtype='float32')
        self.vars = np.zeros(no_actions, dtype='float32')

        self.revenues = defaultdict(list)

        self.reset()

        log_file = Path('./logs/%s.log' %('behi'))  #logging as SuccessiveBlockElimination (sbe)
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        self.log = logging.getLogger('behi')
        logging.info("Running Successive Block Elimination algorithm with Hoeffdings Bound")

    def reset(self):
        self.max = None # maximum value observed thus far
        self.argmax = None # argument at which maximum value was observed

    def act(self,t):
        self.round = t

        self.pmeans[:] = self.p_mean
        self.pvars[:] = self.p_var
        if self.block_array:    #contains indexes of actions in each block
            for i,x in enumerate(self.block_array):
                self.pmeans[x] = self.block_pmean[i]
                self.pvars[x] = self.block_pvar[i]
                self.means[x] = self.block_mean[i]
                self.vars[x] = self.block_var[i]

        self.pmeans[np.nonzero(self.actions_pmean)] = self.actions_pmean[np.nonzero(self.actions_pmean)]
        self.pvars[np.nonzero(self.actions_pvar)] = self.actions_pvar[np.nonzero(self.actions_pvar)]
        self.means[np.nonzero(self.actions_mean)] = self.actions_mean[np.nonzero(self.actions_mean)]
        self.vars[np.nonzero(self.actions_var)] = self.actions_var[np.nonzero(self.actions_var)]

        p_var = self.rg.gamma(self.deg_fd/2, (self.deg_fd*self.pvars)/2)
        p_mean = self.rg.normal(self.pmeans, 1/(self.prior_samp_sz*self.pvars))

        incl_indx = np.nonzero(self.incl_actions)[0]
        samples = lognormal(p_mean[incl_indx], p_var[incl_indx])
        pl_arm_ind = incl_indx[choice(np.flatnonzero(samples == np.max(samples)))]
        #ravel_pl_arm_ind = self.flt_inds_actv_actions[np.argmax(lognormal(self.means[self.flt_inds_actv_actions],self.vars[self.flt_inds_actv_actions]))]
        pl_arm = self.full_actions[pl_arm_ind]
        return list(pl_arm), pl_arm_ind

    def eliminate_arms(self):
        logging.info("==== Runnung Hoeffding bound based block elimination algorithms ====")
        self.block_array = []
        self.block_mean = []
        self.block_var = []
        self.block_pmean = []
        self.block_pvar = []

        block_revenues = []
        incl_indx = np.nonzero(self.incl_actions)[0]
        def update_bounds_get_indcs():    #update the bounds of the blocks
            for i in np.arange(self.no_clus):
                i_block_indcs =  incl_indx[self.blocks.labels_ == i]
                logging.info("cluster: {} corresponding actions: {}".format(i,i_block_indcs))
                self.block_array.append(i_block_indcs)
                vals = [y for x in i_block_indcs for y in self.revenues[x]]
                if not vals:
                    vals = [0]
                if len(vals) >= 4:
                  block_revenues.append([min(vals),max(vals),fmean(vals),fmean(vals),hoeff(vals),variance(vals),vals])
                else:
                  if len(vals) < 2:
                    block_revenues.append([0,1000,0,fmean(vals),0,0,vals])
                  else:
                    block_revenues.append([0,1000,0,fmean(vals),0,variance(vals),vals])

            '''logging.info("==Revenues from the blocks==")
            for i in range(len(block_revenues)):
                logging.info("Block: {} min:{} max:{} pseudo-mean:{} real-mean:{} hoeff:{} var:{} full:{}".format(self.block_incl_inds[i],block_revenues[i][0],block_revenues[i][1],block_revenues[i][2],block_revenues[i][3],block_revenues[i][4],block_revenues[i][5],block_revenues[i][6]))
            '''
            for i in range(len(block_revenues)):
                if len(block_revenues[i][6]) >=2:
                    self.block_pmean.append(fmean(np.log(block_revenues[i][6])))
                    self.block_pvar.append(variance(np.log(block_revenues[i][6])))
                else:
                    self.block_pvar.append(self.p_var)
                    self.block_pmean.append(fmean(np.log(block_revenues[i][6]))) if block_revenues[i][6][0] != 0 else self.block_pmean.append(block_revenues[i][6][0])

                self.block_mean.append(block_revenues[i][3])
                self.block_var.append(block_revenues[i][5])

            unexpl_block_indcs = [i for i,x in enumerate(block_revenues) if x[1] == 1000]
            logging.info("not sufficiently enough explored block indices are: {}".format(unexpl_block_indcs))
            LBs = [ x[2] - self.gamma*x[4] for x in block_revenues ]
            logging.info("Lower bounds are: {}".format(LBs))
            max_LB_ind = np.argmax(LBs)
            logging.info("Max lower bound is:{}".format(block_revenues[max_LB_ind][2]-self.gamma*block_revenues[max_LB_ind][4]))
            for i in unexpl_block_indcs:
              block_revenues[i][2] = block_revenues[max_LB_ind][2]
              block_revenues[i][4] = block_revenues[max_LB_ind][4]

            excl_inds = [i for i,x in enumerate(block_revenues) if (x[2]+self.gamma * x[4] < LBs[max_LB_ind]) ]
            for i in excl_inds:
                self.incl_actions[self.block_array[i]] = False
            elm_ratio = ((~self.incl_actions).sum())/((self.incl_actions).sum())
            logging.info("elimination ratio is: {}".format(elm_ratio))
            if ( elm_ratio < 7):
                logging.info("Enough blocks are not removed.. continue blocking")
                self.block_switch = 1
                self.block_inc = floor(elm_ratio)+1
            else:
                logging.info("Enough blocks are removed .. Continuing with individual arm selection")
                self.block_switch = 0

        def hoeff(vals):
            return np.sqrt(np.log(self.round)/(2*len(vals)))

        def get_block(arr):
            self.no_clus = min(self.init_clus_sz*self.block_inc, len(incl_indx))
            logging.info("Block the actions into {} clusters".format(self.no_clus))
            return KMeans(n_clusters=self.no_clus, random_state=0).fit(arr)

        if (self.block_switch == 1):    # do clustering
            self.blocks = get_block(self.full_actions[self.incl_actions])
            update_bounds_get_indcs() #update lower and upper bounds
        else:
            logging.info("Running individual arm elimination")
            arm_revenues = []
            for ind in incl_indx:
                vals = self.revenues[ind]
                if len(vals) >= 3:
                    arm_revenues.append([min(vals), max(vals), fmean(vals), hoeff(vals), vals])
                else:
                    arm_revenues.append([0, 1000, 0, 0, vals])

            unexpl_arm_indcs = [i for i,x in enumerate(arm_revenues) if x[2] == 0]
            logging.info("Unexplored actions are: {}".format(incl_indx[unexpl_arm_indcs]))
            LBs = [ arm_revenues[i][2]- self.gamma*arm_revenues[i][3] for i,x in enumerate(arm_revenues) ]
            logging.info("Lower bounds are: {}".format(LBs))
            max_LB_ind = np.argmax(LBs)
            logging.info("Max lower bound is: {}".format(arm_revenues[max_LB_ind][2]-self.gamma*arm_revenues[max_LB_ind][3]))
            for i in unexpl_arm_indcs:
              arm_revenues[i][2] = arm_revenues[max_LB_ind][2]
              arm_revenues[i][3] = arm_revenues[max_LB_ind][3]

            excl_inds = [i for i,x  in enumerate(arm_revenues) if ( x[2]+self.gamma * x[3] < LBs[max_LB_ind]) ]
            #logging.info("Excluded arms are: " + ','.join('{}'.format(incl_indx[el]) for el in excl_inds))
            #logging.info("Excluded arms are: {}".format(incl_indx[excl_inds]))
            for i in excl_inds:
                self.incl_actions[incl_indx[i]] = False

        if self.incl_actions.sum() == 0:
            logging.info("All actions are eliminated. Check the code")
            exit(-1)

        logging.info("Number of active actions: {}".format(self.incl_actions.sum()))

        return 0

    def update(self, state, revenue, ind):
        incl_indx = np.nonzero(self.incl_actions)[0]

        def hoeff(vals):
            return np.sqrt(np.log(self.round)/(2*len(vals)))

        state_weights = np.array([0.075, 1.12, 1.18])
        elm_rnd = 1.5*self.init_clus_sz

        self.revenues[ind].append(revenue)
        #arm_sem = 2
        self.p_mean = fmean(np.log(list(itertools.chain(*self.revenues.values()))))
        if self.round > 1:
            self.p_var = variance(np.log(list(itertools.chain(*self.revenues.values()))))
            #arm_sem = sem(list(itertools.chain(*self.revenues.values())))

        arm_sem = -2 #we keep this high to 
        arm_var = 2
        arm_pvar = self.p_var
        #arm_pvar = self.p_var
        if len(self.revenues[ind]) > 1:
            arm_sem = hoeff(self.revenues[ind])
            arm_var = variance(self.revenues[ind])
            arm_pvar = variance(np.log(self.revenues[ind]))

        arm_mean = fmean(self.revenues[ind])
        arm_pmean = fmean(np.log(self.revenues[ind]))
        self.actions_mean[ind] = arm_mean
        self.actions_sem[ind] = arm_sem
        self.actions_var[ind] = arm_var

        self.actions_pmean[ind] = arm_pmean
        self.actions_pvar[ind] = arm_pvar

        self.means[ind ] = arm_mean
        self.vars[ind] = arm_var
        self.actions_wards[ind] += 1
        reward = 0
        #weighted_sum = np.dot(state_weights,state)

        logging.info("round:{}, arm:{}, state:{}, revenues:{}, meanr:{}, sem:{}, ind:{}, count:{}".format(self.round, self.full_actions[ind].tolist(), state, revenue, arm_mean, arm_sem, ind, self.actions_wards[ind]))
        logging.info("sample mean:{} sample var:{}".format(self.p_mean,self.p_var))

        #logging.info("current bounds: {}".format((self.actions_mean-self.gamma*self.actions_sem).flatten()[self.flt_inds_actv_actions]))
        maxx_lin_ind = (self.actions_mean + self.gamma*self.actions_sem)[incl_indx].argmax()
        maxx_ind = incl_indx[maxx_lin_ind]
        self.argmax = self.full_actions[maxx_ind].tolist()
        self.max = [self.actions_mean[maxx_ind]]
        bnd = [self.actions_mean[maxx_ind] + self.gamma*self.actions_sem[maxx_ind]]
        logging.info("optimizer solution:{}, revenue:{}, bound:{}, index:{}, linear index:{}, count:{}".format(self.argmax, self.max, bnd, maxx_ind, maxx_lin_ind, self.actions_wards[maxx_ind]))

        if not (self.round%elm_rnd):
            self.eliminate_arms()
        return reward
