# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:26:33 2020

@author: lsi
"""

from pyomo.environ import *
from pyomo.gdp import Disjunct
import matplotlib.pyplot as plt
import numpy as np
from pyomo.kernel import sos
import inspect
from pyomo.core.kernel.piecewise_library.transforms import *
data = DataPortal()
data.load(filename='p1.dat')


data_list = [tup[1] for tup in list(data.items())]
N = data_list[0]
setN = data_list[1]
setK = data_list[2]
K = [data_list[5][n+1] for n in range(N)]
lb_inj = [data_list[6][n+1] for n in range(N)]
ub_inj = [data_list[7][n+1] for n in range(N)]
Qinj_max = data_list[8]
Qliq_max = data_list[9]
Qgas_max = data_list[10]
WCut = [data_list[11][n+1] for n in range(N)]
GOR = [data_list[12][n+1] for n in range(N)]
Qinj = [[data_list[13][n+1,k+1] for k in range(K[n])]for n in range(N)]
Qoil = [[data_list[14][n+1,k+1] for k in range(K[n])]for n in range(N)]



def getRequiredArgs(func):
    args, varargs, varkw, defaults = inspect.getargspec(func)
    if defaults:
        args = args[:-len(defaults)]
    return args   # *args and **kwargs are not required, so ignore them.


# %% create model
def general_model():
    m = AbstractModel()
    
    m.N = Param() # number of wells
    m.setN = Set()
    
    m.K = Param(m.setN) # number of breakpoint for each well "n"
    m.setK = Set() # set of breakpoints for each well
    
    m.setNK = Set()
    m.setN_KEND = Set()
    
    m.Qinj = Param(m.setN,m.setK,domain=NonNegativeReals)
    m.Qoil = Param(m.setN,m.setK,domain=NonNegativeReals)
    
    m.Qinj_max = Param(domain=NonNegativeReals, mutable=True)
    m.Qliq_max = Param(domain=NonNegativeReals)
    m.Qgas_max = Param(domain=NonNegativeReals)
    
    m.GOR = Param(m.setN)
    m.WCut = Param(m.setN)
    Disjunct()
    m.lb_inj = Param(m.setN)
    m.ub_inj = Param(m.setN)
    
    m.y = Var(m.setN, bounds = (0,1), within=Binary)#, initialize=0)
    m.Qinj_var = Var(m.setN, within=NonNegativeReals, initialize=225)#, initialize= max(_inj))
    m.Qoil_var = Var(m.setN, within=NonNegativeReals)#, initialize=0)
    m.Qw_var = Var(m.setN, within=NonNegativeReals)#, initialize=0)
    m.Qg_var = Var(m.setN, within=NonNegativeReals)#, initialize=0)
    m.lbd = Var(m.setNK, bounds=(0,1), within=NonNegativeReals)
    
    
    
# %% Equalities:
    def q_g_constraint(m, i):
        return m.Qg_var[i] == m.GOR[i] * m.Qoil_var[i]
    def q_w_constraint(m,i):
        return m.Qw_var[i] == m.WCut[i]/(1-m.WCut[i]) * m.Qoil_var[i]

    # %% Objective
    def objRule(m):
        return sum(m.Qoil_var[:])
    
    # %% Inequalities
    def inj_constraint(m):
        return sum(m.Qinj_var[k] for k in m.setN) <= m.Qinj_max
    def liq_constraint(m):
        return sum(m.Qoil_var[k] + m.Qw_var[k] for k in m.setN) <= m.Qliq_max
    def gas_constraint(m):
        return sum(m.Qinj_var) + sum(m.Qg_var) <= m.Qgas_max

    def q_inj_binary_constraint_lb(m,i):
        return m.Qinj_var[i] >= m.lb_inj[i]*m.y[i]
    def q_inj_binary_constraint_ub(m,i):
        return m.Qinj_var[i] <= m.ub_inj[i]*m.y[i]
    
    m.c_inj = Constraint(rule=inj_constraint)
    m.c_liq = Constraint(rule=liq_constraint)
    m.c_gas = Constraint(rule=gas_constraint)
    m.c_q_g = Constraint(m.setN, rule=q_g_constraint)
    m.c_q_w = Constraint(m.setN, rule=q_w_constraint)
    m.c_q_inj_bin_lb = Constraint(m.setN, rule=q_inj_binary_constraint_lb)
    m.c_q_inj_bin_ub = Constraint(m.setN, rule=q_inj_binary_constraint_ub)
    
    m.OBJ = Objective(rule=objRule, sense=maximize)

    return m

# %% Convex Combination Constraints
def add_CC_constraints(m):
    m.z = Var(m.setNK,within=Binary, initialize=0)#, initialize=0)
    
    def CC_x_constraint(m,n):
        Qinj_n = list(m.Qinj[n,:])
        return  sum([m.lbd[n,k+1] * Qinj_n[k] for k in range(K[n-1])])== m.Qinj_var[n]
    def CC_y_constraint(m,n):
        Qoil_n = list(m.Qoil[n,:])
        return  sum([m.lbd[n,k+1] * Qoil_n[k] for k in range(K[n-1])])== m.Qoil_var[n]
    def CC_lbd_sum_constraint(m,n):
        return 1 == sum(m.lbd[n,:])
    def CC_lbd_k_constraint(m,n,k):
        return m.lbd[n,k] <= m.z[n,k]
    def CC_lbd_end_constraint(m,n,k):
        return m.lbd[n,k] <= m.z[n,k]
    def CC_z_constraint(m,n):
        return sum(m.z[n,:]) == 1
    
    m.CC_x = Constraint(m.setN, rule=CC_x_constraint)
    m.CC_y = Constraint(m.setN, rule=CC_y_constraint)
    m.CC_lbd = Constraint(m.setN,rule=CC_lbd_sum_constraint)
    m.CC_lbd_k = Constraint(m.setNK, rule=CC_lbd_k_constraint)
    m.CC_lbd_end = Constraint(m.setN_KEND, rule=CC_lbd_end_constraint)
    m.CC_z = Constraint(m.setN,rule=CC_z_constraint)
    return m

# %% SOS2
def add_SOS2_constraints(m):    
    m.setNK_1 = Set()
    m.setNK_2 = Set()
    m.setNK_3 = Set()
    m.setNK_4 = Set()
    m.setNK_5 = Set()
    m.setNK_6 = Set()
    m.setNK_7 = Set()
    m.setNK_8 = Set()


    def SOS2_x_constraint(m,n):
        return sum([m.lbd[n,k+1]*m.Qinj[n,k+1] for k in range(m.K[n])]) == m.Qinj_var[n]
    
    def SOS2_y_constraint(m,n):
        return sum([m.lbd[n,k+1]*m.Qoil[n,k+1] for k in range(m.K[n])]) == m.Qoil_var[n]
    
    def SOS2_lbd_constraint(m,n):
        return sum(m.lbd[n,:]) == 1
    
    
    m.SOS2_x = Constraint(m.setN, rule=SOS2_x_constraint)
    m.SOS2_y = Constraint(m.setN, rule=SOS2_y_constraint)
    m.SOS2_lbd = Constraint(m.setN, rule=SOS2_lbd_constraint)
    
    return m



# %% Construct CC-model
m_general = general_model()
m_CC = add_CC_constraints(m_general.clone())
m_SOS2 = add_SOS2_constraints(m_general.clone())

mCC = m_CC.create_instance(data)
mS = m_SOS2.create_instance(data)
mS.SOS2_con = []

for i in range(N):
    mS.SOS2_con.append(piecewise(mS.Qinj[i+1,:], mS.Qoil[i+1,:], \
    input=mS.Qinj_var, output=mS.Qoil_var, \
    require_bounded_input_variable=False, equal_slopes_tolerance=1e-16))
    
#%% Solve

opt = SolverFactory('glpk')
results_CC = opt.solve(mCC, )
results_SOS2 = opt.solve(mS)

# %% Retrieve solution:
Qinj_sol_CC = list(mCC.Qinj_var.get_values().values())
Qoil_sol_CC = list(mCC.Qoil_var.get_values().values())

Qinj_sol_SOS2 = list(mS.Qinj_var.get_values().values())
Qoil_sol_SOS2 = list(mS.Qoil_var.get_values().values())

# %% Plot
fig1, ax1 = plt.subplots(2,1)
K = mCC.K.sparse_values()
Qoil_vals = mCC.Qoil.sparse_values()
Qinj_vals = mCC.Qinj.sparse_values()
Qoil_vec = []
Qinj_vec = []

ind = 0
for i in range(len(K)):
    ax1[0].plot(Qinj[i], Qoil[i], label=str(i))
    ax1[0].scatter(Qinj_sol_CC[i], Qoil_sol_CC[i], marker='x', c='g')
    
    ax1[1].plot(Qinj[i], Qoil[i])
    ax1[1].scatter(Qinj_sol_SOS2[i], Qoil_sol_SOS2[i], marker='x', c='k')
    ind += K[i]
ax1[0].legend()
plt.show()

# %% Constraint Multipliers:
CC_lbc = mCC.lbd.extract_values()
SOS2_lbc = mS.lbd.extract_values()

CC_active = []
SOS2_active = []
for val, ind in enumerate(CC_lbc):
    if CC_lbc[ind] != 0:
        CC_active.append(ind)
for val, ind in enumerate(SOS2_lbc):
    if SOS2_lbc[ind] != 0:
        SOS2_active.append(ind)

print("Active constraints, CC: ", CC_active, ", SOS2: ", SOS2_active)
                     


# %% Solve for varying Qinj_max:

Qinj_max_vals = list(np.linspace(0,1500,20))
Qoil_val_CC = []
Qoil_val_SOS = []
Qinj_val_CC = []
Qinj_val_SOS = []
for val in Qinj_max_vals:
    mCC.Qinj_max = val
    mS.Qinj_max = val
    result_CC = opt.solve(mCC)
    result_SOS = opt.solve(mS)
    Qoil_val_CC.append(sum(list(mCC.Qoil_var.get_values().values())))
    Qinj_val_CC.append(sum(list(mCC.Qinj_var.get_values().values())))
    Qoil_val_SOS.append(sum(list(mS.Qoil_var.get_values().values())))
    Qinj_val_SOS.append(sum(list(mS.Qinj_var.get_values().values())))

# %% Plot

fig2, ax2 = plt.subplots(num=1)
ax2.plot(Qinj_val_CC, Qoil_val_CC, label="Convex Combination")
ax2.plot(Qinj_val_SOS, Qoil_val_SOS, label="SOS2")
ax2.grid()
ax2.legend()

