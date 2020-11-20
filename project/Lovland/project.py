# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:26:33 2020

@author: lsi ft. kl
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
from pyomo.environ import *

# Create model
m = AbstractModel()

m.N = Param() # number of wells
m.setN = Set()

m.K = Param(m.setN) # number of breakpoint for each well "n"
m.setK = Set() # set of breakpoints for each well

m.Qinj = Param(m.setN,m.setK,domain=NonNegativeReals)
m.Qoil = Param(m.setN,m.setK,domain=NonNegativeReals)

m.Qinj_max = Param(domain=NonNegativeReals, mutable=True)
m.Qliq_max = Param(domain=NonNegativeReals)
m.Qgas_max = Param(domain=NonNegativeReals)

m.GOR = Param(m.setN);
m.WCut = Param(m.setN)

m.lb_inj = Param(m.setN)
m.ub_inj = Param(m.setN)

# Define system variables, constraints and objective function
m.Qinj_var = Var(m.setN, domain=NonNegativeReals)
m.Qoil_hat = Var(m.setN, domain=NonNegativeReals)
m.Qwater = Var(m.setN, domain=NonNegativeReals)
m.Qgas = Var(m.setN, domain=NonNegativeReals)
m.y = Var(m.setN, domain=Binary)

m.lmbd = Var(m.setN, m.setK, domain=NonNegativeReals)
m.z = Var(m.setN, m.setK, domain=Binary)

modelling_method = "CC"

# (2a)
def Qoil_tot(m):
    return summation(m.Qoil_hat)
m.obj = Objective(rule=Qoil_tot, sense=maximize)

# (2b)
def Qinj_constraint(m):
    return summation(m.Qinj_var) <= m.Qinj_max
m.con2b = Constraint(rule=Qinj_constraint)

# (2c)
def liquid_constraint(m):
    return summation(m.Qwater) + summation(m.Qoil_hat) <= m.Qliq_max
m.con2c = Constraint(rule=liquid_constraint)

# (2d)
def gas_constraint(m):
    return summation(m.Qinj_var) + summation(m.Qgas) <= m.Qgas_max
m.con2d = Constraint(rule=gas_constraint)

# (2e)
def gas_definition(m, n):
    return m.Qgas[n] == m.GOR[n] * m.Qoil_hat[n]
m.con2ei = Constraint(m.setN, rule=gas_definition)

def water_definition(m, n):
    return m.Qwater[n] == m.WCut[n] / (1 - m.WCut[n]) * m.Qoil_hat[n]
m.con2eii = Constraint(m.setN, rule=water_definition)

# (2f)
if modelling_method == "CC":
    def qinj_definition(m, n):
        return m.Qinj_var[n] == sum([m.Qinj[n, k] * m.lmbd[n, k] for k in range(1, m.K[n]+1)])
    m.con2fi = Constraint(m.setN, rule=qinj_definition)

    def qoil_definition(m, n):
        return m.Qoil_hat[n] == sum([m.Qoil[n, k] * m.lmbd[n, k] for k in range(1, m.K[n]+1)])
    m.con2fii = Constraint(m.setN, rule=qoil_definition)

    def lambda_sums_to_one(m, n):
        return sum([m.lmbd[n, k] for k in range(1, m.K[n]+1)]) == 1
    m.con2fiii = Constraint(m.setN, rule=lambda_sums_to_one)

    def z_constraint(m, n):
        return sum([m.z[n, k] for k in range(1, m.K[n]+1)]) == 1
    m.con2fiv = Constraint(m.setN, rule=z_constraint)

    m.z0_is_nonexistent = Constraint(m.setN, rule=(lambda m, n: m.z[n, 1] == 0))

    def lambda_constraint_lower(m, n):
        return m.lmbd[n, 1] <= m.z[n, 2]
    m.con2fv = Constraint(m.setN, rule=lambda_constraint_lower)

    def lambda_constraint_middle(m, n, k):
        # Only evaluate k up to K-1
        if k < m.K[n]:
            return m.lmbd[n, k] <= m.z[n, k] + m.z[n, k+1]
        else:
            return Constraint.Feasible
    m.cons2fvi = Constraint(m.setN, m.setK, rule=lambda_constraint_middle)    

    def lambda_constraint_upper(m, n):
        return m.lmbd[n, m.K[n]] <= m.z[n, m.K[n]]
    m.con2fvii = Constraint(m.setN, rule=lambda_constraint_upper)

    def kickoff_rate_lower(m, n):
        return m.Qinj_var[n] >= m.lb_inj[n]*m.y[n]
    m.con2fiix = Constraint(m.setN, rule=kickoff_rate_lower)

    def kickoff_rate_higher(m, n):
        return m.Qinj_var[n] <= m.ub_inj[n]*m.y[n]
    m.con2fix = Constraint(m.setN, rule=kickoff_rate_higher)
   
    # TODO: Use proper naming convention, not the roman numeral mess which is used now 

# Load data
data = DataPortal()
data.load(filename='p1.dat')

m_instance = m.create_instance(data)

# Define list that may be copied directly into LaTeX
Qinj_max_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
                 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450,
                 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600,
                 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750,
                 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900,
                 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040,
                 1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170,
                 1180, 1190, 1200, 1210, 1220, 1230, 1240, 1250, 1260, 1270, 1280, 1290, 1300,
                 1310, 1320, 1330, 1340, 1350, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 
                 1440, 1450, 1460, 1470, 1480, 1490, 1500]

for Qinj_max in tqdm(Qinj_max_list, ascii=True):
    m_instance.Qinj_max = Qinj_max
    
    # Optimize
    opt = SolverFactory('glpk')
    results = opt.solve(m_instance)

    # Plot stuff
    K = [k for k in m_instance.K.values()]
    N = m_instance.N.value

    Qinj = [[m_instance.Qinj._data[(i, j)] for j in range(1, K[i-1] + 1)] for i in range(1, N+1)]
    Qoil = [[m_instance.Qoil._data[(i, j)] for j in range(1, K[i-1] + 1)] for i in range(1, N+1)]
    Qinj_var = [m_instance.Qinj_var._data[i].value for i in range(1, N+1)]
    Qoil_hat = [m_instance.Qoil_hat._data[i].value for i in range(1, N+1)]
    Qoil_tot_list.append(sum(Qoil_hat))
    lb_inj = [m_instance.lb_inj._data[i] for i in range(1, N+1)]

    for i in range(N):
        Qinj[i].insert(1, lb_inj[i])
        Qoil[i].insert(1, Qoil[i][0])

    fig, [[ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]] = plt.subplots(2, 4)
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

    for i in range(N):
        axs[i].plot(Qinj[i], Qoil[i], label="Flowrate vs gaslift curve")
        axs[i].plot(Qinj_var[i], Qoil_hat[i], 'ro', label="Operating point")
        axs[i].set_xlabel('Qinj')
        axs[i].set_ylabel('Qoil')

    fig.set_size_inches(12, 6)
    #! This will overwrite the current, saved files
    # fig.savefig("figures/operating_points_qinj_max=" + str(Qinj_max)  + ".pdf", format='pdf')
    # plt.show()
    plt.close(fig)

fig, ax = plt.subplots()
ax.plot(Qinj_max_list, Qoil_tot_list)
ax.set_xlabel('Qinj_max')
ax.set_ylabel('Qtot')
#! Uncomment only if you want to overwrite figure
# fig.savefig("figures/total_production.pdf", format='pdf')
plt.show()
