# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:26:33 2020

@author: lsi
"""

from pyomo.environ import *

# create model
m = AbstractModel()

m.N = Param() # number of wells
m.setN = Set()

m.K = Param(m.setN) # number of breakpoint for each well "n"
m.setK = Set() # set of breakpoints for each well

m.Qinj = Param(m.setN, m.setK, domain=NonNegativeReals)
m.Qoil = Param(m.setN, m.setK, domain=NonNegativeReals)

m.Qinj_max = Param(domain=NonNegativeReals)
m.Qliq_max = Param(domain=NonNegativeReals)
m.Qgas_max = Param(domain=NonNegativeReals)

m.GOR = Param(m.setN)
m.WCut = Param(m.setN)

m.lb_inj = Param(m.setN)
m.ub_inj = Param(m.setN)


# Variables
m.q_oil_hat = Var(m.setN, domain=NonNegativeReals)
m.q_inj_hat = Var(m.setN, domain=NonNegativeReals)
m.y = Var(m.setN, domain=Binary)
m.lam = Var(m.setN, m.setK, domain=NonNegativeReals) # todo: test med m.K senere
m.z = Var(m.setN, m.setK, domain=Binary)

# Objective function
def obj_func(m):
    return summation(m.q_oil_hat)

m.obj = Objective(rule=obj_func, sense=maximize)

# Constraints
def total_inj_ub(m):
    return summation(m.q_inj_hat) <= m.Qinj_max

def total_liq_ub(m):
    return sum([m.q_oil_hat[n] / (1 - m.WCut[n]) for n in m.setN]) <= m.Qliq_max

def total_gas_ub(m):
    return sum([m.q_inj_hat[n] + m.GOR[n] * m.q_oil_hat[n] for n in m.setN]) <= m.Qgas_max # calc q_g

# TODO: Find out what the Qoil_max values are
def oil_prod_ub(m, n):
    return m.q_oil_hat[n] <= 1000000 * m.y[n]

def q_inj_min_constraint(m, n):
    return m.lb_inj[n] * m.y[n] <= m.q_inj_hat[n]

def q_inj_max_constraint(m, n):
    return m.q_inj_hat[n] <= m.ub_inj[n] * m.y[n]

m.c2b = Constraint(rule=total_inj_ub)
m.c2c = Constraint(rule=total_liq_ub)
m.c2d = Constraint(rule=total_gas_ub)
#m.c2f = Constraint(m.setN, rule=oil_prod_ub)
m.c2g1 = Constraint(m.setN, rule=q_inj_min_constraint)
m.c2g2 = Constraint(m.setN, rule=q_inj_max_constraint)

# Convex Combination Constraints
def cc_constraint1(m, n):
    return m.q_inj_hat[n] == sum([m.lam[n,k] * m.Qinj[n,k] for k in range(1, m.K[n]+1)])

def cc_constraint2(m, n):
    return m.q_oil_hat[n] == sum([m.lam[n,k] * m.Qoil[n,k] for k in range(1, m.K[n]+1)])
    
def cc_constraint3(m, n):
    return 1 == sum([m.lam[n,k] for k in range(1, m.K[n]+1)])

def cc_constraint4(m, n):
    return 1 == sum([m.z[n,k] for k in range(2, m.K[n]+1)])

def cc_constraint5(m, n, k):
    # tilfellet lambda i, 0 < i < n
    if k > 1 and k < m.K[n]:
        return m.lam[n, k] <= m.z[n,k] + m.z[n, k+1]
    else:
        return Constraint.Skip

def cc_constraint6(m, n):
    # tilfellet lambda 0
    return m.lam[n, 1] <= m.z[n,2]

def cc_constraint7(m, n):
    # tilfellet lambda n
    return m.lam[n, m.K[n]] <= m.z[n, m.K[n]]


m.cc1 = Constraint(m.setN, rule=cc_constraint1)    
m.cc2 = Constraint(m.setN, rule=cc_constraint2)
m.cc3 = Constraint(m.setN, rule=cc_constraint3)
m.cc4 = Constraint(m.setN, rule=cc_constraint4)
m.cc5 = Constraint(m.setN, m.setK, rule=cc_constraint5)
m.cc6 = Constraint(m.setN, rule=cc_constraint6)
m.cc7 = Constraint(m.setN, rule=cc_constraint7)

# Load data from dat file
data = DataPortal()
data.load(filename='p1.dat')

m_instance = m.create_instance(data)

# Solve problem
opt = SolverFactory('glpk')
results = opt.solve(m_instance)
#m_instance.q_inj_hat.display()
#m_instance.lam.display()
#m_instance.y.display()
m_instance.display()
