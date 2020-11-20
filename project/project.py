from pyomo.environ import *

# create model
m = AbstractModel()

m.N = Param() # number of wells
m.setN = Set()

m.K = Param(m.setN) # number of breakpoint for each well "n"
m.setK = Set() # set of (maximum number) breakpoints for each well

# A more elegant solution for setK would be to have it as an "indexed set".
# m.setK = Set(m.setN)
# Thus, each set would correspond to the different numbers of breakpoints
# for each well. This is rather easy to implement in the data file, and
# read in using the same procedure as below. However, I did not find
# an elegant way to define parameters (and variables) dependent on indexed 
# sets -- it seems it is not possible in Pyomo, while this functionality
# is in AMPL and GAMS (I think). Therefore, I choose to define everything
# for the maximum number of breakpoints (so I have more parameters and 
# variables than strictly necessary), and make sure I don't use the extra 
# parameters/variables when defining constraints.
#
# Of course, there are other ways to do this.

m.Qinj = Param(m.setN,m.setK,domain=NonNegativeReals)
m.Qoil = Param(m.setN,m.setK,domain=NonNegativeReals)

m.Qinj_max = Param(domain=NonNegativeReals)
m.Qliq_max = Param(domain=NonNegativeReals)
m.Qgas_max = Param(domain=NonNegativeReals)

m.GOR = Param(m.setN);
m.WCut = Param(m.setN)

m.lb_inj = Param(m.setN)
m.ub_inj = Param(m.setN)

m.y = Var(m.setN,domain=Binary)
m.q_inj = Var(m.setN,domain=NonNegativeReals)
m.q_oil = Var(m.setN,domain=NonNegativeReals)
m.q_water = Var(m.setN,domain=NonNegativeReals)
m.q_gas = Var(m.setN,domain=NonNegativeReals)

m.lam = Var(m.setN,m.setK,domain=NonNegativeReals)
m.z = Var(m.setN,m.setK,domain=Binary)

# Objective function
def objective_rule(m):
    return sum(m.q_oil[i] for i in m.setN)
m.Obj = Objective(rule=objective_rule,sense=maximize)

# Production constraints
def constr_qinj_max(m):
    return sum(m.q_inj[i] for i in m.setN) <= m.Qinj_max
m.constr_qinj_max = Constraint(rule=constr_qinj_max)

def constr_qliq_max(m):
    return sum(m.q_oil[i] + m.q_water[i] for i in m.setN) <= m.Qliq_max
m.constr_qliq_max = Constraint(rule=constr_qliq_max)

def constr_qgas_max(m):
    return sum(m.q_gas[i] for i in m.setN) <= m.Qgas_max
m.constr_qgas_max = Constraint(rule=constr_qgas_max)

# Define flows as piecewise linear
def constr_Qinj_pwl(m,n):
    return m.q_inj[n] == sum(m.Qinj[n,k]*m.lam[n,k] for k in RangeSet(m.K[n]))
m.constr_Qinj_pwl = Constraint(m.setN,rule=constr_Qinj_pwl)

def constr_Qoil_pwl(m,n):
    return m.q_oil[n] == sum(m.Qoil[n,k]*m.lam[n,k] for k in RangeSet(m.K[n]))
#   return m.q_oil[n] == m.Qoil[n,1] + sum((m.Qoil[n,k]-m.Qoil[n,1])*m.lam[n,k] for k in RangeSet(m.K[n]))  # To include production from wells that flow naturally, even when there is no injection
m.constr_Qoil_pwl = Constraint(m.setN,rule=constr_Qoil_pwl)

# Calculate water and gas
def constr_water(m,n):
    return m.q_water[n] == m.WCut[n]/(1-m.WCut[n])*m.q_oil[n]
m.constr_water = Constraint(m.setN,rule=constr_water)

def constr_gas(m,n):
    return m.q_gas[n] == m.GOR[n]*m.q_oil[n]
m.constr_gas = Constraint(m.setN,rule=constr_gas)

# Define constraints related to lambda and z
def constr_lam_pwl(m,n):
    return sum(m.lam[n,k] for k in RangeSet(m.K[n])) == m.y[n];
m.constr_lam_pwl = Constraint(m.setN,rule=constr_lam_pwl)

def constr_z_pwl(m,n):
    return sum(m.z[n,k] for k in RangeSet(m.K[n]-1)) == m.y[n];
m.constr_z_pwl = Constraint(m.setN,rule=constr_z_pwl)

def constr_lam_z_1(m,n):
    return m.lam[n,1] <= m.z[n,1]
m.constr_lam_z_1 = Constraint(m.setN,rule=constr_lam_z_1)

def constr_lam_z_int(m,k,n):
    if (k>=2) and (k<= m.K[n]-1): 
        rule = m.lam[n,k] <= m.z[n,k-1] + m.z[n,k] 
    else: 
        rule = m.lam[n,k] <= 1000  # just to return something
    return rule
m.constr_lam_z_int = Constraint(m.setK,m.setN,rule=constr_lam_z_int)

def constr_lam_z_n(m,n):
    return m.lam[n,m.K[n]] <= m.z[n,m.K[n]-1]
m.constr_lam_z_n = Constraint(m.setN,rule=constr_lam_z_n)


# Bound gas injection
def constr_lb_inj(m,n):
    return m.lb_inj[n]*m.y[n] <= m.q_inj[n] 
m.constr_lb_inj = Constraint(m.setN,rule=constr_lb_inj)

def constr_ub_inj(m,n):
    return m.q_inj[n] <= m.ub_inj[n]*m.y[n]
m.constr_ub_inj = Constraint(m.setN,rule=constr_ub_inj)


data = DataPortal()
data.load(filename='p1.dat')

m_instance = m.create_instance(data)

opt = SolverFactory('glpk')
results = opt.solve(m_instance)

results.write(num=1)



