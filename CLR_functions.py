"""
Last update: Fri Aug 12 15:32 2022

@author: Oscar F. Carrasco Heine
"""

from gurobipy import *

import elkai
import math
import networkx as nx
import numpy as np
import re
import time

## Gurobi callback: Interrupt Gurobi if improving the incumbent solution takes longer than a certain time limit

def interruptcallback(model, where):
    if where == GRB.Callback.MIP:
        if time.time() - model._timereset > model._intlim:
            model.terminate()
    elif where == GRB.Callback.MIPSOL:
        model._timereset = time.time()

## Read file: extract number of customers and number of possible facility locations

def get_parameters(file):
    
    with open(file) as f:
        ncus = int(f.readline().strip()) # number of customers
        nloc = int(f.readline().strip()) # number of possible locations
    
    ## Identify lines containing values of interest
    
    loc_end = 3 + nloc
    cus_end = loc_end + 1 + ncus
    vehcap_line = cus_end + 1
    depcap_end = vehcap_line + 2 + nloc
    cusdem_end = depcap_end + 1 + ncus
    depcos_end = cusdem_end + 1 + nloc
    intbool_line = depcos_end + 3
    routecost_line = intbool_line - 2
    
    loc_strarray = []
    cus_strarray = []
    vehcap_strarray = []
    depcap_strarray = []
    cusdem_strarray = []
    depcos_strarray = []
    intbool_strarray = []
    routecost_strarray = []
    
    with open(file) as f:
        for i, line in enumerate(f):
            if i in range(3,loc_end):
                loc_strarray.append(line)
            elif i in range(loc_end+1,cus_end):
                cus_strarray.append(line)
            elif i == vehcap_line:
                vehcap_strarray.append(line)
            elif i in range(vehcap_line+2,depcap_end):
                depcap_strarray.append(line)
            elif i in range(depcap_end+1,cusdem_end):
                cusdem_strarray.append(line)
            elif i in range(cusdem_end+1,depcos_end):
                depcos_strarray.append(line)
            elif i == routecost_line:
                routecost_strarray.append(line)
            elif i == intbool_line:
                intbool_strarray.append(line)
    
    ## Transform arrays of strings into arrays of numbers

    loc_array = []
    cus_array = []
    U = int(vehcap_strarray[0]) # Vehicle capacity
    depcap_array = []
    cusdem_array = []
    depcos_array = []
    intbool = int(intbool_strarray[0])
    routecost = float(routecost_strarray[0])
    
    for i in range(nloc):
        loc_array.append([float(s) for s in loc_strarray[i].split()])
        depcap_array.append([int(s) for s in depcap_strarray[i].split()][0])
        depcos_array.append([float(s) for s in depcos_strarray[i].split()][0])
    
    for i in range(ncus):
        cus_array.append([float(s) for s in cus_strarray[i].split()])
        cusdem_array.append([int(s) for s in cusdem_strarray[i].split()][0])
    
    print('\nPossible facility locations: ' + str(len(depcap_array)))
    print('Amount of customers: ' + str(len(cusdem_array)))
    
    return ncus, nloc, loc_array, cus_array, U, depcap_array, cusdem_array, depcos_array, intbool, routecost


## Function used to convert costs in case they should be integer

def conv_int_cost(x, intbool):
    if intbool == 0:
        return math.floor(100*x)
    else:
        return x


## Define and solve MST
        
def solve_mst(nloc, ncus, loc_array, cus_array, intbool, routecost, depcos_array):
    
    print('\nSolving MST to obtain F_1:')
    
    ## Create modified graph
    
    Nodes = []
    
    for i in range(nloc):
        Nodes.append('facility ' + str(i))
    for i in range(ncus):
        Nodes.append(i)
    
    G = nx.complete_graph(Nodes)
    
    for i in range(nloc):
        G.nodes['facility ' + str(i)]['type'] = 'facility'
    
    for i in range(ncus):
        G.nodes[i]['type'] = 'customer'
    
    G.add_node('r')
    G.nodes['r']['type'] = 'aux'
    
    for i in range(nloc):
        G.add_edge('r', 'facility ' + str(i))
        G.edges['r', 'facility ' + str(i)]['weight'] = 0
    
    for i in range(ncus-1):
        for j in range(i+1,ncus):
            G.edges[i, j]['weight'] = conv_int_cost(math.sqrt(pow(cus_array[i][0] - cus_array[j][0],2) + pow(cus_array[i][1] - cus_array[j][1],2)), intbool) # Distance between customers
    
    for i in range(nloc-1):
        for j in range(i+1, nloc):
            G.edges['facility ' + str(i), 'facility ' + str(j)]['weight'] = conv_int_cost(math.sqrt(pow(loc_array[i][0] - loc_array[j][0],2) + pow(loc_array[i][1] - loc_array[j][1],2)), intbool) # Distance between facilities
    
    for i in range(nloc):
        for j in range(ncus):
            G.edges['facility ' + str(i), j]['weight'] = conv_int_cost(math.sqrt(pow(loc_array[i][0] - cus_array[j][0],2) + pow(loc_array[i][1] - cus_array[j][1],2)), intbool) + 0.5*routecost + 0.5*depcos_array[i] # Distance from facilities to customers
    
    ## Calculate MST
    
    T = nx.minimum_spanning_tree(G)
    
    LB_MST = 0
    for e in T.edges(): LB_MST += T.edges[e]['weight'] # MST Lower Bound
    
    ## Final step: identify facilities assigned to at least one customer
    
    set_F1 = []
    
    for i in range(nloc):
        if len(T['facility ' + str(i)]) > 1:
            set_F1.append(i)
    
    print('\nF_1 = ')
    print(set_F1)

    return T, set_F1, LB_MST    


## Fill set with nodes belonging to a subtree

def sub_tree_set(X, T, i):
    X.append(i)
    if len(T.nodes[i]['children']) > 0:
        for j in T.nodes[i]['children']:
            sub_tree_set(X, T, j)


## Recursively compute the load of a customer

def compute_load(T, i, out = False):
    load = T.nodes[i]['demand']
    if len(T.nodes[i]['children']) > 0:
        for j in T.nodes[i]['children']:
            load += compute_load(T, j, True)
    T.nodes[i]['load'] = load
    if out:
        return load    


## Subtree relieving procedure
    
def relieve(T, nloc, ncus, cusdem_array, set_F1, U, SDG_mod):
    
    print('\nRelieving overloaded subtrees...')

    ## We begin by setting all necessary attributes of the tree
    
    T.nodes['r']['children'] = []
    for i in range(nloc):
        T.nodes['r']['children'].append('facility ' + str(i))
    
    for f in range(nloc):
        T.nodes['facility ' + str(f)]['children'] = []
        T.nodes['facility ' + str(f)]['parent'] = 'r'
    
    for c in range(ncus):
        T.nodes[c]['demand'] = cusdem_array[c]
        T.nodes[c]['children'] = []
    
    A = set_F1[:]
    B = []
    for a in A:
        for i in T['facility ' + str(a)]:
            if i != T.nodes['facility ' + str(a)]['parent']:
                T.nodes['facility ' + str(a)]['children'].append(i)
                T.nodes[i]['parent'] = 'facility ' + str(a)
                B.append(i)
    
    F1_cus = B[:] # Customers connected directly to open facilities
    
    A = B[:]
    while len(A) > 0:
        B = []
        for a in A:
            if len(T[a]) > 1: # We create a copy of all nodes that are not leaves
                name = str(a) + '-copy'
                T.add_node(name)
                T.add_edge(name, a)
                T.nodes[name]['type'] = 'copy'
                T.nodes[name]['demand'] = T.nodes[a]['demand']
                T.nodes[a]['demand'] = 0
                T.nodes[name]['children'] = []
            for i in T[a]:
                if i != T.nodes[a]['parent']:
                    T.nodes[a]['children'].append(i)
                    T.nodes[i]['parent'] = a
                    B.append(i)
        A = B[:]
        
    ## Now we actually relieve the subtrees
    
    S = [] # Set used to store the subtrees
    
    while True:
    
        # First we update the loads and check if we are done relieving
        
        check = 0
        for i in F1_cus:
            compute_load(T, i)
            if T.nodes[i]['load'] <= U:
                check += 1
        if check == len(F1_cus):
            break
    
        # If we are not done, we proceed to find v' and relieve it
    
        for i in T.nodes():
            ofInterest = True
            if i not in range(ncus):
                continue
            if T.nodes[i]['load'] > U:
                for j in T.nodes[i]['children']:
                    if T.nodes[j]['load'] > U:
                        ofInterest = False
                        break
                if ofInterest:
                    childCopy = T.nodes[i]['children'][:] # "i = v'"
                    while True:
                        for j in childCopy:
                            A_j = [j]
                            sum_j = T.nodes[j]['load']
                            for k in childCopy:
                                if k == j:
                                    continue
                                elif sum_j + T.nodes[k]['load'] <= U:
                                    sum_j += T.nodes[k]['load']
                                    A_j.append(k)
                            if sum_j >= U/2:
                                N = [i] # Every subtree that gets removed from the tree contains node v'. This ensures subtree connectivity
                                for a in A_j:
                                    sub_tree_set(N, T, a)
                                    T.nodes[i]['children'].remove(a)
                                S.append(T.copy().subgraph(N))
                                N.remove(i)
                                T.remove_nodes_from(N)
                            for k in A_j:
                                childCopy.remove(k)
                            break
                        if len(childCopy) == 0:
                            break
                    break
    
    ## Finally, we deal with the groups that are still attached to the tree
    
    if SDG_mod == 0: # Add remaining large demand groups to S
        for i in F1_cus:
            compute_load(T, i)
            if T.nodes[i]['load'] >= U/2:            
                N = []
                sub_tree_set(N, T, i)
                S.append(T.copy().subgraph(N))               
                pre = T.nodes[i]['parent']
                T.nodes[pre]['children'].remove(i)
                T.remove_nodes_from(N)
    else: # Add ALL remaining groups to S, including small ones
        for i in F1_cus:
            compute_load(T, i)
            if T.nodes[i]['load'] > 0:
                
                N = []
                sub_tree_set(N, T, i)
                S.append(T.copy().subgraph(N))
                
                pre = T.nodes[i]['parent']
                T.nodes[pre]['children'].remove(i)
                T.remove_nodes_from(N)
            
    return S


## Define and solve the Assignment Problem 
    
def solve_assign(CFL_mod, set_F1, Assign_mod, S, nloc, loc_array, ncus, cus_array, cusdem_array, intbool, routecost, gurobi_output_print, depcap_array, depcos_array, GAMMA, t_lim, int_lim, U, EPSILON, LS_print):
    
    pre_start = time.time()
    
    improve_time = 60*int_lim # Time limit for improving LS solution
    depcos_array_F1 = depcos_array[:]
    for i in set_F1: depcos_array_F1[i] = 0
    
    # Variables used to save the time required for each step of this function
    ls_time = 0
    mip_time = 0
    ip_time = 0
    ip2_time = 0
    alp_time = 0
    rou_time = 0
    aip_time = 0
    aip2_time = 0
    ############
    
    print('\nAssigning groups of customers to facilities:\n')
    
    infeasible = False
    assign_sw = {} # Array used to assign groups of customers to facilities    
    cTilde = {}
    set_F2 = [] # Set of open facilities    
    
    time_stop_ls = 0
    time_stop_mip = 0
    time_stop_ip = 0
    time_stop_ip2 = 0
    time_stop_alp = 0
    time_stop_aip = 0
    time_stop_aip2 = 0
    
    tourDemand = {}
    x_rel = {} # Array to save the relaxed assignment of the LS
    
    ## The distance between a subtree and a facility is the minimum distance to one of the customers
    
    for s in S:
        for w in range(nloc):
            cTilde[s,w] = [math.inf, math.nan]
            for i in s.nodes():
                if type(i) == str:
                    iNum = int(re.search(r'\d+', i).group())
                else:
                    iNum = i
                dist = (2/U) * (conv_int_cost(math.sqrt(pow(loc_array[w][0] - cus_array[iNum][0],2) + pow(loc_array[w][1] - cus_array[iNum][1],2)), intbool) + 0.5*routecost)                
                if dist <= cTilde[s,w][0]:
                    if dist < cTilde[s,w][0] or str(i) < str(cTilde[s,w][1]): # Avoid possible output variance by defining criteria to tackle draws
                        cTilde[s,w] = [dist,i]
    
    ## Next, we compute the total demand of the subtrees
    
    for s in S:
        tourDemand[s] = 0
        for i in s.nodes():
            tourDemand[s] += s.nodes[i]['demand']
        
    totalDemand = sum([cusdem_array[j] for j in range(ncus)])
        
    if sum(depcap_array) < totalDemand: # If the instance is not feasible, interrupt
        print('WARNING: NO ASSIGNMENT CAN BE FOUND\n')
        infeasible = True
        pre_end = time.time()
        pre_time = pre_end - pre_start
        return infeasible, assign_sw, cTilde, set_F2, pre_time, ls_time, mip_time, ip_time, ip2_time, alp_time, rou_time, aip_time, aip2_time, time_stop_ls, time_stop_mip, time_stop_ip, time_stop_ip2, time_stop_alp, time_stop_aip, time_stop_aip2
    
    pre_end = time.time()
    pre_time = pre_end - pre_start
    
    ## Subtree-Facility assignment
                   
    if CFL_mod == 0: # CFL LS formulation
        
        print('Determining set of facilities via Local Search...')
        
        ### Initial Solution ##################################################
        
        y = [0 for i in range(nloc)]
        fac_dist_dem = [0 for i in range(nloc)]
        for i in range(nloc):
            for s in S:
                fac_dist_dem[i] += cTilde[s,i][0]/tourDemand[s]
        order = sorted(range(len(fac_dist_dem)), key=lambda k: fac_dist_dem[k])
        for fac in order:
            y[fac] = 1
            if sum([depcap_array[i]*y[i] for i in range(nloc)]) >= totalDemand: break
        
        #######################################################################
        
        if LS_print: 
            print('Starting Point:')
            print(y)        
        open_facilities = sum(y)
        
        mod = Model('Assign')
        mod.Params.OutputFlag = gurobi_output_print
        mod.Params.TimeLimit = improve_time
        x = {}
        
        for i in range(nloc):
            for s in S:
                x[s,i] = mod.addVar(lb = 0, ub = tourDemand[s]*y[i], vtype = GRB.CONTINUOUS)
        
        mod.setObjective(quicksum(cTilde[s,i][0]*x[s,i] for i in range(nloc) for s in S), GRB.MINIMIZE)
        
        for i in range(nloc):
            mod.addConstr(quicksum(x[s,i] for s in S) <= depcap_array[i])
        
        for s in S:
            mod.addConstr(quicksum(x[s,i] for i in range(nloc)) >= tourDemand[s])
        
        start_cfl = time.time()
        last_improve = start_cfl
        mod.optimize()
        last_value = mod.getObjective().getValue() + sum([depcos_array_F1[i]*y[i] for i in range(nloc)]) # Value that will be updated to check the stop criterion of the local search
        mod.Params.Cutoff = (1 - EPSILON/(4*nloc))*last_value
        
        if mod.status == 9:
            
            print('ASSIGNMENT LS TIME LIMIT REACHED')
        
        else:
            
            while True:
                
                move_found = False
                
                if 0 < open_facilities < nloc : # Move: swap facility
                    
                    for k in range(nloc):
                        for l in range(nloc):
                            
                            right_now = time.time()
                            if right_now - start_cfl > 60*t_lim or right_now - last_improve > improve_time: # Check if the time limit has been reached
                                print('ASSIGNMENT LS TIME LIMIT REACHED')       
                                time_stop_ls = 1
                                break
                            
                            if y[k] != y[l]:
                                yk0 = y[k]
                                yl0 = y[l]
                                y[k] = yl0
                                y[l] = yk0
                                if sum([depcap_array[i]*y[i] for i in range(nloc)]) < totalDemand: # Check if there's even enough capacity
                                    y[k] = yk0
                                    y[l] = yl0
                                    continue
                                
                                #################### Update LP ####################
                            
                                for s in S:
                                    x[s,k].ub = tourDemand[s]*y[k]
                                    x[s,l].ub = tourDemand[s]*y[l]
                                mod.optimize()
                                
                                ###################################################
                                
                                if mod.status == 9: # Check if optimization was interrupted early
                                    print('ASSIGNMENT LS TIME LIMIT REACHED')       
                                    y[k] = yk0
                                    y[l] = yl0
                                    time_stop_ls = 1
                                    break
                                
                                if mod.status == 3 or mod.status == 6: # Check if the problem is not feasible or if the solution cannot be good enough                                
                                    y[k] = yk0
                                    y[l] = yl0
                                    for s in S:
                                        x[s,k].ub = tourDemand[s]*y[k]
                                        x[s,l].ub = tourDemand[s]*y[l]
                                    continue
                                
                                new_value = mod.getObjective().getValue() + sum([depcos_array_F1[i]*y[i] for i in range(nloc)])
                                    
                                if last_value - new_value > EPSILON*last_value/(4*nloc): # Check the stop criterion of the local search
                                    last_improve = time.time()
                                    move_found = True
                                    last_value = new_value
                                    mod.Params.Cutoff = (1 - EPSILON/(4*nloc))*last_value
                                    break
                                
                                # If the move doesn't improve the solution, the variables are left untouched:                            
                                y[k] = yk0
                                y[l] = yl0
                                for s in S:
                                    x[s,k].ub = tourDemand[s]*y[k]
                                    x[s,l].ub = tourDemand[s]*y[l]
                     
                        if move_found or (time_stop_ls == 1): break
                
                if time_stop_ls == 1: break    
                
                if move_found:
                    if LS_print: 
                        print('\nSwap Facilities:')
                        print(y)
                    continue
                
                if open_facilities < nloc: # Move: add facility
    
                    for k in range(nloc):
                        
                        right_now = time.time()
                        if right_now - start_cfl > 60*t_lim or right_now - last_improve > improve_time: # Check if the time limit has been reached
                            print('ASSIGNMENT LS TIME LIMIT REACHED')        
                            time_stop_ls = 1
                            break
                        
                        if y[k] == 0:
                            y[k] = 1                        
                            if sum([depcap_array[i]*y[i] for i in range(nloc)]) < totalDemand: # Check if there's even enough capacity
                                y[k] = 0
                                continue
                            
                            #################### Update LP ####################
                            
                            for s in S:
                                x[s,k].ub = tourDemand[s]*y[k]                                       
                            mod.optimize()
    
                            ###################################################
                            
                            if mod.status == 9: # Check if optimization was interrupted early
                                print('ASSIGNMENT LS TIME LIMIT REACHED')       
                                y[k] = 0
                                time_stop_ls = 1
                                break
                            
                            if mod.status == 3 or mod.status == 6: # Check if the problem is not feasible or if the solution cannot be good enough
                                y[k] = 0
                                for s in S:
                                    x[s,k].ub = tourDemand[s]*y[k]
                                continue
                            
                            new_value = mod.getObjective().getValue() + sum([depcos_array_F1[i]*y[i] for i in range(nloc)])
                            
                            if last_value - new_value > EPSILON*last_value/(4*nloc): # Check the stop criterion of the local search
                                last_improve = time.time()
                                move_found = True
                                open_facilities += 1
                                last_value = new_value
                                mod.Params.Cutoff = (1 - EPSILON/(4*nloc))*last_value
                                break
                            
                            # If the move doesn't improve the solution, the variable is left untouched:
                            y[k] = 0
                            for s in S:
                                x[s,k].ub = tourDemand[s]*y[k]
                            
                if time_stop_ls == 1: break
                
                if move_found:
                    if LS_print: 
                        print('\nAdd Facility:')
                        print(y)
                    continue
                
                if open_facilities > 1: # Move: drop facility
        
                    for k in range(nloc):
                        
                        right_now = time.time()
                        if right_now - start_cfl > 60*t_lim or right_now - last_improve > improve_time: # Check if the time limit has been reached                        
                            print('ASSIGNMENT LS TIME LIMIT REACHED')
                            time_stop_ls = 1
                            break
                        
                        if y[k] == 1:
                            y[k] = 0                       
                            if sum([depcap_array[i]*y[i] for i in range(nloc)]) < totalDemand: # Check if there's even enough capacity
                                y[k] = 1
                                continue
    
                            #################### Update LP ####################
                            
                            for s in S:
                                x[s,k].ub = tourDemand[s]*y[k]
                            mod.optimize()
                            
                            ###################################################
                            
                            if mod.status == 9: # Check if optimization was interrupted early
                                print('ASSIGNMENT LS TIME LIMIT REACHED')       
                                y[k] = 1
                                time_stop_ls = 1
                                break
                            
                            if mod.status == 3 or mod.status == 6: # Check if the problem is not feasible or if the solution cannot be good enough
                                y[k] = 1
                                for s in S:
                                    x[s,k].ub = tourDemand[s]*y[k]
                                continue
                            
                            new_value = mod.getObjective().getValue() + sum([depcos_array_F1[i]*y[i] for i in range(nloc)])
                            
                            if last_value - new_value > EPSILON*last_value/(4*nloc): # Check the stop criterion of the local search
                                last_improve = time.time()
                                move_found = True
                                open_facilities -= 1
                                last_value = new_value
                                mod.Params.Cutoff = (1 - EPSILON/(4*nloc))*last_value
                                break
                            
                            # If the move doesn't improve the solution, the variable is left untouched:
                            y[k] = 1
                            for s in S:
                                x[s,k].ub = tourDemand[s]*y[k]
                            
                if time_stop_ls == 1: break
                
                if move_found:
                    if LS_print: 
                        print('\nDrop Facility:')
                        print(y)
                    continue
                
                break
        
        for i in range(nloc):
            if y[i] > 0.5:
                set_F2.append(i)
        
        print('\nF_2 = ')
        print(set_F2, '\n')
        
        ls_end = time.time()
        ls_time = ls_end - pre_end
        
    elif CFL_mod == 1: # CFL MIP formulation
        
        print('Solving CFL MIP...')
        
        mod = Model('Assign')
        mod.Params.OutputFlag = gurobi_output_print
        mod.Params.TimeLimit = 60*t_lim
        x, y = {}, {}
        
        for i in range(nloc):
            y[i] = mod.addVar(vtype = 'B')
            for s in S:
                x[s,i] = mod.addVar(lb = 0, ub = 1, vtype = GRB.CONTINUOUS)
        
        mod.setObjective(quicksum(depcos_array_F1[i]*y[i] for i in range(nloc)) + quicksum(cTilde[s,i][0]*x[s,i]*tourDemand[s] for i in range(nloc) for s in S), GRB.MINIMIZE)
        
        for i in range(nloc):
            mod.addConstr(quicksum(x[s,i]*tourDemand[s] for s in S) <= depcap_array[i])
        
        for s in S:
            mod.addConstr(quicksum(x[s,i] for i in range(nloc)) >= 1)
            
        for i in range(nloc):
            for s in S:
                mod.addConstr(x[s,i] <= y[i])
        
        mod.optimize()
                
        if mod.status == 9:
            print('CFL MIP TIME LIMIT REACHED')
            time_stop_mip = 1
        
        if mod.SolCount == 0: # Check if no solution was found
            print('WARNING: NO SOLUTION TO THE CFL MIP WAS FOUND')
            infeasible = True
            mip_end = time.time()
            mip_time = mip_end - pre_end
            return infeasible, assign_sw, cTilde, set_F2, pre_time, ls_time, mip_time, ip_time, ip2_time, alp_time, rou_time, aip_time, aip2_time, time_stop_ls, time_stop_mip, time_stop_ip, time_stop_ip2, time_stop_alp, time_stop_aip, time_stop_aip2
        
        for i in range(nloc):
            if y[i].X > 0.5:
                set_F2.append(i)
        
        print('\nF_2 = ')
        print(set_F2, '\n')
        
        mip_end = time.time()
        mip_time = mip_end - pre_end
        
    if CFL_mod != 2:
        
        if Assign_mod == 0:
            
            print('Actually assigning clusters to open facilities (LP + rounding)...')
            
            modA = Model('Assign')
            modA.Params.OutputFlag = gurobi_output_print
            modA.Params.TimeLimit = 60*t_lim
            x = {}
            
            for w in set_F2:
                for s in S:
                    x[s,w] = modA.addVar(lb = 0, ub = tourDemand[s], vtype = GRB.CONTINUOUS)
            
            modA.setObjective(quicksum(cTilde[s,w][0]*x[s,w]/tourDemand[s] for w in set_F2 for s in S), GRB.MINIMIZE)
            
            for w in set_F2:
                modA.addConstr(quicksum(x[s,w] for s in S) <= depcap_array[w])
            
            for s in S:
                modA.addConstr(quicksum(x[s,w] for w in set_F2) >= tourDemand[s])
            
            modA.optimize()
            
            if modA.status == 9:
                print('ASIGNMENT LP TIME LIMIT REACHED')
                time_stop_alp = 1
            
            if modA.SolCount == 0: # Check if no solution was found
                print('WARNING: NO SOLUTION TO THE ASSIGNMENT LP WAS FOUND')
                infeasible = True
                alp_end = time.time()
                alp_time = alp_end - ls_end if CFL_mod == 0 else alp_end - mip_end
                return infeasible, assign_sw, cTilde, set_F2, pre_time, ls_time, mip_time, ip_time, ip2_time, alp_time, rou_time, aip_time, aip2_time, time_stop_ls, time_stop_mip, time_stop_ip, time_stop_ip2, time_stop_alp, time_stop_aip, time_stop_aip2
            
            alp_end = time.time()
            alp_time = alp_end - ls_end if CFL_mod == 0 else alp_end - mip_end
            
            new_round = True # Decide if the new or the old rounding procedure should be applied
            
            ## New rounding Procedure ########################################
                      
            if new_round:
                
                ## Creation of helper graph, with flow corresponding to the Assignment LP's solution
                
                for w in set_F2:
                    for s in S:
                        x_rel[s,w] = x[s,w].X
                        
                Gx = nx.Graph()
                for w in set_F2:
                    Gx.add_node(w)
                for s in S:
                    Gx.add_node(s)                    
                    for w in set_F2:
                        if 0 < x_rel[s,w] < tourDemand[s]:
                            Gx.add_edge(s,w)
                
                while True:
                            
                    if len(Gx.edges()) == 0: break # The procedure ends once all edges have been deleted
                    
                    cComp0 = nx.connected_components(Gx) # Find connected components 
                    cComp = [Gx.subgraph(comp).copy() for comp in cComp0] # Set of induced subgraphs of each component
                    
                    tbr = [] # Nodes to be removed
                    for comp in cComp:
                        
                        if len(comp.edges()) == 0:
                            tbr.append(list(comp.nodes())[0]) # Add isolated nodes to the "to be removed" list
                            continue
                        
                        leaves = [] # Array used to save 2 leaves of the subtree
                        for v in list(comp.nodes()):
                            if comp.degree(v) == 1: leaves.append(v)                                
                            if len(leaves) == 2: break
                        
                        path = list(nx.all_simple_paths(comp, leaves[0], leaves[1]))[0] # Path between the 2 leaves
                        
                        path_cost = 0 # Up next, we will compute the per-unit cost of the path
                        for v in range(len(path)-1):
                            if type(path[v]) == int: path_cost += cTilde[path[v+1],path[v]][0] # Current node is facility
                            else: path_cost -= cTilde[path[v],path[v+1]][0] # Current node is cluster
                            
                        path_dir = 1 if path_cost <= 0 else -1 # We define in which direction the flow will be sent
                        
                        bnc = math.inf # bottleneck capacity
                        for v in range(len(path)-1):
                            v_ind = v if path_dir == 1 else len(path) - v - 1 # Determine the index in terms of the direction
                            if type(path[v_ind]) == int: capacity = tourDemand[path[v_ind + path_dir]] - x_rel[path[v_ind + path_dir],path[v_ind]] # Current node is facility                                
                            else: capacity = x_rel[path[v_ind],path[v_ind + path_dir]] # Current node is cluster
                            if capacity < bnc: bnc = capacity
                            
                        for v in range(len(path)-1):
                            v_ind = v if path_dir == 1 else len(path) - v - 1 # Determine the index in terms of the direction
                            if type(path[v_ind]) == int: # Current node is facility: add flow
                                x_rel[path[v_ind + path_dir],path[v_ind]] = x_rel[path[v_ind + path_dir],path[v_ind]] + bnc
                                if x_rel[path[v_ind + path_dir],path[v_ind]] == tourDemand[path[v_ind + path_dir]]: Gx.remove_edge(path[v_ind + path_dir],path[v_ind])
                            else: # Current node is cluster: remove flow
                                x_rel[path[v_ind],path[v_ind + path_dir]] = x_rel[path[v_ind],path[v_ind + path_dir]] - bnc
                                if x_rel[path[v_ind],path[v_ind + path_dir]] == 0: Gx.remove_edge(path[v_ind],path[v_ind + path_dir])
                            
                    Gx.remove_nodes_from(tbr) # Remove isolated nodes
                    
                for w in set_F2:
                    for s in S:
                        assign_sw[s,w] = 1 if x_rel[s,w] > tourDemand[s]/2 else 0 # Final assignment of clusters to facilities

            ## New rounding procedure: End ###################################
            
            ## Old rounding procedure ########################################
            
            if not new_round:
            
                ## Creation of helper graph
                
                for w in set_F2:
                    for s in S:
                        x_rel[s,w] = x[s,w].X/tourDemand[s]
                
                G_bar = nx.Graph()
                
                for w in set_F2:
                    G_bar.add_node(w)
                for s in S:
                    G_bar.add_node(s)
                    for w in set_F2:
                        if x_rel[s,w] > 0:
                            G_bar.add_edge(s,w)
                            
                ## Identify and direct pseudo-trees; assign according to gamma criterion
                
                for s in S:
                    for w in set_F2:
                        assign_sw[s,w] = 0
                
                cComp0 = nx.connected_components(G_bar) # Find connected components 
                cComp = [G_bar.subgraph(comp).copy() for comp in cComp0] # Set of induced subgraphs of each component
                
                for comp in cComp:
                    
                    try: # We check if the connected component has got a cycle
                        
                        cycle = nx.find_cycle(comp)
                        
                        compTree0 = nx.dfs_tree(comp,cycle[0][0]) # We direct the edges of the tree
                        compTree = compTree0.reverse() # We point the edges towards the root
                        
                        # Next, we make sure all edges in the cycle are pointing in the same direction:
                        
                        for c in cycle:
                            if c in compTree.edges():
                                continue
                            elif (c[1],c[0]) in compTree.edges():
                                compTree.remove_edge(c[1],c[0])
                                compTree.add_edge(c[0],c[1])
                            else:
                                compTree.add_edge(c[0],c[1])
                        
                    except nx.NetworkXNoCycle: # If the component doesn't have a cycle:
                        
                        for v in comp.nodes():
                            if type(v) == int:
                                root = v # We define any facility as the root of the directed tree
                                break
                        
                        compTree0 = nx.dfs_tree(comp,root) # We direct the edges of the tree 
                        compTree = compTree0.reverse() # We point the edges towards the root
                
                    # Now we define every customer group's facility:
                    
                    for s in compTree.nodes():
                        
                        if type(s) == int:
                            continue
                        
                        w_suc = list(compTree.successors(s))[0]
                        pre_list = list(compTree.predecessors(s))
                        
                        if ((GAMMA == 1) and (x_rel[s,w_suc] >= GAMMA)) or (x_rel[s,w_suc] > GAMMA):
                            min_c_fac = w_suc
                            min_c = cTilde[s,w_suc][0]
                            for w in pre_list:
                                if cTilde[s,w][0] < min_c:
                                    min_c = cTilde[s,w][0]
                                    min_c_fac = w
                        else:
                            min_c_fac = math.nan
                            min_c = math.inf
                            for w in pre_list:
                                if cTilde[s,w][0] < min_c:
                                    min_c = cTilde[s,w][0]
                                    min_c_fac = w
                                    
                        assign_sw[s,min_c_fac] = 1
        
            ## Old rounding procedure: END ###################################        
        
            rou_end = time.time()
            rou_time = rou_end - alp_end
        
        else: # Assignment IP
            
            print('Actually assigning clusters to open facilities (IP approach)...')
            
            modA = Model('Assign')
            modA.Params.OutputFlag = gurobi_output_print
            modA.Params.TimeLimit = 60*t_lim
            
            x = {}
            for w in set_F2:
                for s in S:
                    x[s,w] = modA.addVar(vtype = 'B')
            
            modA.setObjective(quicksum(cTilde[s,w][0]*x[s,w] for w in set_F2 for s in S), GRB.MINIMIZE)
            
            for w in set_F2:
                modA.addConstr(quicksum(x[s,w]*tourDemand[s] for s in S) <= depcap_array[w])
            
            for s in S:
                modA.addConstr(quicksum(x[s,w] for w in set_F2) >= 1)
            
            modA.optimize()
            
            if modA.status == 9:
                print('ASIGNMENT IP TIME LIMIT REACHED')
                time_stop_aip = 1
            
            if modA.status == 3 or modA.SolCount == 0: # If the problem is infeasible or couldn't be solved, it will be tackled again, but allowing depot capacity expansion
                
                aip_end = time.time()
                aip_time = aip_end - ls_end if CFL_mod == 0 else aip_end - mip_end
            
                print('ASSIGNMENT IP IS INFEASIBLE OR COULD NOT BE SOLVED. DEPOT CAPACITIES WILL BE EXPANDED...')
                
                modB = Model('Assign2')
                modB.Params.OutputFlag = gurobi_output_print
                modB.Params.TimeLimit = 60*t_lim
                
                x = {}
                for w in set_F2:
                    for s in S:
                        x[s,w] = modB.addVar(vtype = 'B')
                K = modB.addVar(vtype = 'C', lb = 0)
                
                modB.setObjectiveN(K, index = 0, priority = 1)
                modB.setObjectiveN(quicksum(cTilde[s,w][0]*x[s,w] for s in S for w in set_F2), index = 1, priority = 0)
               
                for w in set_F2:
                    modB.addConstr(quicksum(x[s,w]*tourDemand[s] for s in S) <= depcap_array[w]*(1 + K))
                
                for s in S:
                    modB.addConstr(quicksum(x[s,w] for w in set_F2) >= 1)
                
                modB.optimize()
                
                if modB.status == 9:
                    print('ASIGNMENT IP (v2) TIME LIMIT REACHED')
                    time_stop_aip2 = 1
                    
                if modB.SolCount == 0: # Check if no solution was found
                    print('WARNING: NO SOLUTION TO THE ASSIGNMENT IP (v2) WAS FOUND\n')
                    infeasible = True
                    aip2_end = time.time()
                    aip2_time = aip2_end - aip_end
                    return infeasible, assign_sw, cTilde, set_F2, pre_time, ls_time, mip_time, ip_time, ip2_time, alp_time, rou_time, aip_time, aip2_time, time_stop_ls, time_stop_mip, time_stop_ip, time_stop_ip2, time_stop_alp, time_stop_aip, time_stop_aip2
            
            for s in S:
                for w in set_F2:
                    assign_sw[s,w] = x[s,w].X
            
            if modA.status != 3 and modA.SolCount != 0:
                aip_end = time.time()
                aip_time = aip_end - ls_end if CFL_mod == 0 else aip_end - mip_end
            else:
                aip2_end = time.time()
                aip2_time = aip2_end - aip_end
        
    else: # IP formulation for both CFL and Assignment
        
        print('Solving the CFL-Assignment IP...')
        
        mod2 = Model('Assign')
        mod2.Params.OutputFlag = gurobi_output_print
        mod2.Params.TimeLimit = 60*t_lim
        x, y = {}, {}
        
        for w in range(nloc):
            y[w] = mod2.addVar(vtype = 'B')
            for s in S:
                x[s,w] = mod2.addVar(vtype = 'B')
        
        mod2.setObjective(quicksum(depcos_array[w]*y[w] for w in range(nloc)) + U*quicksum(cTilde[s,w][0]*x[s,w] for s in S for w in range(nloc)), GRB.MINIMIZE)
        for w in range(nloc):
            mod2.addConstr(quicksum(x[s,w]*tourDemand[s] for s in S) <= depcap_array[w]*y[w])
        
        for s in S:
            mod2.addConstr(quicksum(x[s,w] for w in range(nloc)) >= 1)
        
        mod2._intlim = 60*int_lim
        mod2._timereset = time.time()
        mod2.optimize(interruptcallback)
        
        if mod2.status == 9 or mod2.status == 11:
            print('CFL-ASSIGNMENT IP TIME LIMIT REACHED')
            time_stop_ip = 1
        
        if mod2.status == 3 or mod2.SolCount == 0: # If the problem is infeasible or couldn't be solved, it will be tackled again, but allowing depot capacity expansion
            
            ip_end = time.time()
            ip_time = ip_end - pre_end
        
            print('CFL-ASSIGNMENT IP IS INFEASIBLE OR COULD NOT BE SOLVED. DEPOT CAPACITIES WILL BE EXPANDED...')
            
            mod2b = Model('Assign2')
            mod2b.Params.OutputFlag = gurobi_output_print
            mod2b.Params.TimeLimit = 60*t_lim
            
            x, y = {}, {}
            for w in range(nloc):
                y[w] = mod2b.addVar(vtype = 'B')
                for s in S:
                    x[s,w] = mod2b.addVar(vtype = 'B')            
            K = mod2b.addVar(vtype = 'C', lb = 0)
            
            mod2b.setObjectiveN(K, index = 0, priority = 1)
            mod2b.setObjectiveN(quicksum(depcos_array[w]*y[w] for w in range(nloc)) + U*quicksum(cTilde[s,w][0]*x[s,w] for s in S for w in range(nloc)), index = 1, priority = 0)
            
            for w in range(nloc):
                mod2b.addConstr(quicksum(x[s,w]*tourDemand[s] for s in S) <= depcap_array[w]*(1 + K))
                for s in S:
                    mod2b.addConstr(x[s,w] <= y[w])
            
            for s in S:
                mod2b.addConstr(quicksum(x[s,w] for w in range(nloc)) >= 1)
            
            mod2b._intlim = 60*int_lim
            mod2b._timereset = time.time()
            mod2b.optimize(interruptcallback)
            
            if mod2b.status == 9 or mod2b.status == 11:
                print('CFL-ASSIGNMENT IP (v2) TIME LIMIT REACHED')
                time_stop_ip2 = 1
                
            if mod2b.SolCount == 0: # Check if no solution was found
                print('WARNING: NO SOLUTION TO THE CFL-ASSIGNMENT IP (v2) WAS FOUND\n')
                infeasible = True
                ip2_end = time.time()
                ip2_time = ip2_end - ip_end
                return infeasible, assign_sw, cTilde, set_F2, pre_time, ls_time, mip_time, ip_time, ip2_time, alp_time, rou_time, aip_time, aip2_time, time_stop_ls, time_stop_mip, time_stop_ip, time_stop_ip2, time_stop_alp, time_stop_aip, time_stop_aip2
        
        for w in range(nloc):
            if y[w].X > 0.5:
                set_F2.append(w)
            for s in S:
                assign_sw[s,w] = x[s,w].X
        
        if mod2.status != 3 and mod2.SolCount != 0:
            ip_end = time.time()
            ip_time = ip_end - pre_end
        else:
            ip2_end = time.time()
            ip2_time = ip2_end - ip_end
                
    return infeasible, assign_sw, cTilde, set_F2, pre_time, ls_time, mip_time, ip_time, ip2_time, alp_time, rou_time, aip_time, aip2_time, time_stop_ls, time_stop_mip, time_stop_ip, time_stop_ip2, time_stop_alp, time_stop_aip, time_stop_aip2


## Turn subtrees into tours
    
def subtrees_into_tours(nloc, set_F2, S, assign_sw, cTilde, set_F1, T, SDG_mod, Assign_mod, loc_array, cus_array, intbool, print_sol):
    
    print('\nTurning subtrees into tours...\n')

    facility_demand = np.zeros(nloc)
    tour_array = []
    assigned_cluster = {}
    for s in S: assigned_cluster[s] = 0
    
    for w in set_F2:
        if print_sol: print('facility ' + str(w))
        for s in S:
            if assigned_cluster[s] == 1:
                continue
            elif assign_sw[s,w] > 0.5:
                assigned_cluster[s] = 1
                s_copy = nx.Graph(s)
                s_copy.add_node('facility ' + str(w))
                s_copy.add_edge(cTilde[s,w][1],'facility ' + str(w))
                s_copy = s_copy.to_directed() # Doubling edges
                
                ## Find eulerian circuit
                
                V_s0 = list(s_copy.nodes())
                E_s0 = list(s_copy.edges())
                
                # Sort arrays in order to avoid varying output ###############
                
                V_s0_num = []
                V_s0_let = []
                for v in V_s0:
                    if type(v) == int: 
                        V_s0_num.append(v)
                    else:
                        V_s0_let.append(v)                
                V_s =  sorted(V_s0_let, reverse = True) + sorted(V_s0_num)
                
                E_s = []
                E_s0_enc = {}
                Codes0 = []
                for (i,j) in E_s0:                    
                    code = ''                                        
                    if type(i) == int:
                        code = code + str(i) + 'n'
                    elif '-' in i:
                        code = code + i[0:i.find('-')] + 'c'
                    else:
                        code = code + i[i.find(' ')+1:] + 'f'                                            
                    if type(j) == int:
                        code = code + str(j) + 'n'
                    elif '-' in j:
                        code = code + j[0:j.find('-')] + 'c'
                    else:
                        code = code + j[j.find(' ')+1:] + 'f'
                    Codes0.append(code)
                    E_s0_enc[code] = (i,j)
                    Codes = sorted(Codes0)
                for code in Codes:
                    E_s.append(E_s0_enc[code])

                ##############################################################
                
                visits = {}
                for v in V_s: visits[v] = 0
                e_circ = []
                location = V_s[0]
                
                while len(E_s) > 0:
                    e_circ.append(location)
                    visits[location] += 1
                    
                    candidates = []
                    for (i,j) in E_s:
                        if i == location:
                            candidates.append((i,j))
                    
                    e_next = candidates[0] 
                    for (i,j) in candidates:
                        if visits[j] < visits[e_next[1]]:
                            e_next = (i,j)
                     
                    E_s.remove(e_next)
                    location = e_next[1]
                    
                ## Shortcutting
    
                tour = []
                for j in e_circ:
                    if j not in tour:
                        if j == 'facility ' + str(w):
                            tour.append('facility ' + str(w))
                        elif s_copy.nodes[j]['demand'] > 0:
                            tour.append(j)
                            facility_demand[w] += s.nodes[j]['demand']
    
                ## Remove '-copy' tag
    
                for j in range(len(tour)):
                    if type(tour[j]) == str:
                        jStr = ''.join([i for i in tour[j] if not i.isdigit()])
                        if jStr == '-copy':
                            tour[j] = int(re.search(r'\d+', tour[j]).group())
    
                ## Place facility at the beginning of the tour
    
                while tour[0] != 'facility ' + str(w):
                    jMove = tour[0]
                    tour.remove(jMove)
                    tour.append(jMove)
    
                tour_array.append(tour)
    
                if print_sol: print(tour)       
                
        if print_sol: print('')
    
    if SDG_mod == 0:
    
        print('Potential additional tours...\n') # The ones that weren't removed as subtrees from the original MST
        
        for w in set_F1:
            if len(T.nodes['facility ' + str(w)]['children']) > 0:
                load_w = 0
                for j in T.nodes['facility ' + str(w)]['children']:
                    load_w += compute_load(T, j, True)
                if load_w > 0:
                    print('facility ' + str(w))
                    N = []
                    sub_tree_set(N, T, 'facility ' + str(w))
                    subG = T.copy().subgraph(N)
                    subG = subG.to_directed()                 
                    
                    ## Find eulerian circuit
                
                    V_s0 = list(subG.nodes())
                    E_s0 = list(subG.edges())
                    
                    # Sort arrays in order to avoid varying output ###########
                
                    V_s0_num = []
                    V_s0_let = []
                    for v in V_s0:
                        if type(v) == int: 
                            V_s0_num.append(v)
                        else:
                            V_s0_let.append(v)                
                    V_s =  sorted(V_s0_let, reverse = True) + sorted(V_s0_num)
                    
                    E_s = []
                    E_s0_enc = {}
                    Codes0 = []
                    for (i,j) in E_s0:                    
                        code = ''                                        
                        if type(i) == int:
                            code = code + str(i) + 'n'
                        elif '-' in i:
                            code = code + i[0:i.find('-')] + 'c'
                        else:
                            code = code + i[i.find(' ')+1:] + 'f'                                            
                        if type(j) == int:
                            code = code + str(j) + 'n'
                        elif '-' in j:
                            code = code + j[0:j.find('-')] + 'c'
                        else:
                            code = code + j[j.find(' ')+1:] + 'f'
                        Codes0.append(code)
                        E_s0_enc[code] = (i,j)
                        Codes = sorted(Codes0)
                    for code in Codes:
                        E_s.append(E_s0_enc[code])
                    
                    ##########################################################
                    
                    visits = {}
                    for v in V_s: visits[v] = 0
                    e_circ = []
                    location = V_s[0]
                    
                    while len(E_s) > 0:
                        e_circ.append(location)
                        visits[location] += 1
                        
                        candidates = []
                        for (i,j) in E_s:
                            if i == location:
                                candidates.append((i,j))
                        
                        e_next = candidates[0] 
                        for (i,j) in candidates:
                            if visits[j] < visits[e_next[1]]:
                                e_next = (i,j)
                         
                        E_s.remove(e_next)
                        location = e_next[1]
                    
                    tour = []
                    for j in e_circ:
                        if j not in tour:
                            if j == 'facility ' + str(w):
                                tour.append('facility ' + str(w))
                            elif subG.nodes[j]['demand'] > 0:
                                tour.append(j)
                                facility_demand[w] += T.nodes[j]['demand']
        
                    ## Remove '-copy' tag
        
                    for j in range(len(tour)):
                        if type(tour[j]) == str:
                            jStr = ''.join([i for i in tour[j] if not i.isdigit()])
                            if jStr == '-copy':
                                tour[j] = int(re.search(r'\d+', tour[j]).group())
        
                    ## Place facility at the beginning of the tour
        
                    while tour[0] != 'facility ' + str(w):
                        jMove = tour[0]
                        tour.remove(jMove)
                        tour.append(jMove)
        
                    tour_array.append(tour)
                    print(tour)
                    
                    print('')
    
    # Last step: Improve tours applying LKH
    print('Alternatively determining tours through LKH...\n')
    tour_array_lkh = tour_array[:]
    for t in range(len(tour_array_lkh)):
        size = len(tour_array_lkh[t])
        if intbool == 0:
            M = np.zeros((size, size), dtype = int)
        else:
            M = np.zeros((size, size))
        for j in range(1,size):
            fac_index = int(re.search(r'\d+', tour_array_lkh[t][0]).group())
            M[0,j] = conv_int_cost(math.sqrt(pow(loc_array[fac_index][0] - cus_array[tour_array_lkh[t][j]][0],2) + pow(loc_array[fac_index][1] - cus_array[tour_array_lkh[t][j]][1],2)), intbool)
            M[j,0] = M [0,j]
        for i in range(1,size-1):
            for j in range(i+1,size):
                M[i,j] = conv_int_cost(math.sqrt(pow(cus_array[tour_array_lkh[t][i]][0] - cus_array[tour_array_lkh[t][j]][0],2) + pow(cus_array[tour_array_lkh[t][i]][1] - cus_array[tour_array_lkh[t][j]][1],2)), intbool)
                M[j,i] = M[i,j]
        if intbool == 0:
            lkh_output = elkai.solve_int_matrix(M)
        else:
            lkh_output = elkai.solve_float_matrix(M)
        temp_tour = [tour_array_lkh[t][i] for i in lkh_output]
        tour_array_lkh[t] = temp_tour 
            
    return facility_demand, tour_array, tour_array_lkh
       

## Check capacity constraints
    
def check_capacity(nloc, depcap_array, facility_demand):
            
    max_exc_cap_fac = 0 # facility with the max excess/capacity
    max_exc_cap = -math.inf # max excess/capacity
    load_cap = [] # load/capacity for every depot
    
    space = False
    for w in range(nloc):
        load_cap.append(facility_demand[w]/depcap_array[w])
        dif = depcap_array[w] - facility_demand[w]
        if dif < 0:
            space = True
            print('WARNING: FACILITY ' + str(w) + '\'s CAPACITY EXCEEDED BY ' + str(-dif) + ' (' + str(round(-100*dif/depcap_array[w],2)) + '%)')
        if -dif/depcap_array[w] > max_exc_cap:
            max_exc_cap_fac = w
            max_exc_cap = -dif/depcap_array[w]
    
    if space:
        print('')
        
    return max_exc_cap, max_exc_cap_fac, load_cap
        
        
## Compute costs
        
def compute_costs(nloc, facility_demand, depcos_array, tour_array, tour_array_lkh, cus_array, intbool, routecost, loc_array):
                
    print('FINAL SOLUTION DETERMINED.\n\nCosts:\n')

    facilityCost = 0
    num_open_fac = 0
    for w in range(nloc):
        if facility_demand[w] > 0:
            num_open_fac += 1
            facilityCost += depcos_array[w]
    
    print('Facility Costs: ' + str(facilityCost))
    
    tourCost = 0
    for t in tour_array:
        for j in range(1,len(t)-1):
            tourCost += conv_int_cost(math.sqrt(pow(cus_array[t[j]][0] - cus_array[t[j+1]][0],2) + pow(cus_array[t[j]][1] - cus_array[t[j+1]][1],2)), intbool)
        facNum = int(re.search(r'\d+', t[0]).group())
        tourCost += conv_int_cost(math.sqrt(pow(loc_array[facNum][0] - cus_array[t[1]][0],2) + pow(loc_array[facNum][1] - cus_array[t[1]][1],2)), intbool) + 0.5*routecost
        tourCost += conv_int_cost(math.sqrt(pow(loc_array[facNum][0] - cus_array[t[-1]][0],2) + pow(loc_array[facNum][1] - cus_array[t[-1]][1],2)), intbool) + 0.5*routecost
    
    tourCost_lkh = 0
    for t in tour_array_lkh:
        for j in range(1,len(t)-1):
            tourCost_lkh += conv_int_cost(math.sqrt(pow(cus_array[t[j]][0] - cus_array[t[j+1]][0],2) + pow(cus_array[t[j]][1] - cus_array[t[j+1]][1],2)), intbool)
        facNum = int(re.search(r'\d+', t[0]).group())
        tourCost_lkh += conv_int_cost(math.sqrt(pow(loc_array[facNum][0] - cus_array[t[1]][0],2) + pow(loc_array[facNum][1] - cus_array[t[1]][1],2)), intbool) + 0.5*routecost
        tourCost_lkh += conv_int_cost(math.sqrt(pow(loc_array[facNum][0] - cus_array[t[-1]][0],2) + pow(loc_array[facNum][1] - cus_array[t[-1]][1],2)), intbool) + 0.5*routecost
    
    print('Routing Costs: ' + str(tourCost))
    print('Routing Costs (LKH): ' + str(tourCost_lkh))
    print('\nTOTAL COSTS: ' + str(facilityCost + tourCost))
    print('TOTAL COSTS (LKH): ' + str(facilityCost + tourCost_lkh))
    
    return num_open_fac, facilityCost, tourCost, tourCost_lkh