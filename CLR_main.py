"""
Last update: Fri Aug 12 16:36 2022

@author: Oscar F. Carrasco Heine

Execution example: 

python CLR_main.py SchneiderData/400-25-1e 1 0 1 0.1 0 1

- SchneiderData/400-25-1e: data file's location/name, without .dat extension
- 1: Small-demand groups are considered in the assignment procedure (0 if not)
- 0: CFL is tackled via Local Search
     (1 if it is solved as an MIP;
      2 if it is solved as an IP, including Assignment)
- 1: Assignment IP (0 if Assignment LP + rounding); only relevant if CFL_mod < 2
- 0.1: Value of parameter epsilon (Local search)
- 0: No Gurobi output (enter 1 otherwise)
- 1: Print the tours corresponding to the solution (0 if not)
"""

## INITIALIZING ###############################################################

import time
start1 = time.process_time()
start2 = time.time()

# Additional parameters:
commandline_input = True # Execution through command prompt (True) vs. IDE (False)
t_lim = 360 # (M)IP/LP/LS total time limit (in minutes)
int_lim = 120 # Time limit for improving LS solution (in minutes)
GAMMA = 0.5 # Parameter that was only relevant for a previous version of the algorithm
verify = True # True: Execute a short routine that double-checks if the solution is feasible
LS_print = False # True: Show the progression of the local search procedure.
save_sol = False # True: Textfiles contaning the final solutions (i.e., tours) are created

import CLR_functions as fun
import sys

## Parameters

if commandline_input:
    file = sys.argv[1] + '.dat'
    SDG_mod = int(sys.argv[2])
    CFL_mod = int(sys.argv[3])
    Assign_mod = int(sys.argv[4])        
    EPSILON = float(sys.argv[5])
    gurobi_output_print = int(sys.argv[6])
    print_tours = int(sys.argv[7])
else:
    file = 'SchneiderData/400-25-1e.dat'
    SDG_mod = 1
    CFL_mod = 2
    Assign_mod = 0
    EPSILON = 0.1
    gurobi_output_print = 0
    print_tours = 1

# Read input file to obtain instance parameters

ncus, nloc, loc_array, cus_array, U, depcap_array, cusdem_array, depcos_array, intbool, routecost = fun.get_parameters(file) 
ini_par_end = time.time()
ini_par_time = ini_par_end - start2

## PROCEDURE ##################################################################

# FIRST STEP: MST
T, set_F1, LB_MST = fun.solve_mst(nloc, ncus, loc_array, cus_array, intbool, routecost, depcos_array)
mst_end = time.time()
mst_time = mst_end - ini_par_end

# SECOND STEP: RELIEVING OVERLOADED SUBTREES
S = fun.relieve(T, nloc, ncus, cusdem_array, set_F1, U, SDG_mod)
rel_end = time.time()
rel_time = rel_end - mst_end

# THIRD STEP: ASSIGN GROUPS OF CUSTOMERS TO FACILITIES
infeasible, assign_sw, cTilde, set_F2, pre_time, ls_time, mip_time, ip_time, ip2_time, alp_time, rou_time, aip_time, aip2_time, time_stop_ls, time_stop_mip, time_stop_ip, time_stop_ip2, time_stop_alp, time_stop_aip, time_stop_aip2 = fun.solve_assign(CFL_mod, set_F1, Assign_mod, S, nloc, loc_array, ncus, cus_array, cusdem_array, intbool, routecost, gurobi_output_print, depcap_array, depcos_array, GAMMA, t_lim, int_lim, U, EPSILON, LS_print) 
ass_end = time.time()
ass_time = ass_end - rel_end
if infeasible:
    sys.exit('Procedure finished: no solution found')

# FOURTH STEP: TURN SUBTREES INTO TOURS
facility_demand, tour_array, tour_array_lkh = fun.subtrees_into_tours(nloc, set_F2, S, assign_sw, cTilde, set_F1, T, SDG_mod, Assign_mod, loc_array, cus_array, intbool, False)              
tou_end = time.time()
tou_time = tou_end - ass_end

## FINAL STEPS ################################################################

max_exc_cap, max_exc_cap_fac, load_cap = fun.check_capacity(nloc, depcap_array, facility_demand) # Check capacity constraints   
num_open_fac, facilityCost, tourCost, tourCost_lkh = fun.compute_costs(nloc, facility_demand, depcos_array, tour_array, tour_array_lkh, cus_array, intbool, routecost, loc_array) # Print total cost

if save_sol:
    textfile_dts = open(file[(file.index('/') + 1):-4] + "_dts_" + str(CFL_mod) + ".txt", "w")
    for element in tour_array:
        textfile_dts.write(str(element) + "\n")
    textfile_dts.close()    
    textfile_lkh = open(file[(file.index('/') + 1):-4] + "_lkh_" + str(CFL_mod) + ".txt", "w")
    for element in tour_array_lkh:
        textfile_lkh.write(str(element) + "\n")
    textfile_lkh.close()

pos_end = time.time()
pos_time = pos_end - tou_end

## Total time

elapsed_time = time.time() - start2

print('\nTotal processing time [s]: ' + str(round(time.process_time() - start1, 3)))
print('Actual elapsed time [s]: ' + str(round(elapsed_time, 3)))

## Print tours

if print_tours:
    print("\nOriginal Solution:")
    for tour in tour_array:
        print(tour)
    print("\nSolution after LKH:")
    for tour in tour_array_lkh:
        print(tour)

## Result summary

# print('\nResult row: ' + file[(file.index('/') + 1):-4] + ' ' + str(SDG_mod) + ' ' + str(Assign_mod) + ' ' + str(LB_MST) + ' ' + str(num_open_fac) + ' ' + str(facilityCost) + ' ' + str(len(tour_array)) + ' ' + str(tourCost) + ' ' + str(tourCost_lkh) + ' ' + str(max_exc_cap) + ' ' + str(ini_par_time) + ' ' + str(mst_time) + ' ' + str(rel_time) + ' ' + str(pre_time) + ' ' + str(ls_time) + ' ' + str(mip_time) + ' ' + str(ip_time) + ' ' + str(ip2_time) + ' ' + str(alp_time) + ' ' + str(rou_time) + ' ' + str(aip_time) + ' ' + str(aip2_time) + ' ' + str(ass_time) + ' ' + str(tou_time) + ' ' + str(pos_time) + ' ' + str(elapsed_time)  + ' ' + str(time_stop_ls) + ' ' + str(time_stop_mip) + ' ' + str(time_stop_ip) + ' ' + str(time_stop_ip2) + ' ' + str(time_stop_alp) + ' ' + str(time_stop_aip) + ' ' + str(time_stop_aip2))

## OPTIONAL: verify if the solution is really feasible

if verify:
    print("\nOptional: double-checking results...")
    customers = []
    utilization = {}
    for t in tour_array:
        utilization[t[0]] = 0
        for v in t:
            if type(v) != str: customers.append(v)
    if len(customers) != ncus: print("ERROR: number of assigned customers is not correct.")
    customers.sort()
    for c in range(ncus):
        app = customers.count(c)
        if app != 1: print("ERROR: customer " + str(c) + " was included " + str(app) + " times in the solution.")    
    for t in tour_array:
        for c in t[1:]:
            utilization[t[0]] = utilization[t[0]] + cusdem_array[c]    
    import re     
    for facility in utilization:
        fac_num = int(re.findall("\d+", facility)[0])
        if utilization[facility] > depcap_array[fac_num]: print("WARNING: facility " + str(fac_num) + "'s capacity was exceeded.")
    print("Double check finished.")
    
print("\nThe end.")