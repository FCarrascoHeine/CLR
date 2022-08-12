I) INTRODUCTION

This code ('CLR_main.py' and 'CLR_functions.py') was designed to solve instances of the Capacitated Location Routing problem (CLR). 

Besides Python, a Gurobi license is required. 

The code was created by Oscar F. Carrasco Heine, in order to perform the computational study of the following paper (Carrasco Heine, Demleitner, and Matuschke, 2022): https://doi.org/10.1016/j.ejor.2022.04.028

Details about the origin of the instances contained in the folders 'BenchmarkData', 'SchneiderData', and 'OriginalData' can be found in said paper. Neither the instances in 'BenchmarkData', nor in 'SchneiderData' were created by the author of the code, or by any other of the paper's authors.

--------------------
II) USING THE CODE

'CLR_main.py' allows solving any single instance within the folders 'BenchmarkData', 'SchneiderData', and 'OriginalData'. To execute the code, there are two possibilities:

II.1) Command line

Parameter 'commandline_input' (line 28) needs to be set to 'True'.

Execution example: python CLR_main.py SchneiderData/400-25-1e 1 0 1 0.1 0 1

- SchneiderData/400-25-1e: data file's location/name, without .dat extension
- 1: Small-demand groups are considered in the assignment procedure (0 if not)
- 0: CFL is tackled via Local Search
     (1 if it is solved as an MIP;
      2 if it is solved as an IP, including Assignment)
- 1: Assignment IP (0 if Assignment LP + rounding); only relevant if CFL_mod < 2
- 0.1: Value of parameter epsilon (Local search)
- 0: No Gurobi output (1 otherwise)
- 1: Print the tours corresponding to the solution (0 if not)

II.2) IDE

Parameter 'commandline_input' (line 28) needs to be set to 'False'.

The parameters explained in (II.1) need to be set in the code itself (lines 50-56).

--------------------
III) FINAL COMMENTS

- Some additional parameters can be modified (lines 29-34).
- Understanding the parameters and their effects require not only understanding CLR, but probably also reading the paper (especially section 5).
