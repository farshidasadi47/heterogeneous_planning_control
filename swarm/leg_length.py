#%%
import os
from itertools import combinations, permutations
from collections import deque
import numpy as np

if os.name == "posix":
    os.system("clear")

root = [7.5,5]
n_mode = 3
n_robot = 3
seed = root*2

combos = list(set(permutations(seed,n_robot)))

matrices= []
for combo in combos:
    # Make aggrergeated B.
    B = np.ones((1,n_robot))
    for i in range(n_mode - 1):
        B = np.append(B,[combo],axis = 0)
        B[i+1] = np.roll(B[i+1],-i)
    # Make beta matrix
    beta = B/ np.tile(B[:,0],(n_robot,1)).T
    #
    if np.linalg.matrix_rank(beta) >= n_robot:
        _, e, _= np.linalg.svd(beta, full_matrices=False)
        #e = e/e[0]
        det = e.prod()
        msg = ""
        for row in np.hstack((B,beta)):
            msg += ", ".join(f"{i:05.2f}" for i in row[:n_robot]) + " | "
            msg += ", ".join(f"{i:05.2f}" for i in row[n_robot:]) + "\n"
        msg = msg[:-1]
        matrices.append([det, e, msg])
# Sort based on determinant
matrices = sorted(matrices, key= lambda x: min(map(abs,x[1])))
# Print information.
for matrix in matrices:
    print(matrix[2])
    print(f"det: {matrix[0]:+011.6f}| ",end="")
    print(", ".join(f"{i:05.2f}" for i in matrix[1]))
    print("*"*72)
