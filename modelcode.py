import numpy as np
strategies = [ "Inventory Buffering", "Multi-Sourcing", "Production Flexibility",
    "Information Sharing", "Strategic Contracting", "Backup Facilities"]
scenario1 = np.array([
    [1,4,2,5,4,3,2],
    [2,5,4,4,5,3,2],
    [1,3,5,4,4,2,2],
    [3,3,4,4,4,3,3],
    [4,4,2,3,5,2,3],
    [1,3,4,3,4,1,1]
])

scenario2 = np.array([
    [1,5,2,5,3,4,2],
    [2,2,3,2,3,2,2],
    [1,5,5,5,4,2,2],
    [2,5,3,5,5,2,3],
    [2,3,3,2,3,2,3],
    [1,5,5,5,4,2,1]
])

scenario3 = np.array([
    [1,4,2,4,3,4,2],
    [2,4,4,4,4,3,2],
    [1,3,5,3,4,2,1],
    [3,3,3,4,4,3,3],
    [3,3,3,3,4,2,3],
    [1,4,5,4,4,1,1]
])

weights = np.array([0.04, 0.29, 0.22, 0.12, 0.02, 0.22, 0.08])

def topsis(matrix, weights):
    # Normalize matrix
    norm = matrix / np.sqrt((matrix**2).sum(axis=0))
    # Apply weights
    weighted = norm * weights
    # Ideal best and worst
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    # Distances
    d_pos = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
    d_neg = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))
    # Closeness coefficient
    ci = d_neg / (d_pos + d_neg)
    return ci


def run_topsis(matrix, weights):
    ci = topsis(matrix, weights)
    ranking = np.argsort(-ci)
    return ci, ranking