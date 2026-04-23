import numpy as np
strategies = [ "Inventory Buffering", "Multi-Sourcing", "Production Flexibility",
    "Information Sharing", "Strategic Contracting", "Backup Facilities"]
scenario0 = np.array([
    [2,3,2,3,3,3,2],
    [3,4,3,3,4,3,2],
    [2,3,4,4,4,3,2],
    [4,3,3,4,5,2,4],
    [4,3,3,3,5,2,3],
    [1,4,5,5,4,1,1]
])
scenario1 = np.array([
    [2,4,2,4,3,3,2],
    [3,5,4,4,5,3,2],
    [2,3,5,4,4,3,2],
    [4,3,3,3,5,2,4],
    [4,3,2,3,5,2,3],
    [1,4,4,4,4,1,1]
])

scenario2 = np.array([
    [1,4,3,5,3,3,2],
    [2,2,3,2,3,3,2],
    [1,4,5,5,4,3,2],
    [2,4,3,5,5,2,3],
    [2,3,3,2,3,2,3],
    [1,3,5,5,4,1,1]
])

scenario3 = np.array([
    [2,3,2,3,3,4,2],
    [3,4,4,4,4,3,2],
    [2,4,5,4,4,2,1],
    [3,3,3,4,4,3,3],
    [3,3,3,3,4,2,3],
    [1,5,5,5,4,1,1]
])

weights = np.array([0.04, 0.29, 0.22, 0.12, 0.22, 0.02, 0.08])

def topsis(matrix, weights):
    # Normalize matrix
    norm = matrix / np.sqrt((matrix**2).sum(axis=0))
    # Apply weights
    weighted = norm * weights
    # Define criteria types: True = benefit, False = cost
    benefit = np.array([False, True, True, True, True, False, True])

    # Compute ideal best and worst correctly
    ideal_best = np.zeros(weighted.shape[1])
    ideal_worst = np.zeros(weighted.shape[1])

    for j in range(weighted.shape[1]):
        if benefit[j]:  # benefit criterion → maximize
            ideal_best[j] = weighted[:, j].max()
            ideal_worst[j] = weighted[:, j].min()
        else:  # cost criterion → minimize
            ideal_best[j] = weighted[:, j].min()
            ideal_worst[j] = weighted[:, j].max()
    
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

# ================= RUN TEST =================

scenarios = {
    "Scenario 0 (Stable)": scenario0,
    "Scenario 1 (Supplier)": scenario1,
    "Scenario 2 (Demand)": scenario2,
    "Scenario 3 (Multi-risk)": scenario3
}

for name, matrix in scenarios.items():
    print("\n" + "="*50)
    print(name)

    ci, ranking = run_topsis(matrix, weights)

    print("\nCloseness Coefficients:")
    for i, score in enumerate(ci):
        print(f"{strategies[i]}: {round(score, 4)}")

    print("\nRanking:")
    for rank, idx in enumerate(ranking, 1):
        print(f"{rank}. {strategies[idx]} ({round(ci[idx],4)})")
