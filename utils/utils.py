import numpy as np

from env.simulator.code.simulator import SimulatorState, Simulator

from env.simulator.code.simulator.io.loading import load_init_env_data, load_container_data, load_os_data, \
                                                    load_test_container_data, load_test_os_data, load_be_os_data, load_be_container_data

import yaml,os

import math

import sys

from scipy.spatial.distance import cdist

INF = sys.float_info.max

def behavior_diversity(behavior):
    behavior = np.array(behavior)
    unique_rows = np.unique(behavior, axis=0)

    return unique_rows.shape[0]


def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            # x = x.astype(np.float64)
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

# ==========================
# simulation related
# ==========================

def validation_simulation(size):
    # Warm up the simulation
    init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(0).values()
    sim_state = SimulatorState(init_pms, init_vms, init_containers, init_os, init_pm_types, init_vm_types)

    # load the validation data
    input_os = load_test_os_data(start=0, end=size)
    input_containers, applications, applications_reverse = load_test_container_data(start=0, end=size)

    return sim_state, input_containers, applications, applications_reverse, input_os

# ==================
# behavior related
# ==================
def spearmans_footrule_distance_matrix(A, B):
    """spearmans footrule distance
    """
    
    # Broadcasting to compute the absolute differences
    differences = np.abs(A[:, np.newaxis, :] - B[np.newaxis, :, :])
    
    # Summing the differences along the last axis to get the Footrule distances
    distances = np.sum(differences, axis=2)
    
    return distances

def spearmans_footrule_pair_distance_matrix(A, B):
    """spearmans footrule distance
    """
    differences = np.abs(A - B)
    
    # Summing the differences along the last axis to get the Footrule distances
    distances = np.sum(differences)
    
    return distances

def ABOD_anomaly(tar_behavior, sou_behavior, n_neighbors, contamination=0.2):
    from pyod.models.abod import ABOD
    from sklearn.preprocessing import StandardScaler
    tar_behavior = np.array(tar_behavior, dtype=np.float64)
    sou_behavior = np.array(sou_behavior, dtype=np.float64)

    unique_tar = np.unique(tar_behavior, axis=0)
    unique_sou = np.unique(sou_behavior, axis=0)

    clf = ABOD(contamination=contamination, n_neighbors=n_neighbors)
    clf.fit(unique_tar)

    pred = clf.predict(unique_sou)

    transfer_index = np.where((sou_behavior[:, None] == unique_sou[pred == 0]).all(axis=2))[0]
    transfer_index = np.unique(transfer_index)
    print(len(transfer_index))

    return transfer_index

def knn_anomaly(tar_behavior, sou_behavior, k, percentage=95, metric="spearman"):
    from sklearn.neighbors import NearestNeighbors
    tar_behavior = np.array(tar_behavior)
    sou_behavior = np.array(sou_behavior)
    if metric == "hamming":
        nn = NearestNeighbors(n_neighbors=k, metric="hamming")
    else:
        nn = NearestNeighbors(n_neighbors=k, metric=spearmans_footrule_pair_distance_matrix)
    nn.fit(tar_behavior)
    distances, indices = nn.kneighbors(tar_behavior)
    anomaly_scores = np.sum(distances, axis=1)
    threshold = np.percentile(anomaly_scores, percentage)

    sou_distances = spearmans_footrule_distance_matrix(sou_behavior, tar_behavior)
    sorted_distances = np.sort(sou_distances, axis=1)[:, :k]
    mean_distances = np.sum(sorted_distances, axis=1)

    normal_index = np.where(mean_distances < threshold)[0]

    # normal_candidates = sou_behavior[normal_index]
    # _, unique_indices = np.unique(normal_candidates, axis=0, return_index=True)
    # normal_index = normal_index[unique_indices]
    print(len(normal_index))


    return normal_index

def distance_behavior(tar_be: list, sou_be: list, metric="spearman", axis=-1):
    """get the nearest behavior from the elitism of target task;
    :param tar_be: the target behavior list
    :param sou_be: source tasks behavior
    :param num_transfer: the number of transferred individuals
    """
    tar_be = np.array(tar_be)
    sou_be = np.array(sou_be)

    if metric == "hamming":
        dist = cdist(sou_be, tar_be, metric='hamming') * sou_be.shape[1]
    elif metric == "spearman":
        dist = spearmans_footrule_distance_matrix(sou_be, tar_be)

    if axis == -1:
        mean_dist = np.mean(dist)
    else:
        mean_dist = np.mean(dist, axis=axis)
    # dist = np.squeeze(np.mean(dist, axis=1))
    # epsilon = np.median(dist[dist != 0])
    # weights = np.where(dist == 0, 1 / epsilon, 1 / dist)
    # weights /= weights.sum()
    # print(mean_dist)
    # sample_data = np.random.choice(dist, size=40, p=weights)
    # print(sample_data)
    # print(len(sample_data[sample_data < mean_dist]))

    return mean_dist

def gp_permutation_behavior(gp_tree, pset, toolbox, dim=10, num_candidate=15, seed=20):
    """get the permutation behavior of each GP in decision points
    return: a behavior vector
    """
    np.random.seed(seed)

    func0 = toolbox.compile(expr=gp_tree["vm"], pset=pset["vm"])
    func1 = toolbox.compile(expr=gp_tree["pm"], pset=pset["pm"])
    # ===============
    # decision points
    # ===============
    container_cpus = np.random.uniform(100, 2000, dim)
    container_memories = np.random.uniform(100, 2000, dim)

    vm_cpu_capacities = np.random.uniform(1000, 8000, dim)
    vm_memory_capacities = np.random.uniform(1000, 8000, dim)

    # ============
    # VM candidate
    # ============
    vm_remaining_cpu_capacities = np.random.uniform(1000, 3000, num_candidate)
    vm_remaining_memory_capacities = np.random.uniform(1000, 3000, num_candidate)
    vm_remaining_cpu_capacities = np.array([cpu - cpu * 0.1 for cpu in vm_remaining_cpu_capacities])
    vm_remaining_memory_capacities = np.array([memory - 200 for memory in vm_remaining_memory_capacities])
    vm_cpu_overheads = np.array([cpu * 0.1 for cpu in vm_remaining_cpu_capacities])
    vm_memory_overheads = np.array([200 for memory in vm_remaining_memory_capacities])
    vm_pm_inner_communication = np.random.uniform(500, 1500, num_candidate)
    vm_pm_outer_communication = np.random.uniform(100, 500, num_candidate)
    vm_vector = []
    
    for point in range(len(container_cpus)):
        vm_container_cpus = np.array([container_cpus[point] for i in range(len(vm_remaining_cpu_capacities))])
        vm_container_memories = np.array([container_memories[point] for i in range(len(vm_remaining_cpu_capacities))])
        merged_features = np.array((vm_container_cpus, vm_container_memories,
                                    vm_remaining_cpu_capacities, vm_remaining_memory_capacities,
                                    vm_cpu_overheads, vm_memory_overheads,
                                    vm_pm_inner_communication, vm_pm_outer_communication))
        gp_vm_priority = func0(*merged_features)
        unavailable_vm = np.where((vm_remaining_cpu_capacities < vm_container_cpus)|(vm_remaining_memory_capacities < vm_container_memories))
        gp_vm_priority[unavailable_vm] = -INF
        sorted_indices = np.argsort(gp_vm_priority)
        sorted_indices = sorted_indices[::-1]
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(gp_vm_priority))
        vm_vector.append(ranks)

    pm_remaining_cpu_capacities = np.random.uniform(6000, 10000, num_candidate)
    pm_remaining_memory_capacities = np.random.uniform(6000, 10000, num_candidate)
    pm_cpu_capacities = pm_remaining_cpu_capacities / np.random.uniform(0, 1, 1)
    pm_memory_capacities = pm_remaining_memory_capacities / np.random.uniform(0, 1, 1)
    pm_inner_communication = np.random.uniform(500, 1500, num_candidate)
    pm_outer_communication = np.random.uniform(100, 500, num_candidate)
    pm_cores = [4, 8, 8, 12, 12, 16, 16, 12, 16, 16, 8, 8, 12, 12, 16]
    # pm_cores = [4, 8, 8, 12, 12]
    pm_vector = []

    for point in range(len(vm_cpu_capacities)):
        pm_vm_cpus = np.array([vm_cpu_capacities[point] for i in range(len(pm_remaining_cpu_capacities))])
        pm_vm_memories = np.array([vm_memory_capacities[point] for i in range(len(pm_remaining_cpu_capacities))])

        merged_features = np.array((pm_vm_cpus, pm_vm_memories,
                                    pm_remaining_cpu_capacities, pm_remaining_memory_capacities,
                                    pm_cpu_capacities, pm_memory_capacities, pm_cores, 
                                    pm_inner_communication, pm_outer_communication))
        gp_pm_priority = func1(*merged_features)
        unavailable_pm = np.where((pm_remaining_cpu_capacities < pm_vm_cpus)|(pm_remaining_memory_capacities < pm_vm_memories))
        gp_pm_priority[unavailable_pm] = -INF
        sorted_indices = np.argsort(gp_pm_priority)
        sorted_indices = sorted_indices[::-1]
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(gp_pm_priority))
        pm_vector.append(ranks)
    
    behavior = np.array(vm_vector + pm_vector)
    behavior = behavior.flatten()
    
    return behavior


def gp_behavior(gp_tree, pset, toolbox, dim=10, num_candidate=15, seed=20):
    """get the behavior of each GP in decision points
    return: a behavior vector
    """
    np.random.seed(seed)

    func0 = toolbox.compile(expr=gp_tree["vm"], pset=pset["vm"])
    func1 = toolbox.compile(expr=gp_tree["pm"], pset=pset["pm"])
    # ===============
    # decision points
    # ===============
    container_cpus = np.random.uniform(100, 2000, dim)
    container_memories = np.random.uniform(100, 2000, dim)

    vm_cpu_capacities = np.random.uniform(1000, 8000, dim)
    vm_memory_capacities = np.random.uniform(1000, 8000, dim)

    # ============
    # VM candidate
    # ============
    vm_remaining_cpu_capacities = np.random.uniform(1000, 3000, num_candidate)
    vm_remaining_memory_capacities = np.random.uniform(1000, 3000, num_candidate)
    vm_remaining_cpu_capacities = np.array([cpu - cpu * 0.1 for cpu in vm_remaining_cpu_capacities])
    vm_remaining_memory_capacities = np.array([memory - 200 for memory in vm_remaining_memory_capacities])
    vm_cpu_overheads = np.array([cpu * 0.1 for cpu in vm_remaining_cpu_capacities])
    vm_memory_overheads = np.array([200 for memory in vm_remaining_memory_capacities])
    vm_pm_inner_communication = np.random.uniform(500, 1500, num_candidate)
    vm_pm_outer_communication = np.random.uniform(100, 500, num_candidate)
    vm_vector = []
    
    for point in range(len(container_cpus)):
        vm_container_cpus = np.array([container_cpus[point] for i in range(len(vm_remaining_cpu_capacities))])
        vm_container_memories = np.array([container_memories[point] for i in range(len(vm_remaining_cpu_capacities))])
        # vm_container_affinity = np.array([container_affinity[point] for i in range(len(vm_remaining_cpu_capacities))])
        # container_os = np.array([os[point] for i in range(len(vm_remaining_cpu_capacities))])
        merged_features = np.array((vm_container_cpus, vm_container_memories,
                                    vm_remaining_cpu_capacities, vm_remaining_memory_capacities,
                                    vm_cpu_overheads, vm_memory_overheads,
                                    vm_pm_inner_communication, vm_pm_outer_communication))
        gp_vm_priority = func0(*merged_features)
        unavailable_vm = np.where((vm_remaining_cpu_capacities < vm_container_cpus)|(vm_remaining_memory_capacities < vm_container_memories))
        gp_vm_priority[unavailable_vm] = -INF
        key = np.argmax(gp_vm_priority)
        # reference rules: best fit
        best_fit_vm_priority = vm_container_cpus / vm_remaining_cpu_capacities
        # print(best_fit_vm_priority)
        best_fit_vm_priority[unavailable_vm] = -INF
        rank = np.argsort(best_fit_vm_priority)[::-1]
        best_fit_vm_priority = np.empty_like(rank)
        best_fit_vm_priority[rank] = np.arange(len(best_fit_vm_priority))
        # print(key)
        # print(best_fit_vm_priority)
        # print(best_fit_vm_priority[key])
        # print("--------")
        vm_vector.append(best_fit_vm_priority[key])
    # print(vm_vector)
    # print("========")
    # ============
    # PM candidate
    # ============
    pm_remaining_cpu_capacities = np.random.uniform(6000, 10000, num_candidate)
    pm_remaining_memory_capacities = np.random.uniform(6000, 10000, num_candidate)
    pm_cpu_capacities = pm_remaining_cpu_capacities / np.random.uniform(0, 1, 1)
    pm_memory_capacities = pm_remaining_memory_capacities / np.random.uniform(0, 1, 1)
    pm_inner_communication = np.random.uniform(500, 1500, num_candidate)
    pm_outer_communication = np.random.uniform(100, 500, num_candidate)
    pm_cores = [4, 8, 8, 12, 12, 16, 16, 12, 16, 16, 8, 8, 12, 12, 16]
    # pm_cores = [4, 8, 8, 12, 12]
    pm_vector = []

    for point in range(len(vm_cpu_capacities)):
        pm_vm_cpus = np.array([vm_cpu_capacities[point] for i in range(len(pm_remaining_cpu_capacities))])
        pm_vm_memories = np.array([vm_memory_capacities[point] for i in range(len(pm_remaining_cpu_capacities))])
        # pm_container_affinity = np.array([container_affinity[point] for i in range(len(pm_remaining_cpu_capacities))])
        # pm_vm_cores = np.array([vm_core[point] for i in range(len(pm_remaining_cpu_capacities))])

        merged_features = np.array((pm_vm_cpus, pm_vm_memories,
                                    pm_remaining_cpu_capacities, pm_remaining_memory_capacities,
                                    pm_cpu_capacities, pm_memory_capacities, pm_cores, 
                                    pm_inner_communication, pm_outer_communication))
        gp_pm_priority = func1(*merged_features)
        unavailable_pm = np.where((pm_remaining_cpu_capacities < pm_vm_cpus)|(pm_remaining_memory_capacities < pm_vm_memories))
        gp_pm_priority[unavailable_pm] = -INF
        key = np.argmax(gp_pm_priority)
        # print(f"pm: {unavailable_pm}")

        # reference rules: best fit
        best_fit_pm_priority = pm_vm_cpus / pm_remaining_cpu_capacities
        best_fit_pm_priority[unavailable_pm] = -INF
        rank = np.argsort(best_fit_pm_priority)[::-1]
        best_fit_pm_priority = np.empty_like(rank)
        best_fit_pm_priority[rank] = np.arange(len(best_fit_pm_priority))
        pm_vector.append(best_fit_pm_priority[key])
    
    return vm_vector + pm_vector

def tSNE_behavior(X1, X2, X3, dim=2):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.manifold import TSNE
    n_samples = len(X1)
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    y1 = np.zeros(len(X1))
    y2 = np.ones(len(X2))
    y3 = np.full(len(X3), 2)

    # 合并数据
    X = np.vstack((X1, X2, X3))
    y = np.hstack((y1, y2, y3))

    if dim == 2:
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(X)

        # 可视化降维后的数据
        plt.figure(figsize=(10, 8))
        plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c='blue', label='Group 1', marker='o', edgecolor='k', alpha=0.3)
        plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c='red', label='Group 2', marker='^', edgecolor='k', alpha=0.3)
        plt.scatter(X_2d[y == 2, 0], X_2d[y == 2, 1], c='green', label='Group 3', marker='s', edgecolor='k', alpha=0.5)
        plt.title('t-SNE Visualization of Three 20-Dimensional Data Groups')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'behavior_tSNE2D_distribution.png', bbox_inches='tight')
    else:
        tsne = TSNE(n_components=3, random_state=0)
        X_3d = tsne.fit_transform(X)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(X_3d[y == 0, 0], X_3d[y == 0, 1], X_3d[y == 0, 2], c='blue', label='Group 1', marker='o', edgecolor='k', alpha=0.3)
        scatter = ax.scatter(X_3d[y == 1, 0], X_3d[y == 1, 1], X_3d[y == 1, 2], c='red', label='Group 2', marker='^', edgecolor='k', alpha=0.3)
        scatter = ax.scatter(X_3d[y == 2, 0], X_3d[y == 2, 1], X_3d[y == 2, 2], c='green', label='Group 3', marker='s', edgecolor='k', alpha=0.3)

        ax.set_title('t-SNE Visualization of Three 20-Dimensional Data Groups')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.legend()
        plt.savefig(f'behavior_tSNE3D_distribution.png', bbox_inches='tight')


def training_simulation(case, workload="bitbrains", os_dataset="3OS"):
    # warm up simulation
    if case == -1:
        init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(0, workload=workload, os_dataset=os_dataset).values()
    else:
        init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(case, workload=workload, os_dataset=os_dataset).values()
    sim_state = SimulatorState(init_pms, init_vms, init_containers, init_os, init_pm_types, init_vm_types)

    # load the training data
    input_containers, applications, applications_reverse = load_container_data(case, workload=workload, os_dataset=os_dataset)
    input_os = load_os_data(case, workload=workload, os_dataset=os_dataset)

    return sim_state, input_containers, applications, applications_reverse, input_os

def testing_simulation(test_num=0, start=0, end=None, run=0, workload="bitbrains", os_dataset="3OS"):
    """Prepare simulator state and load testing data.

    Parameters:
    - test_num: which test case file to read from Test/ dirs (e.g., testCase{test_num}.csv)
    - start, end: slice boundaries for loading a subset of test rows
    - run: seed key used by test data loader to randomize grouping
    - workload: dataset name (e.g., "bitbrains")
    - os_dataset: OS variant folder (e.g., "3OS")

    Returns:
    - sim_state: initialized SimulatorState (deep-copied inside Simulator)
    - input_containers: DataFrame of test container rows
    - applications: dict (unused placeholder, kept for parity)
    - applications_reverse: mapping Application -> container id list
    - input_os: DataFrame of test OS ids
    """
    # warm up simulation using the same test case id (consistent with training_simulation style)
    init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(test_num, workload=workload, os_dataset=os_dataset).values()
    sim_state = SimulatorState(init_pms, init_vms, init_containers, init_os, init_pm_types, init_vm_types)

    # load the testing data
    input_containers, applications, applications_reverse = load_test_container_data(
        test_num=test_num, start=start, end=end, run=run, dataset=workload, os=os_dataset
    )
    input_os = load_test_os_data(start=start, end=end, test_num=test_num, dataset=workload, os=os_dataset)

    return sim_state, input_containers, applications, applications_reverse, input_os

def eval(i: int, energy_list, communication_list) -> float:
    """ multi-task evaluation
    :param task: 0: 0.5; 1: 0.75 * E; 2: 0.25 * E 
    :param tchebycheff: 0: sum up weight, 1: tchebycheff
    """
    return 0.5 * energy_list[i] / np.max(energy_list) + 0.5 * communication_list[i] / np.max(communication_list),

def set_seed(seed):
    np.random.seed(seed)

def get_config(path: str):
    filename = os.path.join(path)
    config_file = open(filename)
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    return config

# ==========================
# ==========================

# ==========================
# get individuals
# ==========================

def worst_individual(population):
    fitness = [ind.fitness for ind in population]
    worst_index = fitness.index(min(fitness))
    worst_individual = population[worst_index]

    return worst_individual

def best_individual(population):
    fitness = [ind.fitness for ind in population]
    best_index = fitness.index(max(fitness))
    best_individual = population[best_index]

    return best_individual

def optima_individual(population, type=0):
    """ Get the extreme individual in the pareto front
    """
    obj = [ind.fitness.values[type] for ind in population]
    best_index = obj.index(min(obj))

    return population[best_index]

def ensemble_individuals(front, type=0):
    """ Get five ensemble individuals if the front is larger than 5
    """
    ensemble = []
    if len(front) > 5:
        front = sorted(front, key=lambda x: x.fitness.values[type], reverse=True)
        for ind in front[ : 5]:
            ensemble.append(str(ind))
    else:
        for ind in front:
            ensemble.append(str(ind))

    return str(ensemble)

# ==========================
# ==========================
def cal_spearman_relatedness(behavior0, behavior1, behavior2):
    dist0 = distance_behavior(behavior0, behavior1)
    dist1 = distance_behavior(behavior0, behavior2)
    max_dist = max(dist0, dist1)
    norm_dist0 = dist0 / max_dist
    norm_dist1 = dist1 / max_dist
    return (norm_dist0 / (norm_dist0 + norm_dist1), norm_dist1 / (norm_dist0 + norm_dist1))

def cal_relatedness(behavior0, behavior1, behavior2):
    """calculate the relatedness of behaviors
    return: the probabilities of selecting task1 and task2 to assist task0 respectively
    """
    behavior0 = np.array(behavior0).T
    behavior1 = np.array(behavior1).T
    behavior2 = np.array(behavior2).T
    # calculate the number of each action
    counts0 = np.zeros((behavior0.shape[0], 15), dtype=int)
    counts1 = np.zeros((behavior1.shape[0], 15), dtype=int)
    counts2 = np.zeros((behavior2.shape[0], 15), dtype=int)

    for i, row in enumerate(behavior0):
        counts = np.bincount(row, minlength=15)
        counts0[i] = counts
    for i, row in enumerate(behavior1):
        counts = np.bincount(row, minlength=15)
        counts1[i] = counts
    for i, row in enumerate(behavior2):
        counts = np.bincount(row, minlength=15)
        counts2[i] = counts

    # calculate the KL divergence
    counts0 += 1
    counts1 += 1
    counts2 += 1
    sum_counts = np.sum(counts0, axis=1)[0]
    prob0 = counts0 / sum_counts
    prob1 = counts1 / sum_counts
    prob2 = counts2 / sum_counts

    # calculate relatedness
    log_div1 = np.multiply(prob1, np.log(prob1 / prob0))
    kl_diver1 = np.sum(log_div1, axis=1)
    relatedness1 = 1 / (1 + np.mean(kl_diver1))
    # calculate relatedness
    log_div2 = np.multiply(prob2, np.log(prob2 / prob0))
    kl_diver2 = np.sum(log_div2, axis=1)
    relatedness2 = 1 / (1 + np.mean(kl_diver2))

    # print(np.mean(kl_diver1), np.mean(kl_diver2))
    return [relatedness1 / (relatedness1 + relatedness2), relatedness2 / (relatedness1 + relatedness2)], relatedness1, relatedness2

# def prob_task(target_task, candidate_task0, candidate_task1):
#     """calculate the probability of each task
#     return: probability of each candidate task
#     """
#     target_behavior = np.array([ind.fitness.values[0] for ind in target_task])
#     candidate_behavior0 = np.array([ind.fitness.values[0] for ind in candidate_task0])
#     candidate_behavior1 = np.array([ind.fitness.values[0] for ind in candidate_task1])
#     # print(target_behavior)
#     similarity0 = 1 / (1 + kl_divergence(np.array([np.mean(candidate_behavior0)]), np.array([[np.std(candidate_behavior0)]]), np.array([np.mean(target_behavior)]), np.array([[np.std(target_behavior)]])))
#     similarity1 = 1 / (1 + kl_divergence(np.array([np.mean(candidate_behavior1)]), np.array([[np.std(candidate_behavior1)]]), np.array([np.mean(target_behavior)]), np.array([[np.std(target_behavior)]])))

#     return similarity0 / (similarity0 + similarity1), similarity1 / (similarity0 + similarity1)


# ========================================
# anomaly detection for knowledge transfer
# ========================================
def gaus_anomaly_detection(tar_behavior, sou_behavior0, sou_behavior1, K=3, e=0.1):
    """Model the gaussian distribution of behavior matrix
    :param N: size of sub_population
    :param K: number of tasks
    :param e: intensity of knowledge transfering
    return: the index of transferred individuals from source targets
    """
    # the index of transferred individuals
    transfer_list = []
    N = len(sou_behavior0)
    tar_behavior = np.array(tar_behavior)
    sou_behavior0 = np.array(sou_behavior0)
    sou_behavior1 = np.array(sou_behavior1)
    can_behavior = np.vstack((sou_behavior0, sou_behavior1))
    
    mean = np.mean(tar_behavior, axis=0)
    cov = np.cov(tar_behavior, rowvar=False)
    from scipy.stats import multivariate_normal
    rv = multivariate_normal(mean, cov, allow_singular=True)
    prob = rv.pdf(can_behavior)     # probability of candidate behaviors
    # number of transferred individuals
    num = int(e * N * (K - 1))
    sorted_indices = np.argsort(prob)[::-1]

    top_indices = sorted_indices[:num]
    print(prob[top_indices])
    print(len(np.unique(can_behavior[top_indices], axis=0)))
    return top_indices

def adaptive_transfer(individuals, sub_size, num_transferred):
    """adaptive the transfer parameter
    """
    # the worst individual in the original population
    worst_individual = sorted(individuals[ : 2 * sub_size], key=lambda x: x.fitness, reverse=True)[sub_size - 1]
    test = sorted(individuals[ : 2 * sub_size], key=lambda x: x.fitness, reverse=True)

    count = 0       # count the number of preserved transferred individuals
    for ind in individuals[-num_transferred : ]:
        if ind.fitness.values[0] < worst_individual.fitness.values[0]:
            count += 1
    if count / num_transferred == 0 or count / num_transferred == 1:
        return 0.1
    else:
        return count / num_transferred
# # ==========================
# # ==========================

# def degree_vectors(vector1, vector2):
#     dot_product = np.dot(vector1, vector2)

#     norm_vector1 = np.linalg.norm(vector1)
#     norm_vector2 = np.linalg.norm(vector2)

#     cosine_angle = dot_product / (norm_vector1 * norm_vector2)

#     angle_rad = np.arccos(cosine_angle)
#     angle_deg = np.degrees(angle_rad)

#     return angle_deg

# def tan_value(degree):
#     """ Use the degree to calculate the tan value
#     """
#     angle_radians = math.radians(degree)
    
#     return math.tan(angle_radians)

# def scatter_plot(e, c, e1, c1, gen):
#     import matplotlib.pyplot as plt
#     plt.clf()

#     plt.scatter(e, c, s=10, c="red", label="task 0")
#     plt.scatter(e1, c1, s=10, c="green", label="task 1")

#     plt.legend(loc='upper right')
#     plt.savefig(f'{gen}_plot.png', bbox_inches='tight')

# def scatter_plot(e, c, e1, c1, e2, c2, e3, c3, e4, c4, gen):
# # def scatter_plot(e, c, e1, c1, e2, c2, gen):
#     """ Draw the distribution of the individuals
#     """
#     import matplotlib.pyplot as plt

#     plt.clf()

#     def subproblem_1(x):
#         return tan_value(67.5) * x

#     def subproblem_2(x):
#         return tan_value(45) * x

#     def subproblem_3(x):
#         return tan_value(22.5) * x

#     def boundary_0(x):
#         return tan_value(78.75) * x

#     def boundary_1(x):
#         return tan_value(56.25) * x

#     def boundary_2(x):
#         return tan_value(33.75) * x

#     def boundary_3(x):
#         return tan_value(11.25) * x

#     x = np.arange(0, 1.1, 0.1)
#     plt.plot(x, subproblem_1(x), color="black")
#     plt.plot(x, subproblem_2(x), color="black")
#     plt.plot(x, subproblem_3(x), color="black")
#     plt.plot(x, boundary_0(x), color="gray", linestyle=":")
#     plt.plot(x, boundary_1(x), color="gray", linestyle=":")
#     plt.plot(x, boundary_2(x), color="gray", linestyle=":")
#     plt.plot(x, boundary_3(x), color="gray", linestyle=":")
#     plt.fill_between(x, boundary_0(x), np.max(boundary_0(x)) , color='#BAE3DC', interpolate=True)
#     plt.fill_between(x, boundary_0(x), boundary_1(x), color='#DAD7EA', interpolate=True)
#     plt.fill_between(x, boundary_1(x), boundary_2(x), color='#F6B3AC', interpolate=True)
#     plt.fill_between(x, boundary_2(x), boundary_3(x), color='#B3D1E7', interpolate=True)
#     plt.fill_between(x, boundary_3(x), 0, color='#BAE3DC', interpolate=True)

#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.scatter(e4, c4, s=10, c="pink", label="task 4")
#     plt.scatter(e3, c3, s=10, c="yellow", label="task 3")
#     plt.scatter(e2, c2, s=10, c="blue", label="task 2")
#     plt.scatter(e1, c1, s=10, c="green", label="task 1")
#     plt.scatter(e, c, s=10, c="red", label="task 0")
#     plt.legend(loc='upper right')
#     plt.savefig(f'{gen}_plot.png', bbox_inches='tight')


# def divide_region(pop, task, min_e, max_e, min_c, max_c):
#     """
#     :param pop: 
#     :params min_e, max_e, min_c, max_c: used to normalization
#     """
#     sub_region = []

#     for ind in pop:
#         norm_e = (ind.fitness.values[0] - min_e) / (max_e - min_e)
#         norm_c = (ind.fitness.values[1] - min_c) / (max_c - min_c)
#         ind_vectors = [norm_e, norm_c]
#         degree = degree_vectors(ind_vectors, [1, 0])
#         # if degree <= 11.25:
#         #     sub_region[0].append(ind)
#         if task == 0:
#             if degree <= 33.75 :
#                 sub_region.append(ind)
#         else:
#         # elif degree <= 56.25 and degree > 33.75:
#         #     sub_region[2].append(ind)
#             if degree > 56.25:
#                 sub_region.append(ind)
#         # else:
#         #     sub_region[4].append(ind)
    
#     return sub_region
# 
