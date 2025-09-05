import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import multiprocessing
import operator
import time
from datetime import datetime
import os
import json
import argparse

from deap import base
from deap import creator
from deap import tools
from deap import gp

from optim.multi_tree_gp import (
    MultiPrimitiveTree,
    cxOnePoint_type_wise,
    mutUniform_multi_tree,
    staticLimit,
)

from env.simulator.code.simulator import SimulatorState, Simulator

from utils.utils import *

config = get_config(path="./config/communication_mincut_gp.yaml")
sub_population_size0 = config["sub_population_size0"]
sub_population_size1 = config["sub_population_size1"]
generation_num = config["generation_num"]
cxpb = config["cxpb"]
mutpb = config["mutpb"]
arity0 = config["arity0"]
arity1 = config["arity1"]
elitism_size = config["elitism_size"]
tournament_size = config["tournament_size"]
min_depth = config["min_depth"]
max_depth = config["max_depth"]
bloat_control = config["bloat_control"]
mut_min_depth = config["mut_min_depth"]
mut_max_depth = config["mut_max_depth"]


DECIMAL_PRECISION = 6  # decimal precision for float comparison

# Base output directory under results/training with timestamp like YYYYMMDD-HHMM
_OUTPUT_TS = datetime.now().strftime("%Y%m%d-%H%M")
OUTPUT_DIR = os.path.join(".", "results", "training", _OUTPUT_TS)


# ==== MOEA/D toggles & hyperparameters ====
# neighborhood size as a ratio of population size (align with solve_moead_pymoo.py)
moead_T_ratio = 0.10
moead_T = max(2, int(moead_T_ratio * sub_population_size0))
moead_delta = 0.7  # prob. of selecting parents from neighbors
moead_nr = 2  # max neighbor replacements per offspring(keep diversity)


pset = {"vm": None, "pm": None}
# # terminal nodes list
TERMINAL_NODES = {
    "vm": {
        "ARG0": "container_cpu",
        "ARG1": "container_memories",
        "ARG2": "remaining_cpu_capacity",
        "ARG3": "remaining_memory_capacity",
        "ARG4": "vm_cpu_overhead",
        "ARG5": "vm_memory_overhead",
        "ARG6": "vm_pm_innerc",
        "ARG7": "vm_pm_outerc",
        "ARG8": "affinity",
    },
    "pm": {
        "ARG0": "vm_cpu_capacity",
        "ARG1": "vm_memory_capacity",
        "ARG2": "remaining_cpu_capacity",
        "ARG3": "remaining_memory_capacity",
        "ARG4": "pm_cpu_capacity",
        "ARG5": "pm_memory_capacity",
        "ARG6": "pm_core",
        "ARG7": "pm_innerc",
        "ARG8": "pm_outerc",
        "ARG9": "affinity",
    },
}


for type, item in pset.items():
    if type == "vm":
        pset[type] = gp.PrimitiveSet(type, arity0)
        pset[type].renameArguments(**TERMINAL_NODES["vm"])
    else:
        pset[type] = gp.PrimitiveSet(type, arity1)
        pset[type].renameArguments(**TERMINAL_NODES["pm"])
    pset[type].addPrimitive(np.add, 2)
    pset[type].addPrimitive(np.subtract, 2)
    pset[type].addPrimitive(np.multiply, 2)
    pset[type].addPrimitive(protectedDiv, 2)


creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", MultiPrimitiveTree, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
# register two trees as one individual
"""
initial two expressions
"""
toolbox.register(
    "expr",
    lambda: {
        "vm": gp.genHalfAndHalf(pset=pset["vm"], min_=min_depth, max_=max_depth),
        "pm": gp.genHalfAndHalf(pset=pset["pm"], min_=min_depth, max_=max_depth),
    },
)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("mate", cxOnePoint_type_wise)
toolbox.register("expr_mut", gp.genFull, min_=mut_min_depth, max_=mut_max_depth)
toolbox.register("mutate", mutUniform_multi_tree, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate(
    "mate", staticLimit(key=operator.attrgetter("height"), max_value=bloat_control)
)
toolbox.decorate(
    "mutate", staticLimit(key=operator.attrgetter("height"), max_value=bloat_control)
)


# ==========
# Evaluation
# ==========
def eval_individual(
    individual,
    sim_state: SimulatorState,
    containers,
    os,
    applications_reverse: dict,
    test=False,
) -> float:
    """evaluate dual-tree gp individual with min-cut"""
    sim = Simulator(sim_state, test=test)
    func0 = toolbox.compile(expr=individual["vm"], pset=pset["vm"])
    func1 = toolbox.compile(expr=individual["pm"], pset=pset["pm"])
    current_timestamp = sim.current_timestamp

    for app, clist in applications_reverse.items():
        container_clusters, P, P_container_id = app.min_cut(
            containers, os, 24000, 20000
        )

        num = 0
        for row in container_clusters.iterrows():
            vm_selection = sim.vm_selection(func0, app, [], row[1][0:3], row[1][3], 1)
            cid_list = []
            for p in range(len(P[num])):
                cid_list.append(app.vector_id_list[P[num][p]])

            num += 1
            sim.step_first_layer(
                vm_selection, cid_list, row[1][0:3], row[1][3], [False]
            )
            if sim.to_allocate_vm_data != None:
                pm_selection = sim.pm_selection(func1, app, [], 1)
                sim.step_second_layer(pm_selection, cid_list, row[1][0:3], row[1][3])

        # update the communication overhead information
        sim.update_current_communication(app)
        if sim.current_timestamp != current_timestamp:
            previous_timestamp = current_timestamp
            current_timestamp = sim.current_timestamp
            sim.update_total_communication(previous_timestamp, current_timestamp)
    # print(sim.running_energy_consumption, sim.running_communication_overhead)
    return (
        sim.running_energy_consumption,
        sim.running_communication_overhead,
    )


# ==========
# Logging & helpers (aligned with solve_mogp.py)
# ==========
def log_all_fronts(
    pop,
    gen: int,
    run: int,
    cpu_num: int,
    out_file: str | None = None,
    verbose: bool = False,
    log_fn=print,
):
    """Log all Pareto fronts for the given population to a JSON file."""
    fronts = tools.sortNondominated(pop, k=len(pop), first_front_only=False)

    # Compute global min/max across all fronts for normalization
    all_energy = [ind.fitness.values[0] for front in fronts for ind in front]
    all_comm = [ind.fitness.values[1] for front in fronts for ind in front]
    e_min, e_max = (min(all_energy), max(all_energy)) if all_energy else (0.0, 0.0)
    c_min, c_max = (min(all_comm), max(all_comm)) if all_comm else (0.0, 0.0)

    def _norm(v, vmin, vmax):
        return 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)

    payload = {"fronts": []}
    for rank, front in enumerate(fronts):
        exprs = []
        for ind in front:
            e_val = ind.fitness.values[0]
            c_val = ind.fitness.values[1]
            exprs.append(
                {
                    "expr": str(ind),
                    "energy": round(e_val, DECIMAL_PRECISION),
                    "communication": round(c_val, DECIMAL_PRECISION),
                    "energy_norm": round(_norm(e_val, e_min, e_max), DECIMAL_PRECISION),
                    "communication_norm": round(
                        _norm(c_val, c_min, c_max), DECIMAL_PRECISION
                    ),
                }
            )
        payload["fronts"].append({"rank": rank, "count": len(exprs), "exprs": exprs})

    # Default output path if none is provided
    if out_file is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_file = os.path.join(
            OUTPUT_DIR,
            f"nsw_Bitbrains_3OS_MOEA-D_core_{cpu_num}_run_{run}_gen_{gen}_fronts.json",
        )

    with open(out_file, "w") as f:
        json.dump(payload, f)

    if verbose:
        log_fn(f"[gen {gen}] Wrote all fronts to {out_file}")

    return out_file


def _build_first_front(pop):
    """Return first Pareto front entries with expr/energy/communication."""
    fronts = tools.sortNondominated(pop, k=len(pop), first_front_only=True)
    first = list(fronts[0])

    # Min-max for the first front only
    energies = [ind.fitness.values[0] for ind in first]
    comms = [ind.fitness.values[1] for ind in first]
    e_min, e_max = (min(energies), max(energies)) if energies else (0.0, 0.0)
    c_min, c_max = (min(comms), max(comms)) if comms else (0.0, 0.0)

    def _norm(v, vmin, vmax):
        return 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)

    best_front = []
    for ind in first:
        e_val = ind.fitness.values[0]
        c_val = ind.fitness.values[1]
        best_front.append(
            {
                "expr": str(ind),
                "energy": round(e_val, DECIMAL_PRECISION),
                "communication": round(c_val, DECIMAL_PRECISION),
                "energy_norm": round(_norm(e_val, e_min, e_max), DECIMAL_PRECISION),
                "communication_norm": round(
                    _norm(c_val, c_min, c_max), DECIMAL_PRECISION
                ),
            }
        )
    return best_front


def write_generation_log(
    gen: int,
    pop,
    logbook,
    json_file: str,
    run: int,
    cpu_num: int,
    log_fronts_gen: int | None = None,
    log_fronts_file: str | None = None,
):
    """Write summary and first front for a generation; optionally dump all fronts."""
    best_front = _build_first_front(pop)
    new_data = {
        str(gen): {
            "min_energy": round(
                logbook.chapters["energy"].select("min")[-1], DECIMAL_PRECISION
            ),
            "min_communication": round(
                logbook.chapters["communication"].select("min")[-1], DECIMAL_PRECISION
            ),
            "time": round(
                logbook.chapters["energy"].select("time")[-1], DECIMAL_PRECISION
            ),
            "first_front": best_front,
            "first_front_count": len(best_front),
        }
    }
    with open(json_file, "r") as gen_file:
        data = json.load(gen_file)
        data["generation"].update(new_data)
    with open(json_file, "w") as gen_file:
        json.dump(data, gen_file)

    if log_fronts_gen is not None and log_fronts_gen == gen:
        log_all_fronts(
            pop,
            gen=gen,
            run=run,
            cpu_num=cpu_num,
            out_file=log_fronts_file,
            verbose=True,
        )


def record_generation(logbook, mstats, pop, start_time, gen: int):
    """Compile stats, add elapsed minutes, record to logbook and print."""
    record = mstats.compile(pop) if mstats else {}
    record["time"] = (time.time() - start_time) / 60
    logbook.record(gen=gen, nevals=len(pop), **record)
    print(logbook.stream)


def register_sim_evaluator(toolbox, gen: int):
    """Register evaluate_individual with simulation data for the given generation."""
    sim_state, input_containers, applications, applications_reverse, input_os = (
        training_simulation(case=gen)
    )
    toolbox.register(
        "evaluate_individual",
        eval_individual,
        sim_state=sim_state,
        containers=input_containers,
        os=input_os,
        applications_reverse=applications_reverse,
    )


def eval_pop(individuals, toolbox):
    """Evaluate individuals via toolbox.evaluate_individual and assign fitness."""
    results = toolbox.map(toolbox.evaluate_individual, individuals)
    for ind, fit in zip(individuals, results):
        ind.fitness.values = fit


def init_training_json(json_file: str):
    training_result = {
        "sub_population_size": sub_population_size0,
        "cxpb": cxpb,
        "mutpb": mutpb,
        "elitism_size": elitism_size,
        "tournament_size": tournament_size,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "mut_min_depth": mut_min_depth,
        "mut_max_depth": mut_max_depth,
        # MOEA/D settings
        "neighborhood_size": moead_T,
        "neighborhood_select_prob": moead_delta,
        "max_neighbor_replacements": moead_nr,
        "generation": {},
    }
    with open(json_file, "w") as gen_file:
        json.dump(training_result, gen_file)


def resolve_generation_bounds(selected_gen: int | None, generation_num: int):
    """Return (loop_start, loop_end) based on optional selected generation."""
    if selected_gen is not None:
        return selected_gen, selected_gen + 1
    return 1, generation_num


def build_solve_moead_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train MOEA/D-based NSGP2 and optionally dump all Pareto fronts for a generation"
        ),
    )
    parser.add_argument("-r", "--RUN", help="run", dest="run", type=int, default="0")
    parser.add_argument("-s", "--SEED", help="seed", dest="seed", type=int, default="0")
    parser.add_argument(
        "--gen",
        dest="gen",
        type=int,
        default=None,
        help=(
            "Run only this generation index (range [gen, gen+1)). "
            "If omitted, runs 1..generation_num-1."
        ),
    )
    parser.add_argument(
        "--log-fronts-gen",
        dest="log_fronts_gen",
        type=int,
        default=None,
        help=(
            "If set, dump all Pareto fronts for this generation index "
            "to a separate JSON file."
        ),
    )
    parser.add_argument(
        "--log-fronts-file",
        dest="log_fronts_file",
        type=str,
        default=None,
        help=(
            "Optional output file path for --log-fronts-gen. If omitted, a default "
            "path under results/training is used."
        ),
    )
    return parser


def parse_solve_moead_args(argv=None):
    return build_solve_moead_parser().parse_args(argv)


# ==========================================
# MOEA/D helpers
# ==========================================
from math import sqrt


def generate_weight_vectors(n: int):
    """Evenly spaced weight vectors for bi-objective case."""
    if n <= 1:
        return [(1.0, 0.0)]
    return [(i / (n - 1), 1.0 - i / (n - 1)) for i in range(n)]


def build_neighbors(lambdas, T: int):
    """For each weight vector, return indices of T nearest neighbors by Euclidean distance."""
    N = len(lambdas)
    T = min(T, max(1, N - 1))
    neighbors = []
    for i, li in enumerate(lambdas):
        dists = []
        for j, lj in enumerate(lambdas):
            if i == j:
                continue
            dx = li[0] - lj[0]
            dy = li[1] - lj[1]
            d = sqrt(dx * dx + dy * dy)
            dists.append((d, j))
        dists.sort(key=lambda x: x[0])
        neighbors.append([j for _, j in dists[:T]])
    return neighbors


def compute_ideal_point(pop):
    """Compute ideal point z* from current evaluated population (bi-objective)."""
    f1_candidates = [ind.fitness.values[0] for ind in pop if ind.fitness.values[0] > 0]
    f1 = min(f1_candidates)
    f2_candidates = [ind.fitness.values[1] for ind in pop if ind.fitness.values[1] > 0]
    f2 = min(f2_candidates)

    return [f1, f2]


def tchebycheff(fit_tuple, lam, z_star):
    """Weighted Tchebycheff scalarization for bi-objective minimization."""
    return max(
        lam[0] * abs(fit_tuple[0] - z_star[0]), lam[1] * abs(fit_tuple[1] - z_star[1])
    )


def pick_parent_indices(N, neighbor_indices, delta):
    """Pick two parent indices: with prob delta from neighbors else from whole population."""

    if np.random.rand() < delta and len(neighbor_indices) >= 2:
        pool = neighbor_indices
    else:
        pool = list(range(N))
    if len(pool) >= 2:
        a, b = np.random.choice(pool, size=2, replace=False)
    else:
        # fallback in degenerate cases
        a = pool[0]
        b = (pool[0] + 1) % N
    return int(a), int(b)


def make_offspring_from_parents(pop, idx_a, idx_b, toolbox, cxpb, mutpb):
    """Create one offspring using registered GP operators (mate/mutate) decorated with staticLimit."""

    child1 = toolbox.clone(pop[idx_a])
    child2 = toolbox.clone(pop[idx_b])

    # Crossover
    if np.random.rand() < cxpb:
        child1, child2 = toolbox.mate(child1, child2)
        if hasattr(child1.fitness, "values"):
            if "values" in child1.fitness.__dict__:
                try:
                    del child1.fitness.values
                except Exception:
                    pass
        if hasattr(child2.fitness, "values"):
            if "values" in child2.fitness.__dict__:
                try:
                    del child2.fitness.values
                except Exception:
                    pass

    # Choose one of the two to continue
    base_child = child1

    # Mutation
    if np.random.rand() < mutpb:
        (base_child,) = toolbox.mutate(base_child)
        if hasattr(base_child.fitness, "values"):
            try:
                del base_child.fitness.values
            except Exception:
                pass

    return base_child


if __name__ == "__main__":
    args = parse_solve_moead_args()

    set_seed(args.seed)
    run = args.run

    valid_energy = []
    valid_communication = []
    valid_fitness = []
    worst_energy = []
    worst_communication = []

    # Output dir + json setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_file = os.path.join(
        OUTPUT_DIR,
        f"nsw_Bitbrains_3OS_MOEA-D_{run}_core_{config['cpu_num']}.json",
    )
    init_training_json(json_file)

    start_time = time.time()

    # Process Pool
    cpu_count = config["cpu_num"]
    print(f"CPU count: {cpu_count}")
    pool = multiprocessing.Pool(cpu_count)  # use multi-process of evaluation
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=sub_population_size0)

    # stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_obj1 = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_obj2 = tools.Statistics(lambda ind: ind.fitness.values[1])
    for stats in [stats_obj1, stats_obj2]:
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

    mstats = tools.MultiStatistics(energy=stats_obj1, communication=stats_obj2)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (mstats.fields if mstats else [])
    start_time = time.time()  # Start time of generation

    # Evaluate the individuals with an invalid fitness
    start_time = time.time()

    # ===========================================
    # apply simulator to evaluation for generation 0
    # ===========================================
    register_sim_evaluator(toolbox, gen=0)

    # Evaluate generation 0 population
    eval_pop(pop, toolbox)

    record_generation(logbook, mstats, pop, start_time, gen=0)

    # save diversity list (not persisted, kept for parity with mogp)
    fitness_list = [ind.fitness.values[0] for ind in pop]

    write_generation_log(
        gen=0,
        pop=pop,
        logbook=logbook,
        json_file=json_file,
        run=run,
        cpu_num=config["cpu_num"],
        log_fronts_gen=args.log_fronts_gen,
        log_fronts_file=args.log_fronts_file,
    )

    # ==== MOEA/D structures (built after gen 0 eval) ====
    N = len(pop)
    lambdas = generate_weight_vectors(N)
    neighbors = build_neighbors(lambdas, moead_T)
    z_star = compute_ideal_point(pop)
    print(f"[gen 0] z*: {z_star}")

    # Begin the generational process
    loop_start, loop_end = resolve_generation_bounds(args.gen, generation_num)
    for gen in range(loop_start, loop_end):
        # Warm up the simulation
        set_seed(gen)
        start_training_time = time.time()

        # ===== MOEA/D loop =====
        # Register evaluator for this generation's case and re-evaluate the
        # current population so fitness is consistent with the current case
        # before computing z* and doing neighborhood updates.
        register_sim_evaluator(toolbox, gen=gen)
        eval_pop(pop, toolbox)

        z_star = compute_ideal_point(pop)
        print(f"[gen {gen}] z*: {z_star}")

        # Random permutation of subproblems to avoid order bias
        order = np.random.permutation(len(pop))

        # ---- Batch create offsprings for parallel evaluation ----
        parent_pairs = []
        offsprings = []
        for j in order:
            a, b = pick_parent_indices(len(pop), neighbors[j], moead_delta)
            parent_pairs.append((j, a, b))
            child = make_offspring_from_parents(pop, a, b, toolbox, cxpb, mutpb)
            offsprings.append(child)

        # Evaluate offsprings in parallel
        offspring_fits = toolbox.map(toolbox.evaluate_individual, offsprings)
        for child, fit in zip(offsprings, offspring_fits):
            child.fitness.values = fit

        # ---- Neighborhood update (selection) ----
        for (j, a, b), child in zip(parent_pairs, offsprings):
            # Update ideal point with the new evaluated child
            z_star[0] = min(z_star[0], child.fitness.values[0])
            z_star[1] = min(z_star[1], child.fitness.values[1])

            # Replace up to moead_nr neighbors if improved under their respective scalarization
            replaced = 0
            for k in neighbors[j]:
                if tchebycheff(child.fitness.values, lambdas[k], z_star) <= tchebycheff(
                    pop[k].fitness.values, lambdas[k], z_star
                ):
                    pop[k] = toolbox.clone(child)
                    replaced += 1
                    if replaced >= moead_nr:
                        break

        # Ensure the logged population reflects current-case evaluations
        # (safe even if all individuals already have up-to-date fitness).
        eval_pop(pop, toolbox)
        record_generation(logbook, mstats, pop, start_training_time, gen)

        write_generation_log(
            gen=gen,
            pop=pop,
            logbook=logbook,
            json_file=json_file,
            run=run,
            cpu_num=config["cpu_num"],
            log_fronts_gen=args.log_fronts_gen,
            log_fronts_file=args.log_fronts_file,
        )

    pool.close()
    pool.join()
