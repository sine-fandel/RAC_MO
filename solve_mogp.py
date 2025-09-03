"""
training the GPHH to solve dynamic RAC problems

@Author: Zhengxin Fang
    email: zhengxin.fang@ecs.vuw.ac.nz
"""
import numpy as np
import multiprocessing
import operator
import time

from deap import base
from deap import creator
from deap import tools
from deap import gp

from optim.multi_tree_gp import MultiPrimitiveTree, cxOnePoint_type_wise, mutUniform_multi_tree, staticLimit, \
                                genFull_mutual, assignCrowdingDist

from deap.algorithms import varAnd, varOr

from env.simulator.code.simulator import SimulatorState, Simulator

from env.simulator.code.simulator.io.loading import load_init_env_data, load_container_data, load_os_data

from utils.utils import *

config = get_config(path='./config/communication_mincut_gp.yaml')
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


pset = {"vm": None, "pm": None}
# # terminal nodes list
TERMINAL_NODES = {"vm": {"ARG0": "container_cpu", "ARG1": "container_memories", "ARG2": "remaining_cpu_capacity", "ARG3": "remaining_memory_capacity", "ARG4": "vm_cpu_overhead", "ARG5": "vm_memory_overhead", "ARG6": "vm_pm_innerc", "ARG7": "vm_pm_outerc", "ARG8": "affinity"},
            "pm": {"ARG0": "vm_cpu_capacity", "ARG1": "vm_memory_capacity", "ARG2": "remaining_cpu_capacity", "ARG3": "remaining_memory_capacity", "ARG4": "pm_cpu_capacity", "ARG5": "pm_memory_capacity", "ARG6": "pm_core", "ARG7": "pm_innerc", "ARG8": "pm_outerc", "ARG9": "affinity"}}


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
        "vm":
            gp.genHalfAndHalf(
                pset=pset["vm"],
                min_=min_depth,
                max_=max_depth
            ),
        "pm":
            gp.genHalfAndHalf(
                pset=pset["pm"],
                min_=min_depth,
                max_=max_depth
            )

    }
)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("select", tools.selTournamentDCD)
toolbox.register("environment_select", tools.selNSGA2)
toolbox.register("mate", cxOnePoint_type_wise)
toolbox.register("expr_mut", gp.genFull, min_=mut_min_depth, max_=mut_max_depth)
toolbox.register("mutate", mutUniform_multi_tree, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=bloat_control))
toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=bloat_control))

# ==========
# Evaluation
# ==========
def eval_individual(individual, sim_state: SimulatorState, containers, os, \
          applications_reverse: dict, test=False) -> float:
    """evaluate dual-tree gp individual with min-cut
    """
    sim = Simulator(sim_state, test=test)
    func0 = toolbox.compile(expr=individual["vm"], pset=pset["vm"])
    func1 = toolbox.compile(expr=individual["pm"], pset=pset["pm"])
    current_timestamp = sim.current_timestamp

    for app, clist in applications_reverse.items():
        container_clusters, P, P_container_id = app.min_cut(containers, os, 24000, 20000)

        num = 0
        for row in container_clusters.iterrows():
            vm_selection = sim.vm_selection(func0, app, [], row[1][0 : 3], row[1][3], 1)
            cid_list = []
            for p in range(len(P[num])):
                cid_list.append(app.vector_id_list[P[num][p]]) 
            
            num += 1
            sim.step_first_layer(vm_selection, cid_list, row[1][0 : 3], row[1][3], [False])
            if sim.to_allocate_vm_data != None:
                pm_selection = sim.pm_selection(func1, app, [], 1)
                sim.step_second_layer(pm_selection, cid_list, row[1][0 : 3], row[1][3])

        # update the communication overhead information
        sim.update_current_communication(app)
        if sim.current_timestamp != current_timestamp:
            previous_timestamp = current_timestamp
            current_timestamp = sim.current_timestamp
            sim.update_total_communication(previous_timestamp, current_timestamp)
    # print(sim.running_energy_consumption, sim.running_communication_overhead)
    return sim.running_energy_consumption, sim.running_communication_overhead,


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--RUN", help="run", dest="run", type=int, default="0")
    parser.add_argument("-s", "--SEED", help="seed", dest="seed", type=int, default="0")
    args = parser.parse_args()

    set_seed(args.seed)
    run = args.run

    valid_energy = []
    valid_communication = []
    valid_fitness = []
    worst_energy = []
    worst_communication = []

    # create json file to save training data
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
                        "generation": {

                        }
                    }
    
    
    json_file = f"./results/training/nsw_Bitbrains_3OS_NSGP2_{run}_core_{config['cpu_num']}.json"
    with open(json_file, "w") as gen_file:
        json.dump(training_result, gen_file)

    start_time = time.time()

    # Process Pool
    cpu_count = config["cpu_num"]
    print(f"CPU count: {cpu_count}")
    pool = multiprocessing.Pool(cpu_count)      # use multi-process of evaluation
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

    mstats = tools.MultiStatistics(obj1=stats_obj1, obj2=stats_obj2)
    # mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    # mstats.register("avg", np.mean)
    # mstats.register("std", np.std)
    # mstats.register("min", np.min)
    # mstats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
    start_time = time.time()  # Start time of generation

    # Evaluate the individuals with an invalid fitness
    start_time = time.time()  

    # ===========================================
    # apply simulator to evalution the population
    # ===========================================
    # get training data
    sim_state, input_containers, applications, applications_reverse, input_os = training_simulation(case=0)
    toolbox.register("evaluate_individual", eval_individual, sim_state=sim_state, containers=input_containers, os=input_os, applications_reverse=applications_reverse)

    # save the energy and communication for each individual
    fitnesses = toolbox.map(toolbox.evaluate_individual, pop)
    energy_list = []
    communication_list = []
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        energy_list.append(fit[0])
        communication_list.append(fit[1])
    # ===========================================
    # ===========================================

    record = mstats.compile(pop) if mstats else {}
    record["time"] = (time.time() - start_time) / 60  # Time taken for the generation
    logbook.record(gen=0, nevals=len(pop), **record)
    print(logbook.stream)

    # save diversity list
    fitness_list = []
    for i in range(len(pop)):
        fitness_list.append(pop[i].fitness.values[0])

    fronts = tools.sortNondominated(pop, k=len(pop), first_front_only=True)
    best_front = []
    for ind in fronts[0]:
        best_front.append(str(ind))
    # update json file to save the best training result of each generation
    new_data = {str(0): {"obj1": round(logbook.chapters["obj1"].select("min")[-1], 2), "obj2": round(logbook.chapters["obj2"].select("min")[-1], 2), "time": round(logbook.chapters["obj1"].select("time")[-1], 2), "best": best_front}}
    with open(json_file, "r") as gen_file:
        data = json.load(gen_file)
        data["generation"].update(new_data)
    with open(json_file, "w") as gen_file:
        json.dump(data, gen_file)
    
    # Begin the generational process
    for gen in range(1, generation_num):
        # Warm up the simulation
        set_seed(gen)
        start_training_time = time.time()
        # Select the next generation individuals
        assignCrowdingDist(pop)
        offspring = toolbox.select(pop, len(pop))
        # evolutionary operators
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # ===========================================
        # apply simulator to evalution the population
        # ===========================================
        # get training data
        sim_state, input_containers, applications, applications_reverse, input_os = training_simulation(case=gen)
        toolbox.register("evaluate_individual", eval_individual, sim_state=sim_state, containers=input_containers, os=input_os, applications_reverse=applications_reverse)
        # Replace the current population by the offspring
        all_pop = pop + offspring
        individual_results = toolbox.map(toolbox.evaluate_individual, all_pop)
        energy_list = []
        communication_list = []
        for ind, fit in zip(all_pop, individual_results):
            ind.fitness.values = fit
            energy_list.append(fit[0])
            communication_list.append(fit[1])
        energy_list = np.array(energy_list)
        communication_list = np.array(communication_list)
        pop = toolbox.environment_select(all_pop, len(pop))

        # ===========================================
        # ===========================================
        # ===========================================
        # Append the current generation statistics to the logbook
        record = mstats.compile(pop) if mstats else {}
        record["time"] = (time.time() - start_training_time) / 60  # Time taken for the generation
        logbook.record(gen=gen, nevals=len(pop), **record)
        print(logbook.stream)
        

        # update json file to save the best training result of each generation
        fronts = tools.sortNondominated(pop, k=len(pop), first_front_only=True)
        best_front = []
        for ind in fronts[0]:
            best_front.append(str(ind))
        new_data = {str(gen): {"obj1": round(logbook.chapters["obj1"].select("min")[-1], 2), "obj2": round(logbook.chapters["obj2"].select("min")[-1], 2), "time": round(logbook.chapters["obj1"].select("time")[-1], 2), "best": best_front}}
        with open(json_file, "r") as gen_file:
            data = json.load(gen_file)
            data["generation"].update(new_data)
        with open(json_file, "w") as gen_file:
            json.dump(data, gen_file)

    pool.close()
