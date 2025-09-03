# -*- coding: utf-8 -*-

"""
created on 5 November, 2023

@author: Zhengxin Fang
"""

import operator
import math
import random
import pickle

from env.simulator.code.simulator.io.loading import load_init_env_data, load_container_data, load_os_data, \
                                                    load_test_container_data, load_test_os_data, load_valid_init_env_data

from env.simulator.code.simulator import SimulatorState, Simulator

from env.simulator.code.simulator.metrics.pm import FirstFitPMAllocation


import numpy as np

from deap import algorithms, base, creator\
                ,tools, gp

import time

from deap.algorithms import varAnd

class SingleTreeGP():
    def __init__(self, seed) -> None:
        self.name = "single_gp"
        self.population_size = 100
        self.generation_num = 100
        self.cxpb = 0.8
        self.mutpb = 0.2
        self.elitism_size = 50
        self.tournament_size = 7
        # self.env = env
        self.arity = 6

        self.gen = 0
        self.seed = seed

        '''Setting of GP tree
        '''
        self.pset = gp.PrimitiveSet("MAIN", self.arity)
        self.pset.addPrimitive(np.add, 2)
        self.pset.addPrimitive(np.subtract, 2)
        self.pset.addPrimitive(np.multiply, 2)
        self.pset.addPrimitive(protectedDiv, 2)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=6)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.toolbox.register("evaluate", self.eval)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


    def eval(self, individual):
        func = self.toolbox.compile(expr=individual)
        hist_fitness = []       # save all fitness

        test_case_num = testcase_num

        for case in range(test_case_num):
            init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(case).values()
            # Warm up the simulation
            sim_state = SimulatorState(init_pms, init_vms, init_containers, init_os, init_pm_types, init_vm_types)
            sim = Simulator(sim_state)

            # load the training data
            input_containers = load_container_data(case)
            input_os = load_os_data(case)

            for i in range(len(input_os)) :
                # sim.heuristic_method(tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
                candidates = sim.vm_candidates(tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
                largest_priority = float("inf")
                selected_id = -1
                for vm in candidates:
                    if largest_priority > func(*tuple(vm[ : -1])):
                        selected_id = vm[-1]
                        largest_priority = func(*vm[ : -1])

                action = {"vm_num": selected_id}
                sim.step_first_layer(action, *tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
                ff_pm = FirstFitPMAllocation()
                if sim.to_allocate_vm_data != None:
                    pm_selection: dict = ff_pm(sim.state, sim.to_allocate_vm_data[1])
                    sim.step_second_layer(pm_selection, *tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
            
            # # # # # # # # # # # # # # # # # #  
            # debug
            # # # # # # # # # # # # # # # # # # 
            for k in range(len(sim.get_state().pm_actual_usage)):
                if sim.get_state().pm_actual_usage[k][0] < 0 or sim.get_state().pm_actual_usage[k][1] < 0:
                    print("total={:}".format(len(sim.get_state().pm_actual_usage)))
                    print("id={:}".format(k))
                    print("bug")

            hist_fitness.append(sim.running_energy_unit_time)

        # print(math.fsum(hist_fitness) / test_case_num)
        return math.fsum(hist_fitness) / test_case_num,
                
    def perturb(self):
        random.seed(self.seed)

        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(self.elitism_size)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = self.evoGP(pop, self.toolbox, self.cxpb, self.mutpb, self.generation_num, stats=mstats,
                                halloffame=hof, verbose=True)

        print('Best individual : ', str(hof[0]), hof[0].fitness)

        pool.close()

        return pop, log, hof

    def evoGP(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        start_time = time.time()  # Start time of generation

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        record["time"] = time.time() - start_time  # Time taken for the generation
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            print("Generation {:}".format(gen))
            self.gen = gen
            start_time = time.time()  # Start time of generation
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            record["time"] = time.time() - start_time  # Time taken for the generation
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)


        return population, logbook

    def save(self, pop, log, hof):
        with open(f'./results/model/pop_{time.time()}.pkl', 'wb') as pop_file:
            pickle.dump(pop, pop_file)

        with open(f'./results/model/log_{time.time()}.pkl', 'wb') as log_file:
            pickle.dump(log, log_file)

        with open(f'./results/model/hof_{time.time()}.pkl', 'wb') as hof_file:
            pickle.dump(hof, hof_file)

    def plot_use_pgv(self, hof):
        import pygraphviz as pgv
        nodes, edges, labels = gp.graph(hof[0])
        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]

        g.draw("./results/figure/tree.pdf")


def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

def test(a, b, c, d, e, f):
    return a+b+c+d+e
