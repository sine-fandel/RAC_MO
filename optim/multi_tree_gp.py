from collections.abc import Iterator
import copy
from typing import Hashable, Union
from functools import wraps
import random
from enum import Enum
from deap.gp import PrimitiveTree, cxOnePoint, mutUniform, PrimitiveSet

import sys

import numpy as np

from utils.utils import *

__type__ = object

class TerminalTypeEnum(Enum):
    VM = "vm"
    PM = "pm"

class MultiPrimitiveTree(dict):
    def __init__(self, content: dict):
        """Create multi-tree GP
        """
        for type, subtree in content.items():
            self[type] = PrimitiveTree(subtree)
    
    def __str__(self) -> str:
        return super().__str__()
    
    @classmethod
    def from_string(cls, content, pset):
        """Try to convert a dict which contains expression into a MultiPrimitiveTree given a
        a PrimitiveSet `pset`. The primirive set needs to contain every primitive
        """
        _content: dict = None
        if isinstance(content, str):
            # try to convert the string into dict
            try:
                _content = eval(content)
            except NameError as e:
                raise ValueError(f"Argument content should be dict like.")
        elif isinstance(content, dict):
            _content = content
        else:
            raise ValueError("Content should be either dict or str")
        tree_dict: dict = {}
        for type, string in _content.items():
            # convert type from strin to enum type
            _pset = None
            if isinstance(pset, PrimitiveSet):
                _pset = pset
            elif isinstance(pset, dict):
                if not pset.get(type, None):
                    raise ValueError(f"{type} is not in {pset=}")
                _pset = pset[type]
            else:
                raise ValueError(f"pset should be PrimitiveSet or dict")

            tree_dict[type] = PrimitiveTree.from_string(string, _pset)
        return MultiPrimitiveTree(tree_dict)

    def __str__(self) -> str:
        return str({
            type: str(ind)
            for type, ind in self.items()
        })

def cxOnePoint_type_wise(ind1: MultiPrimitiveTree, ind2: MultiPrimitiveTree):
    """Exchange subtrees between each individual with the same tree type
    """
    if ind1.keys() != ind2.keys():
        raise ValueError("ind1 and ind2 doesn't have the same tree type")
    keys = ind1.keys()
    for k in keys:
        ind1[k], ind2[k] = cxOnePoint(ind1=ind1[k], ind2=ind2[k])
    return ind1, ind2


def mutUniform_multi_tree(individual: MultiPrimitiveTree, expr, pset):
    for k in individual.keys():
        mutUniform(individual[k], expr, pset[k])
    return individual,


def assignCrowdingDist(individuals):
    """Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0].fitness.values)

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist



######################################
# Multi-tree GP bloat control decorators        
#######################################


def staticLimit(key, max_value):
    """Implement a static limit on some measurement on a GP tree, as defined
    by Koza in [Koza1989]. It may be used to decorate both crossover and
    mutation operators. When an invalid (over the limit) child is generated,
    it is simply replaced by one of its parents, randomly selected.

    This operator can be used to avoid memory errors occurring when the tree
    gets higher than 90 levels (as Python puts a limit on the call stack
    depth), because it can ensure that no tree higher than this limit will ever
    be accepted in the population, except if it was generated at initialization
    time.

    :param key: The function to use in order the get the wanted value. For
                instance, on a GP tree, ``operator.attrgetter('height')`` may
                be used to set a depth limit, and ``len`` to set a size limit.
    :param max_value: The maximum value allowed for the given measurement.
    :returns: A decorator that can be applied to a GP operator using \
    :func:`~deap.base.Toolbox.decorate`

    .. note::
       If you want to reproduce the exact behavior intended by Koza, set
       *key* to ``operator.attrgetter('height')`` and *max_value* to 17.

    .. [Koza1989] J.R. Koza, Genetic Programming - On the Programming of
        Computers by Means of Natural Selection (MIT Press,
        Cambridge, MA, 1992)

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                for type, sub_ind in ind.items():
                    if key(sub_ind) > max_value:
                        new_inds[i][type] = random.choice(keep_inds)[type]
            return new_inds

        return wrapper

    return decorator


######################
# New generation method to consider mutual information
######################
def get_pair(mutual_matrix):
    """
    get pair according to mutual matrix
    """
    mutual_list = mutual_matrix.flatten()
    index = [i + i * mutual_matrix.shape[0] for i in range(mutual_matrix.shape[0])]
    mutual_list = np.delete(mutual_list, index)
    mutual_list = np.unique(mutual_list)
    
    from scipy.special import softmax
    prob = softmax(mutual_list)
    choice = np.random.choice(mutual_list, 1, p=prob)
    candidate_pairs = np.argwhere(mutual_matrix==choice)

    if candidate_pairs.shape[0] == 0:
        i = np.random.randint(0, mutual_matrix.shape[0])
        j = np.random.randint(0, mutual_matrix.shape[0])
        selected_pair = [i, j]
    else:
        randint = np.random.randint(0, candidate_pairs.shape[0])
        selected_pair = candidate_pairs[randint, :]

    return selected_pair

    
def genFull_mutual(pset, min_, max_, mutual_matrix, type_=None):
    """Generate an expression where each leaf has the same depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A full tree with all leaves at the same depth.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    return generate(pset, mutual_matrix, min_, max_, condition, type_)

def genGrow_mutual(pset, min_, max_, mutual_matrix, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.
        """
        return depth == height or \
            (depth >= min_ and random.random() < pset.terminalRatio)

    return generate(pset, mutual_matrix, min_, max_, condition, type_)


def genHalfAndHalf_mutual(pset, mutual_matrix, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow_mutual, genFull_mutual))
    return method(pset, mutual_matrix, min_, max_, type_)

def generate(pset, mutual_matrix, min_, max_, condition, type_=None):
    """Generate a tree as a list of primitives and terminals in a depth-first
    order. The tree is built from the root to the leaves, and it stops growing
    the current branch when the *condition* is fulfilled: in which case, it
    back-tracks, then tries to grow another branch until the *condition* is
    fulfilled again, and so on. The returned list can then be passed to the
    constructor of the class *PrimitiveTree* to build an actual tree object.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              depending on the condition function.
    """
    mutual_matrix = mutual_matrix

    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    first = 0
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                if first == 0:
                    pair_index = get_pair(mutual_matrix)
                    term = pset.terminals[type_][pair_index[0]]
                    first = 1
                else:
                    term = pset.terminals[type_][pair_index[1]]
                    first = 0

            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a terminal of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a primitive of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr

# =========
# Multitask
# =========
def elite_behavior_mt_varAnd(target_task, source_task0, source_task1, nearest_index, toolbox, cxpb, mutpb, rmp):
# def multitask_varAnd(sub_pop0, sub_pop1, toolbox, cxpb, mutpb, rmp):
    """
    """
    target_offspring = [toolbox.clone(ind) for ind in target_task]
    source_pop0 = [toolbox.clone(ind) for ind in source_task0]
    source_pop1 = [toolbox.clone(ind) for ind in source_task1]
    source_pop = source_pop0 + source_pop1
    sub_size = len(target_offspring)

    # Apply crossover and mutation on the offspring
    # for Task0
    i = 0
    while i < sub_size:
        if random.random() < cxpb:
            if random.random() <= rmp :
                index = random.randint(0, len(nearest_index) - 1)
                target_offspring[i], _ = toolbox.mate(target_offspring[i],
                                                    toolbox.clone(source_pop[nearest_index[index]]))
                del target_offspring[i].fitness.values

                i += 1
            elif sub_size - i >= 2:
                target_offspring[i], target_offspring[i + 1] = toolbox.mate(target_offspring[i],
                                                                        target_offspring[i + 1])
                del target_offspring[i].fitness.values, target_offspring[i + 1].fitness.values
                
                i += 2
            
            else:
                target_offspring[i], _ = toolbox.mate(target_offspring[i],
                                                    target_offspring[i - 1])
                del target_offspring[i].fitness.values
                
                i += 1
            
    for i in range(sub_size):
        if random.random() < mutpb:
            target_offspring[i], = toolbox.mutate(target_offspring[i])
            del target_offspring[i].fitness.values

    return target_offspring


def behavior_mt_varAnd(target_task, source_task0, source_task1, prob, toolbox, cxpb, mutpb, rmp):
# def multitask_varAnd(sub_pop0, sub_pop1, toolbox, cxpb, mutpb, rmp):
    """
    """
    target_offspring = [toolbox.clone(ind) for ind in target_task]
    source_offspring0 = [toolbox.clone(ind) for ind in source_task0]
    source_offspring1 = [toolbox.clone(ind) for ind in source_task1]
    sub_size = len(target_offspring)
    # Apply crossover and mutation on the offspring
    # for Task0
    i = 0
    while i < sub_size:
        if random.random() < cxpb:
            if random.random() <= rmp :
                if random.random() <= prob[0]:
                    selected_individual = toolbox.select(source_offspring0, 1)[0]
                    target_offspring[i], _ = toolbox.mate(target_offspring[i],
                                                        toolbox.clone(selected_individual))
                else:
                    selected_individual = toolbox.select(source_offspring1, 1)[0]
                    target_offspring[i], _ = toolbox.mate(target_offspring[i],
                                                        toolbox.clone(selected_individual))
                del target_offspring[i].fitness.values

                i += 1
            elif sub_size - i >= 2:
                target_offspring[i], target_offspring[i + 1] = toolbox.mate(target_offspring[i],
                                                                        target_offspring[i + 1])
                del target_offspring[i].fitness.values, target_offspring[i + 1].fitness.values
                
                i += 2
            
            else:
                target_offspring[i], _ = toolbox.mate(target_offspring[i],
                                                    target_offspring[i - 1])
                del target_offspring[i].fitness.values
                
                i += 1
            
    for i in range(sub_size):
        if random.random() < mutpb:
            target_offspring[i], = toolbox.mutate(target_offspring[i])
            del target_offspring[i].fitness.values

    return target_offspring



def multitask_varAnd(target_task, source_task0, source_task1, toolbox, cxpb, mutpb, rmp):
# def multitask_varAnd(sub_pop0, sub_pop1, toolbox, cxpb, mutpb, rmp):
    target_offspring = [toolbox.clone(ind) for ind in target_task]
    source_offspring0 = [toolbox.clone(ind) for ind in source_task0]
    source_offspring1 = [toolbox.clone(ind) for ind in source_task1]
    sub_size = len(target_offspring)

    # Apply crossover and mutation on the offspring
    # for Task0
    i = 0
    while i < sub_size:
        if random.random() < cxpb:
            if random.random() <= rmp :
                if random.random() <= 0.5:
                    selected_individual = toolbox.select(source_offspring0, 1)[0]
                    target_offspring[i], _ = toolbox.mate(target_offspring[i],
                                                        toolbox.clone(selected_individual))
                else:
                    selected_individual = toolbox.select(source_offspring1, 1)[0]
                    target_offspring[i], _ = toolbox.mate(target_offspring[i],
                                                        toolbox.clone(selected_individual))
                del target_offspring[i].fitness.values

                i += 1
            elif sub_size - i >= 2:
                target_offspring[i], target_offspring[i + 1] = toolbox.mate(target_offspring[i],
                                                                        target_offspring[i + 1])
                del target_offspring[i].fitness.values, target_offspring[i + 1].fitness.values
                
                i += 2
            
            else:
                target_offspring[i], _ = toolbox.mate(target_offspring[i],
                                                    target_offspring[i - 1])
                del target_offspring[i].fitness.values
                
                i += 1
            
    for i in range(sub_size):
        if random.random() < mutpb:
            target_offspring[i], = toolbox.mutate(target_offspring[i])
            del target_offspring[i].fitness.values

    return target_offspring

def ad_multitask_varAnd(target_task, source_task0, source_task1, toolbox, cxpb, mutpb, rmp, transfer_index, tar_be, sou_be, tournament=0, rate_list=[]):
# def multitask_varAnd(sub_pop0, sub_pop1, toolbox, cxpb, mutpb, rmp):
    target_offspring = [toolbox.clone(ind) for ind in target_task]
    source_offspring0 = [toolbox.clone(ind) for ind in source_task0]
    source_offspring1 = [toolbox.clone(ind) for ind in source_task1]
    source_offspring = [(source_offspring0 + source_offspring1)[i] for i in transfer_index]
    sub_size = len(target_offspring)
    if len(transfer_index) / len(source_offspring0 + source_offspring1) < rmp:
        rmp = len(transfer_index) / len(source_offspring0 + source_offspring1)
    rate_list.append(rmp)
    # Apply crossover and mutation on the offspring
    # for Task0
    i = 0
    while i < sub_size:
        if random.random() < cxpb:
            if random.random() <= rmp and len(transfer_index) != 0 :
                # if tournament == 1:
                #     # random_index = [random.choice(transfer_index) for i in range(14)]
                #     # behavior_matrix = [sou_be[index] for index in random_index]
                #     # mean_distance = distance_behavior(tar_be, behavior_matrix, axis=1)
                #     # min_index = np.argmin(mean_distance)
                #     # index = random_index[min_index]
                # else:
                #     index = random.choice(transfer_index)
                
                # target_offspring[i], _ = toolbox.mate(target_offspring[i],
                #                                     (source_offspring0 + source_offspring1)[index])
                
                transfer_ind = toolbox.select1(source_offspring, 1)[0]
                target_offspring[i], _ = toolbox.mate(target_offspring[i],
                                                    transfer_ind)

                del target_offspring[i].fitness.values

                i += 1
            elif sub_size - i >= 2:
                target_offspring[i], target_offspring[i + 1] = toolbox.mate(target_offspring[i],
                                                                        target_offspring[i + 1])
                del target_offspring[i].fitness.values, target_offspring[i + 1].fitness.values
                
                i += 2
            
            else:
                target_offspring[i], _ = toolbox.mate(target_offspring[i],
                                                    target_offspring[i - 1])
                del target_offspring[i].fitness.values
                
                i += 1
            
    for i in range(sub_size):
        if random.random() < mutpb:
            target_offspring[i], = toolbox.mutate(target_offspring[i])
            del target_offspring[i].fitness.values

    return target_offspring

def bg_multitask_varAnd(target_task, source_task0, source_task1, toolbox, cxpb, mutpb, rmp, tar_be, sou_be0, sou_be1):
# def multitask_varAnd(sub_pop0, sub_pop1, toolbox, cxpb, mutpb, rmp):
    target_offspring = [toolbox.clone(ind) for ind in target_task]
    source_offspring0 = [toolbox.clone(ind) for ind in source_task0]
    source_offspring1 = [toolbox.clone(ind) for ind in source_task1]
    sub_size = len(target_offspring)

    # Apply crossover and mutation on the offspring
    # for Task0
    i = 0
    while i < sub_size:
        if random.random() < cxpb:
            if random.random() <= rmp :
                # if random.random() <= 0.5:
                random_index = np.random.randint(0, len(source_offspring0 + source_offspring1), 7)
                print(len(source_offspring0 + source_offspring1))
                behavior_matrix = [(sou_be0 + sou_be1)[index] for index in random_index]
                
                mean_distance = distance_behavior(tar_be, behavior_matrix)
                min_index = np.argmin(mean_distance)
                index = random_index[min_index]

                target_offspring[i], _ = toolbox.mate(target_offspring[i],
                                                    (source_offspring0 + source_offspring1)[index])
                
                del target_offspring[i].fitness.values
                # else:
                #     random_index = np.random.randint(0, len(source_offspring1), 7)
                #     behavior_matrix = [sou_be1[index] for index in random_index]
                    
                #     mean_distance = distance_behavior(tar_be, behavior_matrix)
                #     min_index = np.argmin(mean_distance)
                #     index = random_index[min_index]

                #     target_offspring[i], _ = toolbox.mate(target_offspring[i],
                #                                         source_offspring1[index])
                    
                #     del target_offspring[i].fitness.values

                i += 1
            elif sub_size - i >= 2:
                target_offspring[i], target_offspring[i + 1] = toolbox.mate(target_offspring[i],
                                                                        target_offspring[i + 1])
                del target_offspring[i].fitness.values, target_offspring[i + 1].fitness.values
                
                i += 2
            
            else:
                target_offspring[i], _ = toolbox.mate(target_offspring[i],
                                                    target_offspring[i - 1])
                del target_offspring[i].fitness.values
                
                i += 1
            
    for i in range(sub_size):
        if random.random() < mutpb:
            target_offspring[i], = toolbox.mutate(target_offspring[i])
            del target_offspring[i].fitness.values

    return target_offspring


def get_weighted_fitness(individuals, weight, *norm):
    """ calculate the tchebycheff scalar function for each individual
    """
    fitness_list = []
    for ind in individuals:
        if weight[0] == 1:
            fitness_list.append(ind.fitness.values[0])
        elif weight[1] == 1:
            fitness_list.append(ind.fitness.values[1])
        else:
            fitness_list.append(max(1 / weight[0] * ((ind.fitness.values[0] - norm[0]) / (norm[1] - norm[0])), 1 / weight[1] * ((ind.fitness.values[1] - norm[2]) / (norm[3] - norm[2]))))
        
    return fitness_list

def weighted_tournament_selection(individuals, au_ind, size, tournsize, weight, *norm):
    chosen = []
    for i in range(size):
        aspirants = [random.choice(individuals) for j in range(tournsize - 3)]
        au_aspirants = [random.choice(au_ind) for j in range(3)]
        aspirants = aspirants + au_aspirants
        fitness_list = get_weighted_fitness(aspirants, weight, *norm)
        best_value = min(fitness_list)
        best_index = fitness_list.index(best_value)
        chosen.append(aspirants[best_index])

    return chosen

def selection_knowledge_sharing(task, auxiliary_task, size, tournsize, min_e, max_e, min_c, max_c, auxiliary):
    """ Select a certain number of individuals for each sub-region
    :param pop: 
    :params min_e, max_e, min_c, max_c: used to normalization
    :params auxiliary: 0, 1: auxiliary task; 2, 3: main task 0, 1
    """
    weight_vectors = [[1 / (tan_value(22.5) + 1), tan_value(22.5) / (tan_value(22.5) + 1)], [1 / (tan_value(67.5) + 1), tan_value(67.5) / (tan_value(67.5) + 1)], [1, 0], [0, 1]]
    sub_offspring = weighted_tournament_selection(task, auxiliary_task, size, tournsize, weight_vectors[auxiliary], min_e, max_e, min_c, max_c)
    # if auxiliary == 0 or auxiliary == 1:
    #     # auxiliary task0 or task1
    #     sub_offspring = weighted_tournament_selection(task, size, tournsize, weight_vectors[auxiliary], min_e, max_e, min_c, max_c)
    # else:
    #     # main task
    #     sub_region = divide_region(auxiliary_task, auxiliary - 2, min_e, max_e, min_c, max_c)
    #     print(len(sub_region))
    #     sub_offspring = weighted_tournament_selection(task + auxiliary_task, size, tournsize, weight_vectors[auxiliary], min_e, max_e, min_c, max_c)
    
    return sub_offspring
    # for i in range(len(sub_region)):
    #     if len(sub_region[i]) >= size_subregion:
    #         # when the subregion has enough number of individuals
    #         sub_offspring = weighted_tournament_selection(sub_region[i], size_subregion, tournsize, weight_vectors[i], min_e, max_e, min_c, max_c)
    #     else:
    #         # when the subregion donot have enough number of individuals
    #         # Then, get indidviduals from neigborhood subregion
    #         sub_offspring0 = weighted_tournament_selection(sub_region[i], len(sub_region[i]), tournsize, weight_vectors[i], min_e, max_e, min_c, max_c)
    #         if i == 0:
    #             if len(sub_region[i + 1]) >= size_subregion - len(sub_region[i]):
    #                 sub_offspring1 = weighted_tournament_selection(sub_region[i + 1], (size_subregion - len(sub_region[i])), tournsize, weight_vectors[4], min_e, max_e, min_c, max_c)
    #             else:
    #                 sub_offspring1 = weighted_tournament_selection(pop, (size_subregion - len(sub_region[i])), tournsize, weight_vectors[4], min_e, max_e, min_c, max_c)
    #         elif i == len(sub_region) - 1:
    #             if len(sub_region[i - 1]) >= size_subregion - len(sub_region[i]):
    #                 sub_offspring1 = weighted_tournament_selection(sub_region[i - 1], (size_subregion - len(sub_region[i])), tournsize, weight_vectors[0], min_e, max_e, min_c, max_c)
    #             else:
    #                 sub_offspring1 = weighted_tournament_selection(pop, (size_subregion - len(sub_region[i])), tournsize, weight_vectors[0], min_e, max_e, min_c, max_c)
    #         else :
    #             if len(sub_region[i - 1]) + len(sub_region[i + 1]) >= size_subregion - len(sub_region[i]):
    #                 sub_offspring1 = weighted_tournament_selection(sub_region[i + 1] + sub_region[i - 1], (size_subregion - len(sub_region[i])), tournsize, weight_vectors[i], min_e, max_e, min_c, max_c)
    #             else:
    #                 sub_offspring1 = weighted_tournament_selection(pop, (size_subregion - len(sub_region[i])), tournsize, weight_vectors[i], min_e, max_e, min_c, max_c)
            
    #         sub_offspring = sub_offspring0 + sub_offspring1

    #     sub_offsprings.append(sub_offspring)

    # return sub_offsprings


