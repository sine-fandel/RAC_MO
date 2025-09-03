import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import re
import json
import ast
import argparse
from datetime import datetime

import numpy as np

from deap import base, gp
from deap import tools, creator

from optim.multi_tree_gp import MultiPrimitiveTree
from utils.utils import testing_simulation, get_config
from env.simulator.code.simulator import Simulator, SimulatorState


def protectedDiv(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def build_pset(config):
    """Build primitive sets for vm and pm identical to solve_mogp."""
    arity0 = config["arity0"]
    arity1 = config["arity1"]

    pset = {"vm": None, "pm": None}
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

    for t in ("vm", "pm"):
        if t == "vm":
            pset[t] = gp.PrimitiveSet(t, arity0)
            pset[t].renameArguments(**TERMINAL_NODES["vm"])
        else:
            pset[t] = gp.PrimitiveSet(t, arity1)
            pset[t].renameArguments(**TERMINAL_NODES["pm"])
        pset[t].addPrimitive(np.add, 2)
        pset[t].addPrimitive(np.subtract, 2)
        pset[t].addPrimitive(np.multiply, 2)
        pset[t].addPrimitive(protectedDiv, 2)

    return pset


def eval_individual(
    individual,
    sim_state: SimulatorState,
    containers,
    os,
    applications_reverse: dict,
    toolbox: base.Toolbox,
    pset,
    test=True,
):
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

        sim.update_current_communication(app)
        if sim.current_timestamp != current_timestamp:
            previous_timestamp = current_timestamp
            current_timestamp = sim.current_timestamp
            sim.update_total_communication(previous_timestamp, current_timestamp)

    return sim.running_energy_consumption, sim.running_communication_overhead


def filter_nondominated_entries(entries):
    """Return only nondominated entries (first front) based on energy and communication.

    entries: list of dicts each containing keys 'energy' and 'communication'.
    Returns a tuple (filtered_entries, kept_indices) where kept_indices are
    the original positions in the input list that belong to the first front.
    """
    if not entries:
        return entries, []

    # Ensure Fitness class exists (minimization for both objectives)
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))

    # Lightweight wrapper to attach DEAP fitness for nondominated sorting
    class _Wrap:
        __slots__ = ("idx", "fitness")

        def __init__(self, idx, energy, comm):
            self.idx = idx
            self.fitness = creator.FitnessMulti()
            self.fitness.values = (energy, comm)

    wraps = []
    for idx, e in enumerate(entries):
        # Guard against missing keys by treating them as +inf (always dominated)
        energy = e.get("energy", float("inf"))
        comm = e.get("communication", float("inf"))
        wraps.append(_Wrap(idx, energy, comm))

    fronts = tools.sortNondominated(wraps, k=len(wraps), first_front_only=True)
    kept = {w.idx for w in fronts[0]}

    # Preserve original order among kept entries
    filtered = [entries[i] for i in range(len(entries)) if i in kept]
    kept_indices = [i for i in range(len(entries)) if i in kept]
    return filtered, kept_indices


def parse_run_from_filename(fname: str):
    # expects pattern like nsw_Bitbrains_3OS_NSGP2_{run}_core_{cpu}.json
    m = re.search(r"NSGP2_(\d+)_core_(\d+)\.json$", fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    # fallback
    return None, None


def get_output_file(cpu_num: int, run: int = None, gen: int = None):
    """Build output path under results/testing and ensure file exists with schema.

    Parent folder format: {YYYYMMDD-HHMM}. If the folder already exists, reuse it.

    Returns: (out_file_path, single_run_mode: bool)
    """
    base_dir = os.path.join(os.getcwd(), "results", "testing")
    # Hardcoded parent folder format: {YYYYMMDD-HHMM}
    ts_folder = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = os.path.join(base_dir, ts_folder)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    gen_suffix = f"_gen_{gen}" if gen is not None else ""

    if run is not None:
        # single-run mode file
        out_file = os.path.join(
            out_dir,
            f"nsw_Bitbrains_3OS_NSGP2_core_{cpu_num}_run_{run}{gen_suffix}.json",
        )
        if not os.path.exists(out_file):
            with open(out_file, "w") as f:
                json.dump({"sub_population_size": None, "generation": {}}, f)
        return out_file, True
    else:
        # multi-run aggregate file
        out_file = os.path.join(
            out_dir,
            f"aggregate_nsw_Bitbrains_3OS_NSGP2_core_{cpu_num}{gen_suffix}.json",
        )
        if not os.path.exists(out_file):
            with open(out_file, "w") as f:
                json.dump({"runs": {}}, f)
        return out_file, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_num", type=int, default=0, help="Test case id under Test/ dirs"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run index to include (default: all runs)",
    )
    parser.add_argument(
        "--hide_expr",
        action="store_true",
        help="Do not include the expression string for each individual in the output JSON",
    )
    parser.add_argument(
        "--gen",
        type=int,
        default=None,
        help="Only evaluate the specified generation index (default: all)",
    )
    args = parser.parse_args()

    config = get_config(path="./config/communication_mincut_gp.yaml")
    pset = build_pset(config)

    # Prepare testing data once (shared across runs/generations)
    sim_state, input_containers, applications, applications_reverse, input_os = (
        testing_simulation(
            test_num=args.test_num, start=args.start, end=args.end, run=0
        )
    )

    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset)

    # Output path and init progressive file
    out_file, single_run_mode = get_output_file(
        cpu_num=config["cpu_num"], run=args.run, gen=args.gen
    )

    train_dir = os.path.join(os.getcwd(), "results", "training")
    for fname in sorted(os.listdir(train_dir)):
        if not fname.endswith(".json"):
            continue
        if "NSGP2" not in fname:
            continue
        run_idx, cpu_num = parse_run_from_filename(fname)
        if args.run is not None:
            # If a specific run is requested, skip files that don't match
            if run_idx is None or run_idx != args.run:
                continue

        with open(os.path.join(train_dir, fname), "r") as f:
            train_data = json.load(f)

        run_key = str(run_idx) if run_idx is not None else fname
        print(f"==> Evaluating run {run_key} from {fname}")
        # Ensure container for this run exists in output (or root in single-run mode)
        with open(out_file, "r") as f:
            agg = json.load(f)
        if single_run_mode:
            # single-run: root holds sub_population_size and generation
            if agg.get("sub_population_size") is None:
                agg["sub_population_size"] = train_data.get("sub_population_size")
        else:
            if run_key not in agg["runs"]:
                agg["runs"][run_key] = {
                    "sub_population_size": train_data.get("sub_population_size"),
                    "generation": {},
                }
            else:
                agg["runs"][run_key].setdefault(
                    "sub_population_size", train_data.get("sub_population_size")
                )
        with open(out_file, "w") as f:
            json.dump(agg, f)

        generations = train_data.get("generation", {})
        for gen_str, gen_payload in generations.items():
            # Filter by specific generation if requested
            if args.gen is not None:
                try:
                    if int(gen_str) != args.gen:
                        continue
                except ValueError:
                    # Non-numeric generation keys are skipped when filtering
                    continue
            best_list = gen_payload.get("best", [])
            front = []
            parsed_exprs = []
            for expr_str in best_list:
                # Parse dict-like string safely to avoid calling overridden eval()
                try:
                    expr_dict = ast.literal_eval(expr_str)
                except Exception:
                    # Skip malformed entry
                    print(
                        f"[Run {run_key} Gen {gen_str}] Skipping malformed individual expr."
                    )
                    continue
                mtree = MultiPrimitiveTree.from_string(expr_dict, pset)
                front.append(mtree)
                parsed_exprs.append(expr_str)

            print(
                f"[Run {run_key} Gen {gen_str}] Evaluating {len(front)} individuals on test data..."
            )

            # Progressive write: evaluate and write each individual
            # Ensure generation entry exists with an array to append into
            with open(out_file, "r") as f:
                agg = json.load(f)
            if single_run_mode:
                agg.setdefault("generation", {})
                agg["generation"].setdefault(
                    gen_str, {"front": [], "front_size": len(parsed_exprs)}
                )
                if "front_size" not in agg["generation"][gen_str]:
                    agg["generation"][gen_str]["front_size"] = len(parsed_exprs)
            else:
                agg.setdefault("runs", {})
                agg["runs"].setdefault(run_key, {})
                agg["runs"][run_key].setdefault("generation", {})
                agg["runs"][run_key]["generation"].setdefault(
                    gen_str, {"front": [], "front_size": len(parsed_exprs)}
                )
                if "front_size" not in agg["runs"][run_key]["generation"][gen_str]:
                    agg["runs"][run_key]["generation"][gen_str]["front_size"] = len(
                        parsed_exprs
                    )
            with open(out_file, "w") as f:
                json.dump(agg, f)

            for ind, expr_str in zip(front, parsed_exprs):
                energy, comm = eval_individual(
                    ind,
                    sim_state,
                    input_containers,
                    input_os,
                    applications_reverse,
                    toolbox,
                    pset,
                    test=True,
                )
                # Load, append and write immediately
                with open(out_file, "r") as f:
                    agg = json.load(f)
                if single_run_mode:
                    existing = agg["generation"][gen_str]["front"]
                else:
                    existing = agg["runs"][run_key]["generation"][gen_str]["front"]
                next_idx = len(existing)
                entry = {
                    "index": next_idx,
                    "energy": round(energy, 2),
                    "communication": round(comm, 2),
                }
                if not args.hide_expr:
                    entry["expr"] = expr_str
                existing.append(entry)
                with open(out_file, "w") as f:
                    json.dump(agg, f)
                print(
                    f"[Run {run_key} Gen {gen_str}] Front {next_idx}: energy={energy:.2f}, communication={comm:.2f} (written)"
                )
            print(
                f"[Run {run_key} Gen {gen_str}] Saved {len(parsed_exprs)} results to {out_file}"
            )

            # After writing all entries for this generation, mark dominated entries
            with open(out_file, "r") as f:
                agg = json.load(f)
            if single_run_mode:
                gen_node = agg.setdefault("generation", {}).setdefault(gen_str, {})
            else:
                gen_node = (
                    agg.setdefault("runs", {})
                    .setdefault(run_key, {})
                    .setdefault("generation", {})
                    .setdefault(gen_str, {})
                )
            original = gen_node.get("front", [])
            _, kept_idx = filter_nondominated_entries(original)
            kept_set = set(kept_idx)
            # Annotate dominated entries, preserve all
            for i, entry in enumerate(original):
                if i not in kept_set:
                    entry["dominated"] = True
                else:
                    entry["dominated"] = False
            # Record sizes and kept indices
            gen_node["first_front_size"] = len(kept_idx)
            with open(out_file, "w") as f:
                json.dump(agg, f)
            print(
                f"[Run {run_key} Gen {gen_str}] Marked dominated entries; first-front size: {len(kept_idx)} of {len(original)}"
            )

    print(f"Testing results written to {out_file}")


if __name__ == "__main__":
    main()

