import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import re
import json
import ast
import argparse
import time
from datetime import datetime

import numpy as np

from deap import base, gp
from deap import tools, creator

from optim.multi_tree_gp import MultiPrimitiveTree
from utils.utils import testing_simulation, get_config
from env.simulator.code.simulator import Simulator, SimulatorState


# Unified decimal precision for rounding, aligned with solve_mogp/solve_moead_pymoo
DECIMAL_PRECISION = 6

DIR = "POP_1024_GEN_70"


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
    time_list = []
    for app, clist in applications_reverse.items():
        container_clusters, P, P_container_id = app.min_cut(
            containers, os, 24000, 20000
        )

        num = 0
        for row in container_clusters.iterrows():
            start_time = time.time()
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
            end_time = time.time()
            time_list.append((end_time - start_time) * 1000)
            # print(len(time_list))

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
    m = re.search(r"NSGP2_(\d+)_core_40.json$", fname)
    if m:
        return int(m.group(1))
    # fallback
    return None, None


def get_output_file(num_test: int, run: int=None, gen: int=None, OS: str=None, multi_gen: bool=False, end: int=None):
    """Build output path under results/testing and ensure file exists with schema.

    Parent folder format: {YYYYMMDD-HHMM}. If the folder already exists, reuse it.

    Returns: (out_file_path, single_run_mode: bool)
    """
    output_dir = os.path.dirname(os.path.abspath(__file__));

    if multi_gen:
        base_dir = os.path.join(output_dir, "output", "testing", "convergence", f"TEST_{num_test}", DIR)
    else:
        base_dir = os.path.join(output_dir, "output", "testing", DIR)
    # Hardcoded parent folder format: {YYYYMMDD-HHMM}
    ts_folder = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = base_dir #os.path.join(base_dir, ts_folder)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    gen_suffix = f"_gen_{gen}_{end}" if gen is not None else ""

    if run is not None:
        # single-run mode file
        out_file = os.path.join(
            out_dir,
            f"Bitbrains_{OS}_NSGP2_run_{run}{gen_suffix}.json",
        )
        if not os.path.exists(out_file):
            with open(out_file, "w") as f:
                json.dump({"sub_population_size": None, "generation": {}}, f)
        return out_file, True
    else:
        # multi-run aggregate file
        out_file = os.path.join(
            out_dir,
            f"aggregate_Bitbrains_{OS}_NSGP2{gen_suffix}.json",
        )
        if not os.path.exists(out_file):
            with open(out_file, "w") as f:
                json.dump({"runs": {}}, f)
        return out_file, False


def build_test_mogp_parser() -> argparse.ArgumentParser:
    """Build the ArgumentParser for this test script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained NSGP2 individuals on test data"
    )
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
    parser.add_argument(
        "--multi_gen",
        type=int,
        nargs='+',
        default=list,
        help="Evaluate multiple generations (default: all)",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help=(
            "Required: folder to read training JSONs from. "
            "Pass a timestamp folder under results/training (e.g. 20250904-1956) "
            "or an absolute/relative path."
        ),
    )
    return parser


def parse_test_mogp_args(argv=None):
    return build_test_mogp_parser().parse_args(argv)


def resolve_train_dir(arg_value: str | None) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__));
    base_train_root = os.path.join(base_dir, "output", "training", DIR)
    if arg_value is None:
        raise SystemExit("Error: --train_dir is required. Example: --train_dir 20250904-1956")
    if os.path.isabs(arg_value) or os.path.sep in arg_value:
        train_dir = arg_value
    else:
        train_dir = os.path.join(base_train_root, arg_value)
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    return train_dir


def iter_training_runs(train_dir: str, run_filter: int | None):
    for fname in sorted(os.listdir(train_dir)):
        if not fname.endswith(".json"):
            continue
        if "NSGP2" not in fname:
            continue
        run_idx = parse_run_from_filename(fname)
        if run_filter is not None and (run_idx is None or run_idx != run_filter):
            continue
        path = os.path.join(train_dir, fname)
        with open(path, "r") as f:
            train_data = json.load(f)
        run_key = str(run_idx) if run_idx is not None else fname
        yield fname, run_key, train_data


def ensure_output_root(out_file: str, single_run_mode: bool, run_key: str, train_data: dict, gen: str, font_size: int):
    with open(out_file, "r") as f:
        agg = json.load(f)
    if single_run_mode:
        if agg.get("sub_population_size") is None:
            agg["sub_population_size"] = train_data.get("sub_population_size")
    else:
        agg.setdefault("runs", {})
        if run_key not in agg["runs"]:
            agg["runs"][run_key] = {
                "sub_population_size": train_data.get("sub_population_size"),
                "generation": {
                                gen: {
                                    "front": [],
                                    "front_size": font_size,
                                }
                            },
            }
        else:
            agg["runs"][run_key].setdefault(
                "sub_population_size", train_data.get("sub_population_size")
            )
    with open(out_file, "w") as f:
        json.dump(agg, f)


def extract_front_exprs(gen_payload: dict) -> list[str]:
    """Extract expression strings from the assumed first_front structure.

    Assumes training JSON uses: first_front: [{"communication", "energy", "expr"}, ...]
    Returns a list of expr strings. Entries missing "expr" are ignored.
    """
    first_front = gen_payload.get("first_front", [])
    exprs: list[str] = []
    for e in first_front:
        if isinstance(e, dict) and e.get("expr") is not None:
            exprs.append(e["expr"])
    return exprs


def ensure_generation_slot(out_file: str, single_run_mode: bool, run_key: str, gen_str: str, size: int):
    with open(out_file, "r") as f:
        agg = json.load(f)
    if single_run_mode:
        agg.setdefault("generation", {})
        agg["generation"].setdefault(gen_str, {"front": [], "front_size": size})
        if "front_size" not in agg["generation"][gen_str]:
            agg["generation"][gen_str]["front_size"] = size
    else:
        agg.setdefault("runs", {})
        agg["runs"].setdefault(run_key, {})
        agg["runs"][run_key].setdefault("generation", {})
        agg["runs"][run_key]["generation"].setdefault(
            gen_str, {"front": [], "front_size": size}
        )
        if "front_size" not in agg["runs"][run_key]["generation"][gen_str]:
            agg["runs"][run_key]["generation"][gen_str]["front_size"] = size
    with open(out_file, "w") as f:
        json.dump(agg, f)


def append_eval_result(out_file: str, single_run_mode: bool, run_key: str, gen_str: str, entry: dict) -> int:
    with open(out_file, "r") as f:
        agg = json.load(f)
    if single_run_mode:
        existing = agg["generation"][gen_str]["front"]
    else:
        existing = agg["runs"][run_key]["generation"][gen_str]["front"]
    next_idx = len(existing)
    entry = {**entry, "index": next_idx}
    existing.append(entry)
    with open(out_file, "w") as f:
        json.dump(agg, f)
    return next_idx


def annotate_dominated(out_file: str, single_run_mode: bool, run_key: str, gen_str: str) -> tuple[int, int]:
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
    for i, it in enumerate(original):
        it["dominated"] = i not in kept_set
    gen_node["first_front_size"] = len(kept_idx)
    with open(out_file, "w") as f:
        json.dump(agg, f)
    return len(kept_idx), len(original)


def add_normalized_fields(
    out_file: str, single_run_mode: bool, run_key: str, gen_str: str
) -> None:
    """Add min-max normalized fields for energy and communication to the stored results.

    Normalization is performed over all entries in the generation's stored 'front'.
    If all values are equal, normalized values default to 0.0.
    """
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
    entries = gen_node.get("front", [])
    if not entries:
        with open(out_file, "w") as f:
            json.dump(agg, f)
        return

    energies = [e.get("energy") for e in entries if isinstance(e.get("energy"), (int, float))]
    comms = [e.get("communication") for e in entries if isinstance(e.get("communication"), (int, float))]

    if energies:
        e_min, e_max = min(energies), max(energies)
    else:
        e_min = e_max = 0.0
    if comms:
        c_min, c_max = min(comms), max(comms)
    else:
        c_min = c_max = 0.0

    def _norm(v, vmin, vmax):
        try:
            return 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
        except Exception:
            return 0.0

    for it in entries:
        e_val = it.get("energy")
        c_val = it.get("communication")
        if isinstance(e_val, (int, float)):
            it["energy_norm"] = round(
                _norm(e_val, e_min, e_max), DECIMAL_PRECISION
            )
        if isinstance(c_val, (int, float)):
            it["communication_norm"] = round(
                _norm(c_val, c_min, c_max), DECIMAL_PRECISION
            )

    with open(out_file, "w") as f:
        json.dump(agg, f)

def test_single_run(args, fname: str, run_key: str, train_data: dict,
                    out_file: str, single_run_mode: bool,
                    sim_state: SimulatorState,
                    input_containers,
                    input_os,
                    applications_reverse,
                    toolbox,
                    pset,
                    gen
                    ):
    print(f"==> Evaluating run {run_key} from {fname}")
    generations = train_data.get("generation", {})

    # save json entries
    entries = []
    for gen_str, gen_payload in generations.items():
        if gen is not None:
            try:
                if int(gen_str) != gen:
                    continue
            except ValueError:
                continue

        parsed_exprs = extract_front_exprs(gen_payload)
        front = []
        for expr_str in parsed_exprs:
            # Parse dict-like string safely
            try:
                expr_dict = ast.literal_eval(expr_str)
            except Exception:
                print(f"[Run {run_key} Gen {gen_str}] Skipping malformed individual expr.")
                continue
            mtree = MultiPrimitiveTree.from_string(expr_dict, pset)
            front.append((mtree, expr_str))

        ensure_output_root(out_file, single_run_mode, run_key, train_data, str(gen), len(front))
        print(f"[Run {run_key} Gen {gen_str}] Evaluating {len(front)} individuals on test data...")


        for idx, (ind, expr_str) in enumerate(front):
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
            entry = {
                "energy": round(energy, DECIMAL_PRECISION),
                "communication": round(comm, DECIMAL_PRECISION),
            }
            if not args.hide_expr:
                entry["expr"] = expr_str
            entries.append(entry)
            # idx = append_eval_result(out_file, single_run_mode, run_key, gen_str, entry)
            print(
                f"[Run {run_key} Gen {gen_str}] Front {idx}: energy={energy:.2f}, communication={comm:.2f} (written)"
            )

        
    print(f"==> Finishe run {run_key} from {fname}")
    
    return {run_key: entries}


def main():
    from multiprocessing import Pool

    args = parse_test_mogp_args()

    config = get_config(path="./config/communication_mincut_gp.yaml")
    pset = build_pset(config)

    if args.gen != None:
        gen_list = [args.gen]
        multi_gen = False
    else:
        gen_list = args.multi_gen
        multi_gen = True
    # Prepare testing data once (shared across runs/generations)
    for gen in gen_list:
        sim_state, input_containers, applications, applications_reverse, input_os = (
            testing_simulation(
                test_num=args.test_num, start=args.start, end=args.end, run=0, os_dataset=args.train_dir
            )
        )

        toolbox = base.Toolbox()
        toolbox.register("compile", gp.compile, pset=pset)

        # Output path and init progressive file
        out_file, single_run_mode = get_output_file(
            num_test=args.end, run=args.run, gen=gen, OS=args.train_dir, multi_gen=multi_gen, end=args.end
        )

        train_dir = resolve_train_dir(args.train_dir)
        print(f"Reading training JSONs from: {train_dir}")
        param_list = []

        for fname, run_key, train_data in iter_training_runs(train_dir, args.run):
            param_list.append((args, fname, run_key, train_data,
                            out_file, single_run_mode,
                            sim_state,
                            input_containers,
                            input_os,
                            applications_reverse,
                            toolbox,
                            pset,
                            gen
                            ))

        with Pool(8) as pool:
            results = pool.starmap(test_single_run, param_list)

        for res in results:
            [(run_key, entries)] = res.items()
            for entry in entries:
                append_eval_result(out_file, single_run_mode, run_key, str(gen), entry)
            
            kept, total = annotate_dominated(out_file, single_run_mode, run_key, str(gen))
            # Add normalized fields for the evaluated front of this generation
            add_normalized_fields(out_file, single_run_mode, run_key, str(gen))
            print(f"[Run {run_key} Gen {str(gen)}] Marked dominated entries; first-front size: {kept} of {total}")


    print(f"Testing results written to {out_file}")


if __name__ == "__main__":
    main()
