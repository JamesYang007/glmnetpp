# A simple script to run benchmark programs and visualize the data.
# Assumes that the benchmark program has already been built.

import argparse
import matplotlib.pyplot as plt
import path_names
import analyze_set_vs_vector_loop as asvvl
import analyze_lasso_stress_benchmark as alsb

parser = argparse.ArgumentParser(description='Collects data and produces plots of benchmark programs.')
parser.add_argument('bench_names', nargs='*',
                    help='list of benchmark program names to analyze.')
parser.add_argument('-a', action='store_const', const=True,
                    help='analyze all benchmark programs in build/release/benchmark.')
args = parser.parse_args()

if len(args.bench_names) == 0 and not args.a:
    raise RuntimeError(
        'At least one benchmark name must be specified if -a is not specified.')

# Dictionary of bench name to module name
bench_to_module = {
    asvvl.TESTNAME : asvvl,
    alsb.TESTNAME : alsb
}

mods = [bench_to_module[bench_name] for bench_name in args.bench_names]
for mod in mods:
    mod.plot(mod.run(path_names.bench_dir,
                     path_names.data_dir,
                     path_names.ref_dir,
                     path_names.data_scr_path),
             path_names.fig_path)
