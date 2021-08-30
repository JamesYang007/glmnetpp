import io
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from subprocess import check_output

TESTNAME = 'lasso_stress_benchmark'
ns = np.array([100, 500, 1000, 2000])
ps = 2**(np.linspace(1, 14, 14)).astype(int)

def plot(df, fig_dir):
    grouped = df.groupby('n')
    ncols=2
    nrows = int(np.ceil(grouped.ngroups/ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,6), sharey=True)

    for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
        print(grouped.get_group(key))
        grouped.get_group(key).plot(ax=ax, x='p', y=['glmnet', 'glmnetpp'],
                                    logx=True, logy=True)
        ax.set_xscale('log', base=2)
        ax.set_title('N={n}'.format(n=key))
        ax.legend()
        ax.set_yscale('log', base=2)
        ax.set_ylabel('Time (s)')

    plt.savefig(os.path.join(fig_dir, TESTNAME + '_fig.png'))

# Run benchmark
# bench_dir     directory to glmnetpp benchmark program (e.g. build/release/benchmark)
# data_dir      directory to store our timing data (e.g. docs/data)
# ref_dir       directory to reference (glmnet) program for comparison (e.g. benchmark/reference)
# data_scr_dir  directory to scripts that generate data (e.g. benchmark/data/script)
# gen           boolean whether to generate data or not
def run(bench_dir, data_dir, ref_dir, data_scr_dir, gen):
    df = pd.DataFrame()

    # save current working directory
    cur_path = os.getcwd()

    # generate all the data if gen is true
    if gen:
        os.chdir(data_scr_dir)
        for n in ns:
            for p in ps:
                gen_script = 'gen_random_unif.py'
                args = ('python3', gen_script, '-n', str(n), '-p', str(p))
                check_output(args)
        os.chdir(cur_path)

    # change directory to glmnetpp benchmark location
    os.chdir(bench_dir)

    bench_path = os.path.join('.', TESTNAME)
    print('Benchmark path: {p}'.format(p=bench_path))

    # run our benchmark and get output
    args = (bench_path, "--benchmark_format=csv")
    data = io.StringIO(check_output(args).decode("utf-8"))
    df_bench = pd.read_csv(data, sep=',')
    df = df_bench[['p', 'n']]
    df['glmnetpp'] = df_bench['real_time'] * 1e-9
    df.set_index(['p', 'n'], inplace=True)

    os.chdir(cur_path)

    # now run R script using glmnet
    os.chdir(ref_dir)
    ref_name = TESTNAME + '.R'
    args = ('Rscript', ref_name)
    data = io.StringIO(check_output(args).decode("utf-8"))
    df_bench = pd.read_csv(data, sep=',', header=None)
    df_bench.columns = ['glmnet', 'p', 'n']
    df_bench.set_index(['p', 'n'], inplace=True)
    df_bench['glmnet'] *= 1e-9
    df = pd.concat([df, df_bench], axis=1)
    df['relative'] = df['glmnet'] / df['glmnetpp']

    df.reset_index(inplace=True)

    # save absolute time
    data_path = os.path.join(data_dir, TESTNAME + ".csv")
    df.to_csv(data_path)

    return df

if __name__ == '__main__':
    import argparse
    import path_names

    parser = argparse.ArgumentParser(description='Runs a benchmark against glmnet on uniform(-1,1) data.')
    parser.add_argument('-g', action='store_const', const=True, default=False,
                        help='Generate data if set.')
    args = parser.parse_args()

    df = run(path_names.bench_dir,
             path_names.data_dir,
             path_names.ref_dir,
             path_names.data_scr_dir,
             gen=args.g)
    plot(df, path_names.fig_dir)
