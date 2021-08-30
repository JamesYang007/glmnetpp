import io
import os
import matplotlib.pyplot as plt
import pandas as pd
from subprocess import check_output

TESTNAME = 'set_vs_vector_loop'

# Plot result of running benchmark
def plot(df, fig_dir):
    axes = df.plot(x='size',
                   kind='line',
                   marker='.',
                   xticks=df['size'],
                   xlabel='size',
                   ylabel='Time (ms)',
                   title=TESTNAME,
                   figsize=(8,6))
    axes.set_xscale('log', base=2)
    axes.set_yscale('log', base=2)
    plt.savefig(os.path.join(fig_dir, TESTNAME + '_fig.png'))

# Run benchmark for set_vs_vector_loop
def run(bench_dir, data_dir, ref_dir='', data_scr_dir=''):
    df = pd.DataFrame()

    # save current working directory
    cur_path = os.getcwd()

    # change directory to benchmark location
    os.chdir(bench_dir)

    bench_path = os.path.join('.', TESTNAME)
    print(bench_path)

    # run and get output from each
    args = (bench_path, "--benchmark_format=csv")
    data = io.StringIO(check_output(args).decode("utf-8"))
    df_bench = pd.read_csv(data, sep=',')
    df_set, df_vec = df_bench[df_bench['type'] == 0], \
                     df_bench[df_bench['type'] == 1]
    df_set.set_index('size', inplace=True)
    df_vec.set_index('size', inplace=True)

    def parse_df(df_type, df, tp):
        sep_sizes = df_type['sep_size']
        grouped = df_type.groupby('sep_size')
        for sep_size in sep_sizes:
            name = 'sep_size={sep_size},type={tp}'.format(
                sep_size=sep_size, tp=tp
            )
            df[name] = grouped.get_group(sep_size)['real_time'] # a profile over size
            df[name] *= 1e-6 # in milliseconds

    parse_df(df_set, df, 'set')
    parse_df(df_vec, df, 'vec')

    # change back to current working directory
    os.chdir(cur_path)

    # save absolute time
    data_path = os.path.join(data_dir, TESTNAME + ".csv")
    df.to_csv(data_path)

    df.reset_index(level=df.index.names, inplace=True)

    return df
