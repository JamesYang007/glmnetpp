#include <vector>
#include <set>
#include <benchmark/benchmark.h>

namespace {

/*
 * Benchmark to see whether iterating through a set of active indices
 * is faster than iterating a vector of (longer and potentially sparse) booleans
 * and applying the same routine when a position is true.
 */

struct set_vs_vector_fixture : benchmark::Fixture
{
    std::set<int> s;
    std::vector<bool> v;
    int dummy = 0;

    // size is the total number of entries in each data structure
    // sep_size is how much to separate the elements in v
    // every i * sep_size is considered an active index for the purposes of this benchmark.
    void populate_set(int size, int sep_size)
    {
        for (int i = 0; i < size; ++i) {
            s.emplace(i * sep_size);
        }
    }

    void populate_vec(int size, int sep_size)
    {
        v.resize(size * sep_size, false);
        for (int i = 0; i < size; ++i) {
            v[i*sep_size] = true;
        }
    }

    void routine_set()
    {
        for (auto i : s) {
            dummy += i; // dummy routine with signal i 
        }
    }

    void routine_vec()
    {
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i]) dummy += i; 
        }
    }
};

BENCHMARK_DEFINE_F(set_vs_vector_fixture,
                    set_loop)(benchmark::State& state)
{
    int size = state.range(0);
    int sep_size = state.range(1);
    populate_set(size, sep_size);

    state.counters["size"] = size;
    state.counters["sep_size"] = sep_size;
    state.counters["type"] = 0;
    for (auto _ : state) {
        routine_set();
    }
}

BENCHMARK_DEFINE_F(set_vs_vector_fixture,
                    vec_loop)(benchmark::State& state)
{
    int size = state.range(0);
    int sep_size = state.range(1);
    populate_vec(size, sep_size);

    state.counters["size"] = size;
    state.counters["sep_size"] = sep_size;
    state.counters["type"] = 1;
    for (auto _ : state) {
        routine_vec();
    }
}

BENCHMARK_REGISTER_F(set_vs_vector_fixture,
                     set_loop)
    -> ArgsProduct({
        {1<<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20}, 
        {1, 1<<5, 1<<7, 1<<10}
    })
    ;

BENCHMARK_REGISTER_F(set_vs_vector_fixture,
                     vec_loop)
    -> ArgsProduct({
        {1<<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20}, 
        {1, 1<<5, 1<<7, 1<<10}
    })
    ;

} 
