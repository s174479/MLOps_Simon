import pstats
p = pstats.Stats('exercise_files/profiling')
p.strip_dirs().sort_stats("cumulative").print_stats()
