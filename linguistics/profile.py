#!/usr/bin/python
import pstats

p = pstats.Stats('stats')
p.strip_dirs().sort_stats('tottime').print_stats()
