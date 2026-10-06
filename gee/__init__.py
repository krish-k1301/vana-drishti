"""Earth Engine export pipeline for the dated early-detection set (PRD Phases 3 and 6).

Pure-Python modules (dates, grid, sampling, decode, to_bradd, qa, tasklog, vectors) never touch Earth Engine.
Earth Engine builders (s1, labels, stack, export) import `ee` and are unit-tested against mocks.
"""
