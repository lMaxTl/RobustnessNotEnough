"""Extraction economics of adversarially robust CIFAR-10 classifiers.

Module map (spec.md section numbers in brackets):

    costmodel.py   M1  [S4]  defender-side scaling law, no GPU
    flops.py       M2  [S5]  attacker-side measured FLOPs
    victims.py     M3  [S6]  RobustBench victims + query API
    students.py    M4  [S7]  student architectures, deterministic init
    extract.py     M5  [S8]  query pools + extraction attacks
    evaluate.py    M6  [S9]  clean/AA/PGD-40/fidelity metrics
    margins.py     M7  [S10] FAB-T boundary geometry
    defenses.py    M8  [S11] output perturbation + detection defenses
    schema.py      M9  [S12] results.jsonl record schema
    analysis.py    M11 [S14] tau regression, leverage, bootstrap CIs
    figures.py         [S2]  D5-D8
"""

__version__ = "0.1.0"
