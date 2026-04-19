import os
import subprocess

branches = ['topic-algorithms', 'topic-awg', 'topic-core', 'topic-windows']
files_to_edit = [
    'atommovr/algorithms/source/pcfa.py',
    'atommovr/algorithms/source/tetris.py'
]

for br in branches:
    subprocess.run(['git', 'checkout', br], check=True)
    with open('atommovr/algorithms/source/pcfa.py', 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('described in the PCFA paper and', 'described in the PCFA paper [1]_ and')
    if 'References' not in content:
        content = content.replace('move batches.\n"\"\"', 'move batches.\n\nReferences\n----------\n.. [1] Y. Zhang et al., "A Fast Rearrangement Method for Defect-Free Atom Arrays,"\n       Photonics 12(2), 117 (2025). https://doi.org/10.3390/photonics12020117\n"\"\"')
    with open('atommovr/algorithms/source/pcfa.py', 'w', encoding='utf-8') as f:
        f.write(content)
        
    with open('atommovr/algorithms/source/tetris.py', 'r', encoding='utf-8') as f:
        content2 = f.read()
    content2 = content2.replace('described in Phys. Rev. Applied 19, 054032.', 'described in Wang et al. [1]_.')
    if 'References' not in content2:
        content2 = content2.replace('AtomArray.evaluate_moves`.\n"\"\"', 'AtomArray.evaluate_moves`.\n\nReferences\n----------\n.. [1] S. Wang et al., "Accelerating the Assembly of Defect-Free Atomic Arrays with Maximum Parallelisms,"\n       Phys. Rev. Appl. 19, 054032 (2023). https://doi.org/10.1103/PhysRevApplied.19.054032\n"\"\"')
    with open('atommovr/algorithms/source/tetris.py', 'w', encoding='utf-8') as f:
        f.write(content2)

    subprocess.run(['git', 'add', '-A'], check=True)
    res = subprocess.run(['git', 'commit', '-m', 'docs: update in-code citations for PCFA and Tetris algorithms'], capture_output=True)
