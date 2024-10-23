"""
This is to extract force and position from GPUMD-produced trajectory
as the input of TDEP.
The script is written by XIANG Xing (HKUST).
"""
from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms

a = []
file = '/media/mee2123/data/CsAgBiBr/tdep_nep/2_aimd/1_md/dump.xyz'
temp = read(file,index=":")
for j in temp:
    a.append(j)

file_position = 'infile.positions'
file_forces = 'infile.forces'

file_pos = open(file_position,'w')
for i in a:
    temp = i.get_scaled_positions()
    for j in temp:
        print("{:>17.12f} {:>17.12f} {:>17.12f}".format(j[0],j[1],j[2]),file=file_pos)
file_pos.close()

file_for = open(file_forces,'w')
for i in a:
    temp = i.get_forces()
    for j in temp:
        print("{:>17.12f} {:>17.12f} {:>17.12f}".format(j[0],j[1],j[2]),file=file_for)
file_for.close()

#write("POSCAR", temp[9], direct=True, sort=True)
print(len(a))

