"""
This is a script to rotate MA molecues in MAPbBr3.
It is written by Xiang xing (HKUST), and improved by He qinqin (HKUST).
"""

from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
import random
import math
import heapq
import copy

# generate rorating matrix randomly 
def roration_matrix(angles):
	degree_x = random.choice(angles)
	angle_x = degree_x / 180 * np.pi # set rotation angles and change the unit
	degree_y = random.choice(angles)
	angle_y = degree_y / 180 * np.pi
	degree_z = random.choice(angles)
	angle_z = degree_z / 180 * np.pi
	
	roration_angle = np.array([degree_x,degree_y,degree_z])
	
	# Rotation matrix
	R11 = math.cos(angle_y) * math.cos(angle_z)
	R12 = math.cos(angle_y) * math.sin(angle_z)
	R13 = -1*math.sin(angle_y)
#	R21 = -1 * math.cos(angle_x) * math.sin(angle_z) + math.sin(angle_x) * math.sin(angle_y) * math.sin(angle_z)
	R21 = -1 * math.cos(angle_x) * math.sin(angle_z) + math.sin(angle_x) * math.sin(angle_y) * math.cos(angle_z)
	R22 = math.cos(angle_x) * math.cos(angle_z) + math.sin(angle_x) * math.sin(angle_y) * math.sin(angle_z)
	R23 = math.sin(angle_x) * math.cos(angle_y)
	R31 = math.sin(angle_x) * math.sin(angle_z) + math.cos(angle_x) * math.sin(angle_y) * math.cos(angle_z)
	R32 = -1 * math.sin(angle_x) * math.cos(angle_z) + math.cos(angle_x) * math.sin(angle_y) * math.sin(angle_z)
	R33 = math.cos(angle_x) * math.cos(angle_y)

	R = np.array([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])

	return roration_angle, R

## set_parameters

angles = [22.5,45,67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5,30,60,120,150,210,240,270,300,330] # rotation angles

#angles = [90,90]

# rotate MA molecules
# 1 find MA molecules
a = read('../POSCAR')  #后面换成AIMD选出来的结构
a_rotated = copy.deepcopy(a)
bonds = a.get_all_distances(mic=True)
La, Lb, Lc = a.cell
La_norm = np.sqrt(sum(La**2))
Lb_norm = np.sqrt(sum(Lb**2))
Lc_norm = np.sqrt(sum(Lc**2))

id_C = []
id_N = []
id_H = []
for i in range(len(a)):
	if a[i].symbol == 'C':
		id_C.append(i)
	if a[i].symbol == 'N':
		id_N.append(i)
	if a[i].symbol == 'H':
		id_H.append(i)

MAs = np.zeros([len(id_C),8]) # the atomic id for one MA is C, H, N, H in order.

for i in range(len(id_C)):
	MAs[i,0] = id_C[i]

	dis_C_H = list(bonds[id_C[i],id_H])
	C_H_bonds = heapq.nsmallest(3, dis_C_H)
	bonded_H_temp = list(map(dis_C_H.index, C_H_bonds))
	MAs[i,1] = id_H[bonded_H_temp[0]]
	MAs[i,2] = id_H[bonded_H_temp[1]]
	MAs[i,3] = id_H[bonded_H_temp[2]]   ##find the nearest H around C atoms(ID)

	dis_C_N = list(bonds[id_C[i],id_N])
	C_N_bonds = heapq.nsmallest(1, dis_C_N)
	bonded_N_temp = list(map(dis_C_N.index, C_N_bonds))
	MAs[i,4] = id_N[bonded_N_temp[0]]

	dis_N_H = list(bonds[int(MAs[i,4]),id_H])
	N_H_bonds = heapq.nsmallest(3, dis_N_H)
	bonded_H_temp = list(map(dis_N_H.index, N_H_bonds))
	print(bonded_H_temp)
	MAs[i,5] = id_H[bonded_H_temp[0]]
	MAs[i,6] = id_H[bonded_H_temp[1]]
	MAs[i,7] = id_H[bonded_H_temp[2]]

MAs = np.array(MAs, dtype=int)

# rotate MAs with the center of C-N bond as the rotating dot
for i in range(len(MAs)):
	R_angle, R_matrix = roration_matrix(angles)
	print("Rotation degrees along x, y and z for " + str(i+1) +" MA" )
	print(R_angle)
	print("Rotation matrix for " + str(i) +" MA" )
	print(R_matrix)

	# find rotating dot and regard it as original point to change atomic position
	temp_MA = MAs[i]
	temp_C = temp_MA[0]
	temp_H1 = temp_MA[1]
	temp_H2 = temp_MA[2]
	temp_H3 = temp_MA[3]
	temp_N = temp_MA[4]
	temp_H4 = temp_MA[5]
	temp_H5 = temp_MA[6]
	temp_H6 = temp_MA[7]
	
	## displace atoms to remove effect of periodical boundary
	P_C = a[temp_C].position
	P_H1 = a[temp_H1].position
	P_H2 = a[temp_H2].position
	P_H3 = a[temp_H3].position
	P_N = a[temp_N].position
	P_H4 = a[temp_H4].position
	P_H5 = a[temp_H5].position
	P_H6 = a[temp_H6].position

	bond_CH1 = P_H1 -P_C
	bond_CH2 = P_H2 -P_C
	bond_CH3 = P_H3 -P_C
	bond_CN = P_N -P_C
	bond_CH4 = P_H4 -P_C
	bond_CH5 = P_H5 -P_C
	bond_CH6 = P_H6 -P_C

	bond_CH1_a = np.dot(bond_CH1,La) / np.sqrt(sum(La**2))
	bond_CH1_b = np.dot(bond_CH1,Lb) / np.sqrt(sum(Lb**2))
	bond_CH1_c = np.dot(bond_CH1,Lc) / np.sqrt(sum(Lc**2))
	bond_CH2_a = np.dot(bond_CH2,La) / np.sqrt(sum(La**2))
	bond_CH2_b = np.dot(bond_CH2,Lb) / np.sqrt(sum(Lb**2))
	bond_CH2_c = np.dot(bond_CH2,Lc) / np.sqrt(sum(Lc**2))
	bond_CH3_a = np.dot(bond_CH3,La) / np.sqrt(sum(La**2))
	bond_CH3_b = np.dot(bond_CH3,Lb) / np.sqrt(sum(Lb**2))
	bond_CH3_c = np.dot(bond_CH3,Lc) / np.sqrt(sum(Lc**2))
	bond_CN_a = np.dot(bond_CN,La) / np.sqrt(sum(La**2))
	bond_CN_b = np.dot(bond_CN,Lb) / np.sqrt(sum(Lb**2))
	bond_CN_c = np.dot(bond_CN,Lc) / np.sqrt(sum(Lc**2))
	bond_CH4_a = np.dot(bond_CH4,La) / np.sqrt(sum(La**2))
	bond_CH4_b = np.dot(bond_CH4,Lb) / np.sqrt(sum(Lb**2))
	bond_CH4_c = np.dot(bond_CH4,Lc) / np.sqrt(sum(Lc**2))
	bond_CH5_a = np.dot(bond_CH5,La) / np.sqrt(sum(La**2))
	bond_CH5_b = np.dot(bond_CH5,Lb) / np.sqrt(sum(Lb**2))
	bond_CH5_c = np.dot(bond_CH5,Lc) / np.sqrt(sum(Lc**2))
	bond_CH6_a = np.dot(bond_CH6,La) / np.sqrt(sum(La**2))
	bond_CH6_b = np.dot(bond_CH6,Lb) / np.sqrt(sum(Lb**2))
	bond_CH6_c = np.dot(bond_CH6,Lc) / np.sqrt(sum(Lc**2))

	if np.abs(bond_CH1_a) > 0.5 * La_norm:
		temp_P_H1 = P_H1 - La
		temp_bond_CH1 = temp_P_H1 -P_C
		temp_bond_CH1_a = np.dot(temp_bond_CH1,La) / np.sqrt(sum(La**2))
		if np.abs(temp_bond_CH1_a) > 0.5 * La_norm:
			P_H1 = P_H1 + La
		else:
			P_H1 = temp_P_H1

	if np.abs(bond_CH1_b) > 0.5 * Lb_norm:
		temp_P_H1 = P_H1 - Lb
		temp_bond_CH1 = temp_P_H1 -P_C
		temp_bond_CH1_b = np.dot(temp_bond_CH1,Lb) / np.sqrt(sum(Lb**2))
		if np.abs(temp_bond_CH1_b) > 0.5 * Lb_norm:
			P_H1 = P_H1 + Lb
		else:
			P_H1 = temp_P_H1

	if np.abs(bond_CH1_c) > 0.5 * Lc_norm:
		temp_P_H1 = P_H1 - Lc
		temp_bond_CH1 = temp_P_H1 -P_C
		temp_bond_CH1_c = np.dot(temp_bond_CH1,Lc) / np.sqrt(sum(Lc**2))
		if np.abs(temp_bond_CH1_c) > 0.5 * Lc_norm:
			P_H1 = P_H1 + Lc
		else:
			P_H1 = temp_P_H1


	if np.abs(bond_CH2_a) > 0.5 * La_norm:
		temp_P_H2 = P_H2 - La
		temp_bond_CH2 = temp_P_H2 -P_C
		temp_bond_CH2_a = np.dot(temp_bond_CH2,La) / np.sqrt(sum(La**2))
		if np.abs(temp_bond_CH2_a) > 0.5 * La_norm:
			P_H2 = P_H2 + La
		else:
			P_H2 = temp_P_H2

	if np.abs(bond_CH2_b) > 0.5 * Lb_norm:
		temp_P_H2 = P_H2 - Lb
		temp_bond_CH2 = temp_P_H2 -P_C
		temp_bond_CH2_b = np.dot(temp_bond_CH2,Lb) / np.sqrt(sum(Lb**2))
		if np.abs(temp_bond_CH2_b) > 0.5 * Lb_norm:
			P_H2 = P_H2 + Lb
		else:
			P_H2 = temp_P_H2

	if np.abs(bond_CH2_c) > 0.5 * Lc_norm:
		temp_P_H2 = P_H2 - Lc
		temp_bond_CH2 = temp_P_H2 -P_C
		temp_bond_CH2_c = np.dot(temp_bond_CH2,Lc) / np.sqrt(sum(Lc**2))
		if np.abs(temp_bond_CH2_c) > 0.5 * Lc_norm:
			P_H2 = P_H2 + Lc
		else:
			P_H2 = temp_P_H2


	if np.abs(bond_CH3_a) > 0.5 * La_norm:
		temp_P_H3 = P_H3 - La
		temp_bond_CH3 = temp_P_H3 -P_C
		temp_bond_CH3_a = np.dot(temp_bond_CH3,La) / np.sqrt(sum(La**2))
		if np.abs(temp_bond_CH3_a) > 0.5 * La_norm:
			P_H3 = P_H3 + La
		else:
			P_H3 = temp_P_H3

	if np.abs(bond_CH3_b) > 0.5 * Lb_norm:
		temp_P_H3 = P_H3 - Lb
		temp_bond_CH3 = temp_P_H3 -P_C
		temp_bond_CH3_b = np.dot(temp_bond_CH3,Lb) / np.sqrt(sum(Lb**2))
		if np.abs(temp_bond_CH3_b) > 0.5 * Lb_norm:
			P_H3 = P_H3 + Lb
		else:
			P_H3 = temp_P_H3

	if np.abs(bond_CH3_c) > 0.5 * Lc_norm:
		temp_P_H3 = P_H3 - Lc
		temp_bond_CH3 = temp_P_H3 -P_C
		temp_bond_CH3_c = np.dot(temp_bond_CH3,Lc) / np.sqrt(sum(Lc**2))
		if np.abs(temp_bond_CH3_c) > 0.5 * Lc_norm:
			P_H3 = P_H3 + Lc
		else:
			P_H3 = temp_P_H3

	#print(bond_CN_a)
	if np.abs(bond_CN_a) > 0.5 * La_norm:
		temp_P_N = P_N - La
		print(temp_P_N)
		temp_bond_CN = temp_P_N -P_C
		temp_bond_CN_a = np.dot(temp_bond_CN,La) / np.sqrt(sum(La**2))
		if np.abs(temp_bond_CN_a) > 0.5 * La_norm:
			P_N = P_N + La
		else:
			P_N = temp_P_N
	#print(P_N)

	if np.abs(bond_CN_b) > 0.5 * Lb_norm:
		temp_P_N = P_N - Lb
		temp_bond_CN = temp_P_N -P_C
		temp_bond_CN_b = np.dot(temp_bond_CN,Lb) / np.sqrt(sum(Lb**2))
		if np.abs(temp_bond_CN_b) > 0.5 * Lb_norm:
			P_N = P_N + Lb
		else:
			P_N = temp_P_N
	#print(P_N)

	if np.abs(bond_CN_c) > 0.5 * Lc_norm:
		temp_P_N = P_N - Lc
		temp_bond_CN = temp_P_N -P_C
		temp_bond_CN_c = np.dot(temp_bond_CN,Lc) / np.sqrt(sum(Lc**2))
		if np.abs(temp_bond_CN_c) > 0.5 * Lc_norm:
			P_N = P_N + Lc
		else:
			P_N = temp_P_N
	#print(P_N)

	if np.abs(bond_CH4_a) > 0.5 * La_norm:
		temp_P_H4 = P_H4 - La
		temp_bond_CH4 = temp_P_H4 -P_C
		temp_bond_CH4_a = np.dot(temp_bond_CH4,La) / np.sqrt(sum(La**2))
		if np.abs(temp_bond_CH4_a) > 0.5 * La_norm:
			P_H4 = P_H4 + La
		else:
			P_H4 = temp_P_H4

	if np.abs(bond_CH4_b) > 0.5 * Lb_norm:
		temp_P_H4 = P_H4 - Lb
		temp_bond_CH4 = temp_P_H4 -P_C
		temp_bond_CH4_b = np.dot(temp_bond_CH4,Lb) / np.sqrt(sum(Lb**2))
		if np.abs(temp_bond_CH4_b) > 0.5 * Lb_norm:
			P_H4 = P_H4 + Lb
		else:
			P_H4 = temp_P_H4

	if np.abs(bond_CH4_c) > 0.5 * Lc_norm:
		temp_P_H4 = P_H4 - Lc
		temp_bond_CH4 = temp_P_H4 -P_C
		temp_bond_CH4_c = np.dot(temp_bond_CH4,Lc) / np.sqrt(sum(Lc**2))
		if np.abs(temp_bond_CH4_c) > 0.5 * Lc_norm:
			P_H4 = P_H4 + Lc
		else:
			P_H4 = temp_P_H4


	if np.abs(bond_CH5_a) > 0.5 * La_norm:
		temp_P_H5 = P_H5 - La
		temp_bond_CH5 = temp_P_H5 -P_C
		temp_bond_CH5_a = np.dot(temp_bond_CH5,La) / np.sqrt(sum(La**2))
		if np.abs(temp_bond_CH5_a) > 0.5 * La_norm:
			P_H5 = P_H5 + La
		else:
			P_H5 = temp_P_H5

	if np.abs(bond_CH5_b) > 0.5 * Lb_norm:
		temp_P_H5 = P_H5 - Lb
		temp_bond_CH5 = temp_P_H5 -P_C
		temp_bond_CH5_b = np.dot(temp_bond_CH5,Lb) / np.sqrt(sum(Lb**2))
		if np.abs(temp_bond_CH5_b) > 0.5 * Lb_norm:
			P_H5 = P_H5 + Lb
		else:
			P_H5 = temp_P_H5

	if np.abs(bond_CH5_c) > 0.5 * Lc_norm:
		temp_P_H5 = P_H5 - Lc
		temp_bond_CH5 = temp_P_H5 -P_C
		temp_bond_CH5_c = np.dot(temp_bond_CH5,Lc) / np.sqrt(sum(Lc**2))
		if np.abs(temp_bond_CH5_c) > 0.5 * Lc_norm:
			P_H5 = P_H5 + Lc
		else:
			P_H5 = temp_P_H5

	if np.abs(bond_CH6_a) > 0.5 * La_norm:
		temp_P_H6 = P_H6 - La
		temp_bond_CH6 = temp_P_H6 -P_C
		temp_bond_CH6_a = np.dot(temp_bond_CH6,La) / np.sqrt(sum(La**2))
		if np.abs(temp_bond_CH6_a) > 0.5 * La_norm:
			P_H6 = P_H6 + La
		else:
			P_H6 = temp_P_H6

	if np.abs(bond_CH6_b) > 0.5 * Lb_norm:
		temp_P_H6 = P_H6 - Lb
		temp_bond_CH6 = temp_P_H6 -P_C
		temp_bond_CH6_b = np.dot(temp_bond_CH6,Lb) / np.sqrt(sum(Lb**2))
		if np.abs(temp_bond_CH6_b) > 0.5 * Lb_norm:
			P_H6 = P_H6 + Lb
		else:
			P_H6 = temp_P_H6

	if np.abs(bond_CH6_c) > 0.5 * Lc_norm:
		temp_P_H6 = P_H6 - Lc
		temp_bond_CH6 = temp_P_H6 -P_C
		temp_bond_CH6_c = np.dot(temp_bond_CH6,Lc) / np.sqrt(sum(Lc**2))
		if np.abs(temp_bond_CH6_c) > 0.5 * Lc_norm:
			P_H6 = P_H6 + Lc
		else:
			P_H6 = temp_P_H6

	a_rotated[temp_H1].position = P_H1
	a_rotated[temp_H2].position = P_H2
	a_rotated[temp_H3].position = P_H3
	a_rotated[temp_N].position = P_N
	a_rotated[temp_H4].position = P_H4
	a_rotated[temp_H5].position = P_H5
	a_rotated[temp_H6].position = P_H6

	move_matrix = (a_rotated[temp_C].position + a_rotated[temp_N].position) * 0.5    # also rotating dot
	#print(a_rotated[temp_C].position)
	a_rotated[temp_C].position = np.dot(R_matrix, (a_rotated[temp_C].position - move_matrix)) + move_matrix
	#print(a_rotated[temp_C].position)
	a_rotated[temp_H1].position = np.dot(R_matrix, (a_rotated[temp_H1].position - move_matrix)) + move_matrix
	a_rotated[temp_H2].position = np.dot(R_matrix, (a_rotated[temp_H2].position - move_matrix)) + move_matrix
	a_rotated[temp_H3].position = np.dot(R_matrix, (a_rotated[temp_H3].position - move_matrix)) + move_matrix
	a_rotated[temp_N].position = np.dot(R_matrix, (a_rotated[temp_N].position - move_matrix)) + move_matrix
	a_rotated[temp_H4].position = np.dot(R_matrix, (a_rotated[temp_H4].position - move_matrix)) + move_matrix
	a_rotated[temp_H5].position = np.dot(R_matrix, (a_rotated[temp_H5].position - move_matrix)) + move_matrix
	a_rotated[temp_H6].position = np.dot(R_matrix, (a_rotated[temp_H6].position - move_matrix)) + move_matrix

write('POSCAR_unwrap', a_rotated)
#a_rotated.get_positions(wrap=True)
a_rotated.wrap()
write('POSCAR_wrap', a_rotated)
