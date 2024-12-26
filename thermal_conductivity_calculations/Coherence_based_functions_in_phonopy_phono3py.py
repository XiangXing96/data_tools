'''
This script is written to calculate coherence effect
based on functions in phonopy and phono3py. Note that 
the scattering rate is introduced from the calculation 
in ShengBTE. THe mesh (Nx, Ny, Nz) should all be odd to 
keep agreement with the result from phonopy and phono3py.
The author is XIANG Xing (xxiangad@connect.ust.hk).
'''

from functools import reduce
import numpy as np
from phonopy.interface.vasp import read_vasp
import phonopy
from phonopy import Phonopy
from phono3py.file_IO import read_fc2_from_hdf5

## Pathes and parameters need to be modified according to different cases

# Set some parameters
mesh = [19,19,19]
superce_dia = [3, 3, 3]
temperature = 1500			# unit is K
cutoff = 0.001				# unit is THz
contain_4ph = True			# True or False
distinguish_acco_optical = True		# calculate the contribution of accoustic and optical modes 
main_folder = ""

# Give pathes for all files
file_POSCAR = main_folder + "POSCAR"  # primitive cell is required to interface the alamode
file_fc_matrix =  main_folder + "FORCE_CONSTANTS_2ND" # 2nd-order force constant matrix
file_ir_points =  main_folder + "BTE.qpoints" # irreducable points

file_scattering_iso =  main_folder + "BTE.w_isotopic" # calculated by ShengBTE
file_scattering_3ph =  main_folder + "BTE.w_3ph" # 3-phonon scattering process calculated by ShengBTE
if contain_4ph:
    file_scattering_4ph =  main_folder + "BTE.w_4ph" # 4-phonon scattering processcalculated by ShengBTE
#file_full_points = "BTE.qpoints_full"  # all points

file_scattering_test = "scattering_test.txt"

## construct a Phonopy class by reading files (POSCAR and FORCE_CONSTANTS)  
cell = read_vasp(file_POSCAR)
sup = Phonopy(cell, np.diag(superce_dia), group_velocity_delta_q = 1e-5)
ph = phonopy.load(supercell=superce_dia, unitcell_filename=file_POSCAR, force_constants_filename=file_fc_matrix, primitive_matrix=np.diag([1,1,1]), is_compact_fc=False)
sup._force_constants_decimals = True
sup.force_constants = ph.force_constants
sup.run_mesh(mesh, is_gamma_center=True, with_group_velocities=True)

## trans qpoints into BZ to judge if mesh grid in ShengBTE is equivalent to that in Phonopy
ir_points_ShengBTE = np.loadtxt(file_ir_points)
irqp_ShengBTE = ir_points_ShengBTE[:,3:]
irqp_ShengBTE_BZ = irqp_ShengBTE - np.rint(irqp_ShengBTE)

ir_points_Phonopy = sup.mesh.qpoints
irqp_Phonopy_BZ = ir_points_Phonopy - np.rint(ir_points_Phonopy)
#print(irqp_Phonopy_BZ)

def matrix_equal(mat1, mat2):
    """
    Judge if they are the same
    :param mat1: the first matrix, a 2-D list
    :param mat2: the second matrix, a 2-D list
    :return: if they are the same, return True, or False
    """
    # If they have different lines and columns
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return False

    mat1 = np.around(mat1, decimals=2)
    mat2 = np.around(mat2, decimals=2)

    # check every element
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            if mat1[i][j] != mat2[i][j]:
                print(mat1[i][j])
                print(mat2[i][j])
                return False

    # if all elements are the same
    return True

compare_result = matrix_equal(irqp_ShengBTE_BZ, irqp_Phonopy_BZ)
if compare_result:
    print("q_points in ShengBTE and Phonopy is the same!"+ "\n" \
          + "We use qpoints in Phonopy.")
else:
    print("Notice!!! q_points in ShengBTE and Phonopy is different." + "\n" \
          + "Please check!!")

irqp = irqp_Phonopy_BZ
#print(irqp)


## Calculate group velocity operator
## 1. construct a velocity operator class
from phono3py.phonon.velocity_operator import VelocityOperator
from phonopy.units import VaspToTHz

gv_operator_class = VelocityOperator(sup.dynamical_matrix, symmetry=sup._primitive_symmetry, \
              q_length=1e-6,  frequency_factor_to_THz=VaspToTHz)
gv_operator_class.run(irqp)
gv_operator_all_irqp = gv_operator_class.velocity_operators
#print(np.shape(gv_operator_class.velocity_operators))

# 2.Symmetrize group velocity operator
# Get rotation map, i.e. find the corresponding grid point 
# for points obtained by rotations
from phonopy.utils import similarity_transformation

# full q-points
q_points = np.array(sup.mesh.grid_address)
mesh_N = np.array(mesh)
q_points = q_points / mesh_N
# trans them into BZ area
q_points = q_points - np.rint(q_points)

size_gv = np.array(np.shape(gv_operator_all_irqp))
size_gv_by_gv_sum2 = np.copy(size_gv)
size_gv_by_gv_sum2[-1] = 6 
size_gv_by_gv = np.append(size_gv,3)
#print(size_gv_by_gv_sum2)
gv_by_gv_operator_all_irqp_sym = np.zeros(size_gv_by_gv, dtype=np.complex64)
#print(np.shape(gv_by_gv_operator_all_irqp_sym))

for i_q, ir_qpoint in enumerate(irqp):
    gv_operator = gv_operator_all_irqp[i_q]
    #print(i)
    #print(i)
    #print(ir_qpoint)

    rotated_qp = []
    for r in sup._primitive_symmetry.reciprocal_operations:
        q_in_BZ = ir_qpoint - np.rint(ir_qpoint)
        #print(q_in_BZ)
        rotated_qp.append(np.dot(r, q_in_BZ))

    # trans them into BZ area
    rotated_qp =  rotated_qp - np.rint(rotated_qp)
    #print(np.shape(rotated_qp))
    rotation_map = np.zeros([len(rotated_qp)]) 
    #print(rotation_qp)
    #print(np.shape(rotation_map))
    #print(q_points)
    for j in range(len(rotated_qp)):
        temp_qp = rotated_qp[j]
        #print(j)
        #print(temp_qp)
    #print(r)
        diff_q = np.array(abs(q_points - temp_qp))
    #print(r)
    #print(diff_q)
        x_temp = np.where(diff_q[:,0] < 1e-5)
        y_temp = np.where(diff_q[:,1] < 1e-5)
        z_temp = np.where(diff_q[:,2] < 1e-5)
        common_line = reduce(np.intersect1d, [x_temp, y_temp, z_temp])   
    #print(i)
    #print(common_line)
        rotation_map[j] = common_line[0] + 1    # Note that the order of qpoints starts from 1 !!!!
    #id_map = 
    #print(common_line)
    #print(rotation_map)
#k_star = len(np.unique(rotation_map))
#print(k_star)

#print(np.shape(sup._primitive_symmetry.reciprocal_operations))
#print("179th line:")
#print(np.shape(rotations))
#print(rotations)

    primitive = sup._dynamical_matrix.primitive
    reciprocal_lattice_inv = primitive.cell
    reciprocal_lattice = np.linalg.inv(reciprocal_lattice_inv)


    nat3 = len(primitive)*3
    nbands = np.shape(gv_operator)[0]

    gv_by_gv_operator = np.zeros((nbands, nat3, 3, 3), dtype=complex)
    gv_operator_sym = np.zeros((nbands, nat3, 3), dtype=complex)

    for r in sup._primitive_symmetry.reciprocal_operations:
        r_cart = similarity_transformation(reciprocal_lattice, r)
        gvs_rot_operator = np.zeros((nbands, nat3, 3), dtype=complex)
        for s in range(0, nbands):
            for s_p in range(0, nat3):
                for di in range(0, 3):
                    for dj in range(0, 3):
                        gvs_rot_operator[s, s_p, di] += (gv_operator[s, s_p, dj] * r_cart.T[dj, di])
                        gv_operator_sym[s, s_p, di] += gvs_rot_operator[s, s_p, di]

        for s in range(0, nbands):
            for s_p in range(0, nat3):
                for di in range(0, 3):
                    for dj in range(0, 3):
                        gv_by_gv_operator[s, s_p, di, dj] += gvs_rot_operator[s,s_p,di] * np.conj(gvs_rot_operator[s,s_p,dj])

    order_kstar = len(np.unique(rotation_map))
    gv_operator_sym /= len(rotation_map) // len(np.unique(rotation_map))
    gv_by_gv_operator /= len(rotation_map) // len(np.unique(rotation_map))

    if order_kstar != sup.mesh.weights[i_q]:
        print(order_kstar)
        print(sup.mesh.weights[i_q])
        print("The symmetry of q_points breaks down."
        + "Please check!!!!")

    #print(gv_operator[0,0,1])
    #print(np.shape(gv_operator))
    #print(np.shape(gv_by_gv_operator))
    #print(gv_operator_sym[0,0,1])
    #print(gv_by_gv_operator[0,0,2,2])

    gv_by_gv_operator_all_irqp_sym[i_q] = gv_by_gv_operator

## reshape the format of gv and fill it into gv_operator_sum2
gv_operator_sum2 = np.zeros(size_gv_by_gv_sum2, dtype=np.complex64)

for i in range(int(len(irqp))):
    for j, vxv in enumerate(([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
        gv_operator_sum2[i,:,:,j] = gv_by_gv_operator_all_irqp_sym[i,:,:,vxv[0],vxv[1]]

#print(gv_operator_sum2[2,0,:,:])


# Calculate modal heat capacity
from phonopy.units import EV, Angstrom, Kb, THz, THzToEv
from phonopy.phonon.thermal_properties import mode_cv

N_irqp = int(len(irqp))
primitive = sup._dynamical_matrix.primitive
nat3 = int(len(primitive)*3)
Cv_all_irqp = np.zeros([N_irqp,nat3]) 

'''
temperature = 1500
dyna_mat = sup._dynamical_matrix
dyna_mat.run(qp)
dm = dyna_mat.dynamical_matrix
#print(sup.force_constants)
eigvals, eigvecs = np.linalg.eigh(dm)
eigvals = eigvals.real
freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * VaspToTHz
freqs_eV = freqs * THzToEv
'''

frequency = sup.mesh.frequencies
#frequency[0,0] = 0
#frequency[0,1] = 0
#frequency[0,2] = 0
frequency_eV = frequency * THzToEv

print(np.shape(frequency))

cutoff_eV = cutoff * THzToEv

for i_q in range(int(len(irqp))):
    freqs = frequency[i_q]
    freqs_eV = frequency_eV[i_q]
    cv = np.zeros(len(freqs), dtype="double")
    for i, f in enumerate(freqs_eV):
        if f > cutoff_eV:
            condition = f < 100 * temperature * Kb
            #print(f/THzToEv*2*np.pi)
            #print(condition)
            cv[i] = np.where(condition,
                mode_cv(np.where(condition,temperature,10000), f),
                0)
    Cv_all_irqp[i_q] = cv
#print(cv)


# Read modal scattering rate
# read scattering rate for each mode,
# which is calculated by ShengBTE


# Read isotope-phonon scattering rate
scattering_rate = np.loadtxt(file_scattering_iso) # unit is ps-1
#print(scattering_rate)
#print(len(sup.mesh.qpoints))

N_qp = int(len(sup.mesh.qpoints))
N_omg = int(len(scattering_rate)/N_qp)
scattering_iso = np.zeros([N_qp, N_omg])
omg_BTE = np.zeros([N_qp, N_omg])
for i in range(N_omg):
    for j in range(N_qp):
        order = i * N_qp + j
        omg_BTE[j,i] = scattering_rate[order, 0]
        scattering_iso[j,i] = scattering_rate[order, 1]

# Read 3-phonon scattering rate
scattering_rate = np.loadtxt(file_scattering_3ph) # unit is ps-1
#print(scattering_rate)
#print(len(sup.mesh.qpoints))

N_qp = int(len(sup.mesh.qpoints))
N_omg = int(len(scattering_rate)/N_qp)
scattering_3ph = np.zeros([N_qp, N_omg])
omg_BTE = np.zeros([N_qp, N_omg])
for i in range(N_omg):
    for j in range(N_qp):
        order = i * N_qp + j
        omg_BTE[j,i] = scattering_rate[order, 0]
        scattering_3ph[j,i] = scattering_rate[order, 1]

# Read 4-phonon scattering rate

if contain_4ph:
    scattering_rate = np.loadtxt(file_scattering_4ph) # unit is ps-1
    #print(scattering_rate)
    # #print(len(sup.mesh.qpoints))
    N_qp = int(len(sup.mesh.qpoints))
    N_omg = int(len(scattering_rate)/N_qp)
    scattering_4ph = np.zeros([N_qp, N_omg])
    omg_BTE = np.zeros([N_qp, N_omg])
    for i in range(N_omg):
        for j in range(N_qp):
            order = i * N_qp + j
            omg_BTE[j,i] = scattering_rate[order, 0]
            scattering_4ph[j,i] = scattering_rate[order, 1]
else:
    scattering_4ph = np.zeros([N_qp, N_omg])

# sum all scattering rate
scattering_qp = (scattering_iso + scattering_3ph + scattering_4ph) / 4 / np.pi

#print(scattering_qp)

"""
# Read scattering rate txt
scattering_rate = np.loadtxt(file_scattering_test) # unit is ps-1
#print(scattering_rate)
#print(len(sup.mesh.qpoints))

N_qp = int(len(sup.mesh.qpoints))
N_omg = int(len(scattering_rate)/N_qp)
scattering_qp = np.zeros([N_qp, N_omg])
omg_BTE = np.zeros([N_qp, N_omg])
for i in range(N_omg):
    for j in range(N_qp):
        order = i * N_qp + j
        omg_BTE[j,i] = scattering_rate[order, 0]
        scattering_qp[j,i] = scattering_rate[order, 1]
"""

# Calculate modal thermal conductivity matrix
from phono3py.conductivity.wigner import get_conversion_factor_WTE

primitive = sup._dynamical_matrix.primitive
N_band = len(primitive)*3

vol = primitive.volume
#print(vol)
conversion = get_conversion_factor_WTE(vol)

k_P = np.zeros([len(irqp), nat3, 6])
k_C = np.zeros([len(irqp), nat3, nat3, 6])

N_ignore_modes = np.zeros(len(irqp))

for i, qpoint in enumerate(irqp):
    #print("q_point")
    #print(qpoint)
    cv = Cv_all_irqp[i]
    #print("Heat capacity:")
    #print(cv)
    frequencies = frequency[i]
    #print(frequencies)
    #print(str(frequencies[0])+ " " +str(frequencies[1])+" "+str(frequencies[2])+" "+str(frequencies[3])+" "+str(frequencies[4])+" "+str(frequencies[5]))
    g_sum = scattering_qp[i]  # scattering rate of a mode
    for s1 in range(N_band):
        for s2 in range(N_band):
            hbar_omega_eV_s1 = (frequencies[s1] * THzToEv)
            hbar_omega_eV_s2 = (frequencies[s2] * THzToEv)
            if (frequencies[s1] > cutoff) and (
                frequencies[s2] > cutoff):

                hbar_gamma_eV_s1 = 2.0 * g_sum[s1] * THzToEv
                hbar_gamma_eV_s2 = 2.0 * g_sum[s2] * THzToEv

                #hbar_gamma_eV_s1 = g_sum[s1] * THzToEv
                #hbar_gamma_eV_s2 = g_sum[s2] * THzToEv

                lorentzian_divided_by_hbar = (
                    0.5 *(hbar_gamma_eV_s1 + hbar_gamma_eV_s2)
                    ) / (
                        (hbar_omega_eV_s1 - hbar_omega_eV_s2)**2
                        + 0.25 *((hbar_gamma_eV_s1 + hbar_gamma_eV_s2)**2)
                    )
                prefactor = (
                    0.25 * (hbar_omega_eV_s1 + hbar_omega_eV_s2)
                    * (
                        cv[s1] / hbar_omega_eV_s1
                        + cv[s2] /hbar_omega_eV_s2
                    ))
                if np.abs(frequencies[s1] - frequencies[s2]) < 1e-4:
                    # degenerate or diagonal s1=s2 modes, determine k_P
                    contribution = (
                        gv_operator_sum2[i,s1,s2]
                        * prefactor
                        * lorentzian_divided_by_hbar
                        * conversion
                    ).real
                    
                    #print(np.shape(contribution))
                    
                    k_P[i, s1] += (0.5 * contribution)
                    k_P[i, s2] += (0.5 * contribution)
                    
                    #print(k_P[i, s2])
                    # prefactor 0.5 arises from the fact that degenerate
                    # modes have the same specific heat, hence they give 
                    # conductivity

                else:
                    k_C[i,s1,s2] += (
                        (gv_operator_sum2[i,s1,s2])
                        * prefactor
                        * lorentzian_divided_by_hbar
                        * conversion
                    ).real

            elif s1 == s2:
                N_ignore_modes[i] += 1
    #print(k_C[i,:,:, 0])


#print(np.shape(k_P))
#print(conversion)
#Thermal_conductivity = k_P.sum(axis=0).sum(axis=1)
#print(Thermal_conductivity)
size_k_P = np.shape(k_P)
#TC = np.zeros_like(k_P)
#for i in range(int(size_k_P[0])):
#    TC[i] = k_P[i] * sup.mesh.weights[i]

N_total_modes = np.sum(sup.mesh.weights)
#print(N_total_modes)
Thermal_conductivity_Propagon = k_P.sum(axis=0).sum(axis=0) / N_total_modes
Thermal_conductivity_Coherence = k_C.sum(axis=0).sum(axis=0).sum(axis=0) / N_total_modes

print(Thermal_conductivity_Propagon)
print(Thermal_conductivity_Coherence)

'''
# Distinguish accoustic and optical
for i, qpoint in enumerate(irqp):
	dyna_mat = sup._dynamical_matrix
	dyna_mat.run(qpoint)
	dm = dyna_mat.dynamical_matrix
	#print(sup.force_constants)
	eigvals, eigvecs = np.linalg.eigh(dm)
	eigvals = eigvals.real
	freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * VaspToTHz
	print(qpoint)
	print("the shape of freqs:")
	print(np.shape(freqs))
	print("the shape of eigvecs:")
	print(np.shape(eigvecs))
'''
'''
# Calculate the contribution of accoustic and optical modes
def same_sign(arr):
    """
    judge whether all elements have the same sign
    """
    if len(arr) < 2:
        return True  # if length is smaller than 2, sign are the same
    sign = 1 if arr[0] >= 0 else -1  # record the sign of first element 
    for i in range(1, len(arr)):
        if arr[i] >= 0 and sign < 0 or arr[i] < 0 and sign >= 0:
            return False  #If signs are different, return False
    return True  # If they are the same, return True

accoustic = []
for i, q_point in enumerate(irqp):
	dyna_mat = sup._dynamical_matrix
	#qpoint = [0.0, 0.0, 0.0]
	dyna_mat.run(q_point)
	dm = dyna_mat.dynamical_matrix
	print(q_point)
	eigvals, eigvecs = np.linalg.eigh(dm)
	eigvals = eigvals.real
	eigvecs = eigvecs.real
	freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * VaspToTHz
	size = np.shape(eigvecs)
	number = size[1]
	for j in range(number):
		temp_eigen = eigvecs[:, j]
		print(temp_eigen)
		temp = temp_eigen.reshape((-1,3))
		print(temp)
		if (same_sign(temp[:,0]) and same_sign(temp[:,1])) and same_sign(temp[:,2]):
			accoustic.append(freqs[j])
np.savetxt('accoustic.txt', accoustic)
'''
'''
print(qpoint)
print("freqs:")
print(freqs_eV)
print("the shape of eigvecs:")
print(eigvecs)
np.savetxt('Fre_sq.txt', freqs)
np.savetxt('Eigvec_sq.txt', eigvecs[:,0])
'''

# Output information
np.savetxt('K_P.txt', Thermal_conductivity_Propagon)
np.savetxt('K_C.txt', Thermal_conductivity_Coherence)

Kcx = k_C[:,:,:,0]
Kcy = k_C[:,:,:,1]
Kcz = k_C[:,:,:,2]
Kpx = k_P[:,:,0]
Kpy = k_P[:,:,1]
Kpz = k_P[:,:,2]

#print(k_C[1,12,12,:])
#print(k_P[1,12,:])
#print(np.shape(k_C))
#print(np.shape(k_P))
#print(np.shape(frequency))

Fre = np.copy(frequency)
Fre = Fre.reshape([1,-1])
Fre.sort()
#print(Fre)
matrix_x = np.zeros([len(irqp) * nat3,len(irqp) * nat3])
matrix_y = np.zeros([len(irqp) * nat3,len(irqp) * nat3])
matrix_z = np.zeros([len(irqp) * nat3,len(irqp) * nat3])
for i in range(len(irqp)):
    for s1 in range(nat3):
        for s2 in range(nat3):
            temp_s1 = frequency[i,s1]
            id_s1 = np.argwhere(Fre == temp_s1)
            #print(" id_s1 ")
            #print(id_s1)
            id_s1 = id_s1[0,1]
            #print(" shape of id_s1[0] ")
            #print(id_s1[0])
            temp_s2 = frequency[i,s2]
            id_s2 = np.argwhere(Fre == temp_s2)
            #print(str(temp_s1)+" and "+str(temp_s2))
            #print(" id_s2 "+str(id_s2))
            id_s2 = id_s2[0,1]
            #print(" id_s2[0] "+str(id_s2))
            if s1 == s2:
                matrix_x[id_s1,id_s2] += Kpx[i,s1] / N_total_modes
                matrix_y[id_s1,id_s2] += Kpy[i,s1] / N_total_modes
                matrix_z[id_s1,id_s2] += Kpz[i,s1] / N_total_modes
                #print(matrix_x[id_s1,id_s2] * N_total_modes)
                #print(matrix_x[id_s1,id_s2]* N_total_modes)
                #print(Kpy[i,s1])

            else:
                matrix_x[id_s1,id_s2] += Kcx[i,s1,s2] / N_total_modes
                matrix_y[id_s1,id_s2] += Kcy[i,s1,s2] / N_total_modes
                matrix_z[id_s1,id_s2] += Kcz[i,s1,s2] / N_total_modes
                #print(matrix_x[id_s1,id_s2] * N_total_modes)
                #print(matrix_x[id_s1,id_s2])
                #print(Kcx[i,s1,s2])


np.savetxt('Fre.txt', Fre)
#np.savetxt('Fre.txt', Fre, fmt="%.05")

K_propagon = np.trace(matrix_y)
K_coherence = np.sum(np.sum(matrix_y)) - K_propagon

print(K_propagon)
print(K_coherence)

matrix_x_none_zero = matrix_x[3:,3:]

#np.savetxt('matrix_x.txt', matrix_x_none_zero, delimiter='\t', fmt='%.5f')

with open("matrix_x.txt", "w") as file:
    for row in matrix_x:
        for element in row:
            #file.write(str(round(element, 15)) + " ")
            file.write(str(element) + " ")
        file.write("\n")
with open("matrix_y.txt", "w") as file:
    for row in matrix_y:
        for element in row:
            #file.write(str(round(element, 15)) + " ")
            file.write(str(element) + " ")
        file.write("\n")
with open("matrix_z.txt", "w") as file:
    for row in matrix_z:
        for element in row:
            #file.write(str(round(element, 15)) + " ")
            file.write(str(element) + " ")
        file.write("\n")
