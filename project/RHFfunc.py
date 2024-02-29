from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
import math
def RHF_calculation(atom):
    mol = gto.M(atom = atom, basis = 'ccpvdz')

    MAXITER = 40
    E_conv = 1.0e-6

    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    H_core = T + V
    eri = mol.intor('int2e')

    enuc = mol.get_enuc()
    ndocc = mol.nelec[0]

    # ==> Construct AO orthogonalization matrix A <==

    A = fractional_matrix_power(S, -0.5)
    A = np.asarray(A)
    # Check orthonormality
    ASA = A @ S @ A
    np.allclose(ASA, np.eye(S.shape[0]))

    # ==> Compute C & D matrices with CORE guess <==
    # Transformed Fock matrix

    F_p = A @ H_core @ A

    # Diagonalize F_p for eigenvalues & eigenvectors with NumPy

    C_p = np.linalg.eigh(F_p)[1]

    # Transform C_p back into AO basis

    C = A @ C_p

    # Grab occupied molecular orbitals

    C_occ = C[:, :ndocc]
    #print(C_occ.shape)

    # Build density matrix from occupied orbitals
    D = np.einsum("ik,kj->ij", C_occ, C_occ.T, optimize=True) # TODO wat is verschil met matrixproduct?, waarom 2de transponeren

    # ==> SCF Iterations <==
    # Pre-iteration energy declarations
    SCF_E = 0.0
    E_old = 0.0

    #print('==> Starting SCF Iterations <==\n')

    # Begin Iterations
    for scf_iter in range(1, MAXITER + 1):
        # Build Fock matrix
        J = np.einsum('pqrs,rs->pq', eri, D, optimize=True)
        K = np.einsum('prqs,rs->pq', eri, D, optimize=True)

        F = H_core + (2 * J) - K
        
        # Compute RHF energy

        SCF_E = np.sum(D * (H_core + F)) + enuc
        
        # SCF Converged?
        if (abs(SCF_E - E_old) < E_conv):
            break
        E_old = SCF_E
        
        # Compute new orbital guess

        F_p = A @ F @ A
        C_p = np.linalg.eigh(F_p)[1]

        C = A @ C_p
        C_occ = C[:, :ndocc]
        D = np.einsum("ik,kj->ij", C_occ, C_occ.T, optimize=True)
        
        # MAXITER exceeded?
        if (scf_iter == MAXITER):
            raise Exception("Maximum number of SCF iterations exceeded.")

    # Post iterations
    #print('\nSCF converged.')
    #print('Final RHF Energy: %.8f [Eh]' % (SCF_E))
    return SCF_E


def optimal_angle():
    lijst_E = list()
    opt_E = 0 
    i = 104


    j = 0.94634
    while i <= 105:

        H_2_x = j * math.cos(math.radians(i))
        H_2_y = j * math.sin(math.radians(i))
        SCF_E = RHF_calculation(F'O 0.0 0.0 0.0; H {j} 0.0 0.0; H {H_2_x} {H_2_y} 0.0')
        if SCF_E < opt_E:
            opt_E = SCF_E
            opt_hoek = i
            opt_len = j       
        i += 0.01
            
    
    print(opt_hoek)
    print(opt_len)
    print(opt_E)


def PES_H2O(nsteps):
    hoeken = np.linspace(70, 140, nsteps)
    lengths = np.linspace(0.8, 1, nsteps)

    E = list()
    print("Starting calculations")
    for l in lengths:
        lijst = []
        for i in hoeken:
            print(f"===Starting calculation===\nbondlength: {l}\nangle: {i}")
            H_2_x = l * math.cos(math.radians(i))
            H_2_y = l * math.sin(math.radians(i))
            SCF_E = RHF_calculation(f'O 0.0 0.0 0.0; H {l} 0.0 0.0; H {H_2_x} {H_2_y} 0.0')
            lijst.append(SCF_E)
            print(f"E: {SCF_E}")
            print(f"===Finished calculation===")
        E.append(lijst)

    print(E)

if __name__ == '__main__':
    PES_H2O(40)