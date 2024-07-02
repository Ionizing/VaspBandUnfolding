#!/usr/bin/env python3

import numpy as np
from numpy.typing import NDArray
from vaspwfc import vaspwfc
from spinorb import paw_core_soc_mat, \
                    read_cproj_NormalCar


def calc_Hmm(*, nspin: int=1, nkpoints: int=1, nbands: int) -> NDArray:
    """
    Calculate spin-orbit coupling matrix:
    
    returns: [nkpoints, nbands, nbands], where each [nbands, nbands] be like
                 [[up-up, up-dn]                ]
             H = [[dn-up, dn-dn],     ...       ]
                 [                              ]
                 [    ....            ...       ]
    """
    socmat = paw_core_soc_mat()     # [4, nproj, nproj]
    cprojs = read_cproj_NormalCar() # [nspin*nkpoint*nbands, nproj]

    assert socmat.shape[-1] == cprojs.shape[-1], "No. of projectors from SocCar and NormalCAR not consistent."
    assert cprojs.shape[0] == nspin * nkpoints * nbands, "No. of bands in SocCar and input not consistent."

    nproj = socmat.shape[-1]
    cprojs = cprojs.reshape((nspin, nkpoints, nbands, nproj))

    if 1 == nspin:
        cprojs = np.hstack((cprojs, cprojs))

    Hmm = np.zeros((nkpoints, 2*nbands, 2*nbands), dtype=np.complex128)
    for ikpoint in range(nkpoints):
        Hmm[0::2, 0::2] = cprojs[0, ikpoint, :, :].conj() @ socmat[0, :, :] @ cprojs[0, ikpoint, :, :].T
        Hmm[0::2, 1::2] = cprojs[0, ikpoint, :, :].conj() @ socmat[1, :, :] @ cprojs[1, ikpoint, :, :].T
        Hmm[1::2, 0::2] = cprojs[1, ikpoint, :, :].conj() @ socmat[2, :, :] @ cprojs[0, ikpoint, :, :].T
        Hmm[1::2, 1::2] = cprojs[1, ikpoint, :, :].conj() @ socmat[3, :, :] @ cprojs[1, ikpoint, :, :].T

    return Hmm


def recombine_spinor(A: NDArray, B: NDArray) -> tuple[NDArray, NDArray]:
    """
    Construct the spin-polarized states using SOC matrix using Lagrange multiplier.

    Input: A = [2N] = [2,N]
           B = [2N] = [2,N]

    Solve the matrix:

    [   <a|a>,        0,  Re<a|b>, -Im<a|b>] [x1]   [x1]
    [       0,    <a|a>,  Im<a|b>,  Re<a|b>] [y1] = [y1]
    [ Re<a|b>,  Im<a|b>,    <b|b>,        0] [x2]   [x2]
    [-Im<a|b>,  Re<a|b>,        0,    <b|b>] [y2]   [y2]

    to make   |z1*A[0,:] + z2*B[0,:]| -> maximum
        and   |z3*A[1,:] + z4*B[1,:]| -> maximum

    A and B are column vectors with length of `2N`
    """
    assert len(A.shape) == 1 and A.shape == B.shape
    N = A.shape[0] // 2

    A = A.reshape(2, N)
    B = B.reshape(2, N)

    ret = [np.zeros(1), np.zeros(1)]    # make pyright happy

    for ispin in range(2):
        # Lagrange multiplier, matrix is
        nrmsqr_a = np.linalg.norm(A[ispin,:])**2    # <a|a>
        nrmsqr_b = np.linalg.norm(B[ispin,:])**2
        olap_ab  = A[ispin,:].conj() @ B[ispin,:]   # <a|b>

        lmat = np.zeros((4, 4), dtype=float)
        lmat[:2,:2] = nrmsqr_a  # diagonal part
        lmat[2:,2:] = nrmsqr_b

        lmat[0,2] = olap_ab.real
        lmat[1,3] = olap_ab.real
        lmat[2,0] = olap_ab.real
        lmat[3,1] = olap_ab.real

        lmat[0,3] = olap_ab.conj().imag
        lmat[1,2] = olap_ab.imag
        lmat[2,1] = olap_ab.imag
        lmat[3,0] = olap_ab.conj().imag
        
        _eigvals, eigvecs = np.linalg.eigh(lmat)
        minf = np.inf
        mind = 0
        for ii in range(4):
            z1 = eigvecs[0,ii] + 1j*eigvecs[1,ii]
            z2 = eigvecs[2,ii] + 1j*eigvecs[3,ii]
            imin = np.linalg.norm(z1 * A[ispin,:] + z2 * B[ispin,:])
            if imin < minf:
                minf = imin
                mind = ii
            pass

        z1 = eigvecs[0,mind] + 1j*eigvecs[1,mind]
        z2 = eigvecs[2,mind] + 1j*eigvecs[3,mind]
        
        ret[ispin] = z1 * A.flatten() + z2 * B.flatten()

    rA, rB = ret
    return (rA, rB)


def solve_spinor():
    """
    """
    pass


def make_spinor(wavecar: str="WAVECAR"):
    wfn = vaspwfc(wavecar)
    assert wfn._lsoc is not True, "This WAVECAR contains spinors already."

    nspin     = wfn._nspin
    nkpoints  = wfn._nkpts
    nbands    = wfn._nbands
    band_eigs = wfn._bands

    # spin-orbit coupling matrix, [2*nbands, 2*nbands]
    Hmm = calc_Hmm(nspin=nspin, nkpoints=nkpoints, nbands=nbands)

    ibs = np.mgrid[0:2*nbands]

    # Add the band eigenvalue to the diagonal of Hmm
    # and then diagonalize Hmm to get new band eigenvalues
    # finally correct it
    for ikpoint in range(nkpoints):
        eigs = band_eigs[:, ikpoint, :].flatten()
        if nspin == 1:
            neweigs = np.zeros(eigs.size*2)
            neweigs[0::2] = eigs
            neweigs[1::2] = eigs
            eigs = neweigs

        Hmm[ikpoint, ibs, ibs] += eigs
        assert np.allclose(
                np.abs(Hmm[ikpoint, :, :] - Hmm[ikpoint, :, :].T.conj()),
                np.zeros((2*nbands, 2*nbands)),
               ), "Hmm not Hermitian."
        eigvals, eigvecs = np.linalg.eigh(Hmm)

        # new_eigvec1, new_eigvec2 = eigvecs[]
        eigvals[0::2] = (eigvals[0::2] + eigvals[1::2]) / 2.0
        eigvals[1::2] = eigvals[0::2]


if '__main__' == __name__:
    pass
