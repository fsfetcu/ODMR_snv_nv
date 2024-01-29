#!/usr/bin/python3
# -*- coding: utf-8 -*- 

#TODO 
# Update the docstrings
"""
@author:https://github.com/fsfetcu
"""

import numpy as np
from qutip import Qobj

class vec():
    """
    A module for vector operations.
    """

    @staticmethod
    def cartesianVector(A, theta, phi):
        """ 
        get_vector_cartesian_qutip
        Compute cartesian coordinates of a vector from its spherical coordinates:
        norm A, polar angle theta, azimutal angle phi

        Parameters
        ----------
            A : float
                Norm of the vector.
            theta : float
                Polar angle of the vector.
            phi : float
                Azimutal angle of the vector.
        
        Returns
        -------
            qt.Qobj
                The vector as a QuTiP object.
        """
        vec_array = np.array([A * np.sin(theta) * np.cos(phi), 
                    A * np.sin(theta) * np.sin(phi),
                    A * np.cos(theta)])

        return Qobj(vec_array)

    @staticmethod
    def sphericalCoordinates(vector):
        """
        get_vector_spherical_qutip
        Compute spherical coordinates of a vector from its cartesian coordinates

        Parameters
        ----------
        vector : qt.Qobj
            Vector in cartesian coordinates.
        
        Returns
        -------
        A : float
            Norm of the vector.
        theta : float
            Polar angle of the vector.
        phi : float
            Azimutal angle of the vector.
        """
        Avec = vector.full().flatten()
        A = np.linalg.norm(Avec)
        theta = np.arccos(Avec[2] / A)
        phi = np.arctan2(Avec[1], Avec[0])
        return A, theta, phi
    
    @staticmethod
    def matrixTransformLabNV(idx_nv):
        """ 
        get_rotation_matrix_qutip
        Returns the transformation matrix from lab frame to the desired NV frame, 
        identified by idx_nv (can be 1, 2, 3, or 4).

        The NV center orientations in the diamond lattice are as follows:
        - Index 1: NV axis aligned along the [111] direction (along one of the diagonals in the positive octant).
        - Index 2: NV axis aligned along the [1̅1̅1̅] direction (along one of the diagonals in the negative octant).
        - Index 3: NV axis aligned along the [11̅1̅] direction (in a plane perpendicular to one of the cube faces and along the body diagonal).
        - Index 4: NV axis aligned along the [1̅11] direction (similarly perpendicular to a different cube face and along another body diagonal).

        Parameters
        ----------
        idx_nv : int
            Index of the NV orientation (1, 2, 3, or 4).
            
        Returns
        -------
        qt.Qobj
            The rotation matrix as a QuTiP object.
        """

        if idx_nv == 1:
            return Qobj([[1/np.sqrt(6), -1/np.sqrt(6), -2/np.sqrt(6)],
                        [1/np.sqrt(2),  1/np.sqrt(2),  0],
                        [1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3)]])
        elif idx_nv == 2:
            return Qobj([[-1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)],
                        [-1/np.sqrt(2), -1/np.sqrt(2),  0],
                        [-1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)]])
        elif idx_nv == 3:
            return Qobj([[-1/np.sqrt(6), -1/np.sqrt(6),  2/np.sqrt(6)],
                        [-1/np.sqrt(2),  1/np.sqrt(2),  0],
                        [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)]])
        elif idx_nv == 4:
            return Qobj([[1/np.sqrt(6),  1/np.sqrt(6),  2/np.sqrt(6)],
                        [1/np.sqrt(2), -1/np.sqrt(2),  0],
                        [1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)]])
        else:
            raise ValueError('Invalid index of NV orientation')

    @staticmethod
    def vectorLab_to_NV(vector, idx_nv = 1):
        """
        transform_vector_lab_to_NV_frame_qutip
        Returns the vector in the lab frame, given its value in the NV frame.

        Parameters
        ----------
        vector : qt.Qobj
            Vector in the NV frame.
        idx_nv : int
            Index of the NV orientation (1, 2, 3, or 4).
            
        Returns
        -------
        qt.Qobj
            The vector in the lab frame as a QuTiP object.
        """
        return vec.matrixTransformLabNV(idx_nv) * vector 

    @staticmethod
    def vectorNV_to_Lab(vector, idx_nv = 1):
        """
        transform_vector_NV_to_lab_frame_qutip
        Returns the vector in the NV frame, given its value in the lab frame.

        Parameters
        ----------
        vector : qt.Qobj
            Vector in the lab frame.
        idx_nv : int
            Index of the NV orientation (1, 2, 3, or 4).
            
        Returns
        -------
        qt.Qobj
            The vector in the NV frame as a QuTiP object.
        """
        return vec.matrixTransformLabNV(idx_nv).dag() * vector 
    
    @staticmethod
    def getAllframesCartesian(A, theta, phi):
        """ 
        transform_all_frames_qutip
        Compute cartesian coordinates of a vector in all 4 NV frames, 
        based on its spherical coordinates in lab frame.

        Parameters
        ----------
        A : float
            Norm of the vector.
        theta : float
            Polar angle of the vector.
        phi : float
            Azimutal angle of the vector.
        
        Returns
        -------
        list
            List of the vectors in all 4 NV frames.
        """
        vector = vec.cartesianVector(A, theta, phi)
            
        vector_list = [vec.vectorNV_to_Lab(vector, idx).full().flatten() 
                    for idx in range(1, 5)]

        # return as a list of lists instead of list of arrays
        return [arr.tolist() for arr in vector_list]

    @staticmethod
    def sphericalNV_to_LabFrame(theta_nv, phi_nv, idx_nv=1):
        """ 
        transform_spherical_nv_to_lab_frame_qutip
        Spherical coordinates transformation from given NV frame to lab frame.
        nv_idx can be 1, 2, 3 or 4.

        Parameters
        ----------
        theta_nv : float
            Polar angle of the vector in NV frame.
        phi_nv : float
            Azimutal angle of the vector in NV frame.
        idx_nv : int
            Index of the NV orientation (1, 2, 3, or 4).
        
        Returns
        -------
        theta_lab : float
            Polar angle of the vector in lab frame.
        phi_lab : float
            Azimutal angle of the vector in lab frame.
        """
        vec_in_nv = vec.cartesianVector(1, theta_nv, phi_nv)
        vec_in_lab = vec.vectorNV_to_Lab(vec_in_nv, idx_nv)
        _, theta_lab, phi_lab = vec.sphericalCoordinates(vec_in_lab)
        return theta_lab, phi_lab

    @staticmethod
    def sphericalLab_to_NVFrame(theta_lab, phi_lab, idx_nv=1):
        """ 
        transform_spherical_lab_to_nv_frame_qutip
        Spherical coordinates transformation from lab frame to given NV frame.
        nv_idx can be 1, 2, 3 or 4.

        Parameters
        ----------
        theta_lab : float
            Polar angle of the vector in lab frame.
        phi_lab : float
            Azimutal angle of the vector in lab frame.
        idx_nv : int
            Index of the NV orientation (1, 2, 3, or 4).

        Returns
        -------
        theta_nv : float
            Polar angle of the vector in NV frame.
        phi_nv : float
            Azimutal angle of the vector in NV frame.
        """
        vec_in_lab = vec.cartesianVector(1, theta_lab, phi_lab)
        vec_in_nv = vec.vectorLab_to_NV(vec_in_lab, idx_nv)
        _, theta_nv, phi_nv = vec.sphericalCoordinates(vec_in_nv)
        return theta_nv, phi_nv