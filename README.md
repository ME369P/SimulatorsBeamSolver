# SimulatorsBeamSolver
Object-oriented approach to solving beam deflection problems using FEniCS

Bonnie Chan (bc32569)
Derek Hornung (deh2434)
Brian Tran (bat2285)

Overall Objective:
	Create a user-friendly program that can solve beam deflection problems based on user inputs of loading, cross-section geometries, and boundary conditions.

Packages to be used:
	FEniCS
	numpy
	matplotlib

Inputs:
	Loading
		Point (in the works)
		Uniform
		Variable distributed (in the works)
	Beam or ball valve handle type (standard)
	Boundary conditions
		Clamped-Free
		Clamped-Clamped (in the works)
		Clamped-Rotating (in the works)
		Rotating-Rotating (in the works)

Outputs:
	Location and value of minimum safety factor
	Location and value of maximum stress
	Location and value of maximum deflection
	Heat map of stress
	Heat map of of deformation
	Deformed 3D scatter plot

Useful links:
	https://www.encyclopediaofmath.org/index.php/Lam%C3%A9_constants
	https://www.efunda.com/math/areas/IbeamIndex.cfm
	https://fenicsproject.org/pub/tutorial/pdf/fenics-tutorial-vol1.pdf
  https://fenics-shells.readthedocs.io/en/latest/demo/reissner-mindlin-simply-supported/demo_reissner-mindlin-simply-supported.py.html

Misc:
    x is the axial direction, y is the horizontal transverse direction, and z is the vertical (gravity-affected) transverse direction
    
    Input format for BeamProblem:
        BeamProblem('Material', Cross Section Object(Cross Section dimensions), Length, Number of Elements, Boundary conditions, Load)
        
    Outputs from BeamProblem(....).Solution():
        {'Coordinates': numpy array of coordinates,
        'Displacement Magnitudes': numpy array of displacement magnitudes,
        'Displacement Vectors': numpy array of displacement vectors,
        'Stress Magnitudes': numpy array of stress magnitudes}
        
    Loading input to BeamProblem must be specified as:
        For uniform loads:
            load('uniform',(x value, y value, z value)) where the units for the x, y, or z values are in N/m^3
        For no loading (except gravity)
            None
        For point loads:
            TBD
