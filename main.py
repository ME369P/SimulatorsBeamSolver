from fenics import *
from ufl import nabla_div
import numpy as np
from mshr import *

class Rectangle:
    def __init__(self,b,h):
        self.b = b
        self.h = h
    def BeamMesh(self,L,n):
        domain = Box(Point(0, 0, 0), Point(L, self.b, self.h))
        return generate_mesh(domain,n)
class Boxed:
    def __init__(self,b,h,t):
        self.b = b
        self.h = h
        self.t = t
    def BeamMesh(self,L,n):
        outer = Box(Point(0, 0, 0), Point(L, self.b, self.h))
        inner = Box(Point(0, self.t, self.t), Point(L, self.b-self.t, self.h-self.t))
        domain = outer - inner
        return generate_mesh(domain,n)
class Circle:
    def __init__(self,R):
        self.R = R
    def BeamMesh(self,L,n):
        domain = Cylinder(Point(0,self.R,self.R),Point(L,self.R,self.R),self.R,self.R)
        return generate_mesh(domain,n)
class Tube:
    def __init__(self,R,r):
        self.R = R
        self.r = r
    def BeamMesh(self,L,n):
        outer = Cylinder(Point(0,self.R,self.R),Point(L,self.R,self.R),self.R,self.R)
        inner = Cylinder(Point(0,self.R,self.R),Point(L,self.R,self.R),self.r,self.r)
        domain = outer - inner
        return generate_mesh(domain,n)
class Trapezoid:
    def __init__(self,B,b,h):
        self.B = B
        self.b = b
        self.h = h
    def BeamMesh(self,L,n):
        CS = Polygon([Point(0,0), Point(self.B,0), Point(0.5*(self.B+self.b),self.h), Point(0.5*(self.B-self.b),self.h)])
        domain = Extrude2D(CS,L)
        mesh_trap = generate_mesh(domain,n)
        MeshTransformation.rotate(mesh_trap, 90.0, 0, Point(0,0,0))
        MeshTransformation.rotate(mesh_trap, 90.0, 2, Point(0,0,0))
        return mesh_trap
class Ishape:
    def __init__(self,b,h,tf,tw):
        self.b = b
        self.h = h
        self.tf = tf
        self.tw = tw
    def BeamMesh(self,L,n):
        outer = Box(Point(0,0,0), Point(L,self.b,self.h))
        inner_1 = Box(Point(0,0.5*(self.b-self.tw),self.tf), Point(L,0,self.h-self.tf))
        inner_2 = Box(Point(0,0.5*(self.b+self.tw),self.tf), Point(L,self.b,self.h-self.tf))
        domain = outer - inner_1 - inner_2
        return generate_mesh(domain,n)
class Lshape:
    def __init__(self,b,h,t):
        self.b = b
        self.h = h
        self.t = t
    def BeamMesh(self,L,n):
        outer = Box(Point(0,0,0), Point(L,self.b,self.h))
        inner = Box(Point(0,self.t,self.t), Point(L,self.b,self.h))
        domain = outer - inner
        return generate_mesh(domain,n)

class load:
    def __init__(self,type,value):
        self.type = type
        self.value = value

class BeamProblem:
    
    def __init__(self,material,cross_section,length,num_elements,bc_input,load_input):
        self.material = material
        self.length = length
        self.num_elements = num_elements
        self.mainMesh = cross_section.BeamMesh(length,num_elements)
        self.load_input=load_input
        
        # material properties
        if self.material=="steel":
            E=200e9
            nu=0.285
            self.rho=7700
        elif self.material=="aluminum":
            E=69e9
            #insert more properties for aluminum
        self.mu=E/(2*(1+nu))
        self.lambda_=E*nu/((1+nu)*(1-2*nu))
    
    def Solution(self):
        
        # input variables
        rho = self.rho
        
        # Other variables/constants
        mu = self.mu
        lambda_ = self.lambda_
        g = 9.81
        
        # Create mesh and define function space
        mesh = self.mainMesh
        V = VectorFunctionSpace(mesh, 'P', 1)
        
        # Define boundary condition
        tol = 1E-14
        def clamped_boundary(x, on_boundary):
            return on_boundary and x[0] < tol
        bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)
        
        # Define strain and stress
        def epsilon(u):
            return 0.5*(nabla_grad(u) + nabla_grad(u).T)
            #return sym(nabla_grad(u))
            
        def sigma(u):
            return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

        # Define variational problem
        u = TrialFunction(V)
        d = u.geometric_dimension() # space dimension
        v = TestFunction(V)
        if bool(self.load_input) and self.load_input.type == 'uniform':
            u_l = self.load_input.value
            f = Constant((0+u_l[0], 0+u_l[1], -rho*g+u_l[2]))
        else:
            f = Constant((0,0,-rho*g))
        T = Constant((0,0,0))
        a = inner(sigma(u), epsilon(v))*dx
        L = dot(f, v)*dx + dot(T, v)*ds
        
        # Compute solution
        u = Function(V)
        ps = PointSource(V.sub(1),Point(self.length,0.1,0.1),-1e25)
        ps.apply(u.vector())
        solve(a == L, u, bc)
        
        s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d) # deviatoric stress
        von_Mises = sqrt(3./2*inner(s, s))
        V = FunctionSpace(mesh, 'P', 1)
        von_Mises = project(von_Mises, V)
        
        # Compute magnitude of displacement
        u_magnitude = sqrt(dot(u, u))
        u_magnitude = project(u_magnitude, V)
        
        # Important data outputs
        coordinates = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
        u_mag_array = u_magnitude.vector()[:]
        u_vec_array = np.reshape(u.vector()[:],(np.shape(coordinates)))
        von_Mises_array = von_Mises.vector()[:]
        return {'Coordinates':coordinates,
                'Displacement Magnitudes':u_mag_array, 
                'Displacement Vectors':u_vec_array, 
                'Stress Magnitudes': von_Mises_array}
        
    def Results(self, solution):
        import matplotlib.pyplot as plt
        #from matplotlib.colors import BoundaryNorm #For contour
        #from matplotlib.ticker import MaxNLocator #For contour
        from mpl_toolkits.mplot3d import Axes3D
        
        Len = self.length
        #Create arrays for each reference axis
        coordinates = solution['Coordinates']
        um_plot = solution['Displacement Magnitudes']
        uv_plot = solution['Displacement Vectors']
        vm_plot = solution['Stress Magnitudes']
        
        #xyz coordinates
        x_plot3d = coordinates[:,0]
        y_plot3d = coordinates[:,1]
        z_plot3d = coordinates[:,2]
        #for (x_coord,y_coord,z_coord) in coordinates:
        u_vec_i = uv_plot[:,0]
        u_vec_j = uv_plot[:,1]
        u_vec_k = uv_plot[:,2]
        
        scaling = (z_plot3d.max() - z_plot3d.min()) / abs(u_vec_k).max()
        
        x_plot_def = x_plot3d + scaling * u_vec_i
        y_plot_def = y_plot3d + scaling * u_vec_j
        z_plot_def = z_plot3d + scaling * u_vec_k
        '''
        #For 2d color plots: (Since there's the code for it, function to view x-section contour?)
        #Needs to have one more in each dimension
        #Quantity being plotted should lie in middle of boxes delineated by these
        x_plot = np.zeros((x_div+1,z_div+1))
        #y_plot = np.zeros((x_div+1,y_div+1))
        y_mid = round((Wid / y_div) * (y_div // 2),8) #Central y-value
        z_plot = np.zeros((x_div+1,z_div+1))
        '''
        '''
        #For 3d color and quiver plots
        x_plot3d = np.zeros((x_div,y_div,z_div))
        y_plot3d = np.zeros((x_div,y_div,z_div))
        z_plot3d = np.zeros((x_div,y_div,z_div))
        #Outputs to plot
        um_plot = np.empty((x_div,y_div,z_div))
        um_plot[:,:,:] = np.nan
        uv_plot = np.empty((x_div,y_div,z_div))
        uv_plot[:,:,:] = np.nan
        vm_plot = np.empty((x_div,y_div,z_div))
        vm_plot[:,:,:] = np.nan
        for i in range(x_div):
            for j in range(y_div):
                for k in range(z_div):
                    x_coord = round((Len / x_div) * i,8)
                    y_coord = round((Wid / y_div) * j,8)
                    z_coord = round((Wid / z_div) * k,8)
                    
                    #x_plot[j,k] = x_coord 
                    #y_plot[i,k] = y_coord
                    #z_plot[i,j] = z_coord
                    
                    x_plot3d[i,j,k] = x_coord
                    y_plot3d[i,j,k] = y_coord
                    z_plot3d[i,j,k] = z_coord
                    if (x_coord,y_coord,z_coord) in u_mag_dict.keys():
                        #Change y_mid to y_coord for 3D when implemented
                        um_plot[i,j,k] = u_mag_dict[(x_coord,y_coord,z_coord)]
                        (u_vec_i,u_vec_j,u_vec_k) = np.array(u_vec_dict[(x_coord,y_coord,z_coord)])
                        vm_plot[i,j,k] = von_Mises_dict[(x_coord,y_coord,z_coord)]
        '''
        #Plot Results
        #%matplotlib auto #Uncomment to set backend if ipy file 
        #fig, (ax_u_mag, ax_u_vec, ax_vm) = plt.subplots(nrows=3)
        fig = plt.figure(figsize=plt.figaspect(1.8))
        fig.tight_layout()
        cmap = plt.get_cmap('jet')
        
        '''
        #Function to request contour cross-section along beam?
        ax_um = fig.add_subplot(3,1,1)
        levels_um = MaxNLocator(nbins=15).tick_values(um_plot.min(),um_plot.max())
        norm_um = BoundaryNorm(levels_um, ncolors=cmap.N, clip=True)
        #Contour:
        cf_um = ax_um.contourf(x_plot[:-1, :-1] + (Len / x_div)/2., z_plot[:-1, :-1] + (Wid / z_div)/2., um_plot, levels=levels_um, cmap=cmap)
        fig.colorbar(cf_um,ax=ax_u_mag)
        #Boxes:
        #im_um = ax_um.pcolormesh(x_plot,z_plot,um_plot, cmap=cmap, norm=norm_um)
        #fig.colorbar(im_um, ax=ax_um)
        ax_um.set_title('Displacement magnitudes')
        ax_um.set_xlabel('x')
        ax_um.set_ylabel('z')
        '''
        #Plotting displacement
        ax_uv = fig.add_subplot(2,1,1, projection='3d')
        im_um = ax_uv.scatter(x_plot_def,y_plot_def,z_plot_def, c=um_plot.ravel(), cmap=cmap)
        fig.colorbar(im_um, ax=ax_uv, format = '%.0e')
        ax_uv.auto_scale_xyz([0,Len],[-Len/2,Len/2],[-Len/2,Len/2])
        ax_uv.set_title('Displacement (m)\nVisual Deflection = {:.3e}:1'.format(scaling))
        ax_uv.set_xlabel('x (m)')
        ax_uv.set_ylabel('y (m)')
        ax_uv.set_zlabel('z (m)')
        
        #Plotting von Mises stress
        ax_vm = fig.add_subplot(2,1,2, projection='3d')
        im_vm = ax_vm.scatter(x_plot_def,y_plot_def,z_plot_def, c=vm_plot.ravel(), cmap=cmap)
        ax_vm.auto_scale_xyz([0,Len],[-Len/2,Len/2],[-Len/2,Len/2])
        fig.colorbar(im_vm, ax=ax_vm, format='%.0e')
        ax_vm.set_title('von Mises Stress (Pa)\nVisual Deflection = {:.3e}:1'.format(scaling))
        ax_vm.set_xlabel('x (m)')
        ax_vm.set_ylabel('y (m)')
        ax_vm.set_zlabel('z (m)')
        '''
        #Function to request contour cross-section along beam?
        #cmap = plt.get_cmap('jet')
        levels_vm = MaxNLocator(nbins=15).tick_values(vm_plot.min(),von_Mises_plot.max())
        norm_vm = BoundaryNorm(levels_vm, ncolors=cmap.N, clip=True)
        #Contour:
        cf_vm = ax_vm.contourf(x_plot[:-1, :-1] + (Len / x_div)/2., z_plot[:-1, :-1] + (Wid / z_div)/2., vonm_plot, levels=levels_vm, cmap=cmap)
        fig.colorbar(cf_vm,ax=ax_vm)
        #Boxes:
        #im_vm = ax_vm.pcolormesh(x_plot,z_plot,vm_plot, cmap=cmap, norm=norm_vm)
        #fig.colorbar(im_vm, ax=ax_vm)
        '''
        
        plt.show()
    
#Make a parent class for cross-sections to force implementation of beam mesh?
beam = BeamProblem('steel', Rectangle(0.2,0.2), 1, 16, 'BC', None)
output = beam.Solution()
beam.Results(output)