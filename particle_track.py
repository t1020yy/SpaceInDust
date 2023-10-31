import io
import math 

import cv2
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
import matplotlib.pyplot as plt

from connected_component import Connected_component

class Particle_track:
    """Class to strore particle track"""
    
    def __init__(self, component1: Connected_component, component2: Connected_component, P1, P2):
        self.component1 = component1
        self.component2 = component2

        self.P1, self.P2 = P1, P2
        #self.R, self.T = R, T
        self.grid_height = 0
        self.particle_radius = 0
        self.particle_density = 0

        self._surface = None
        self._triangulated_3d_points = None
        self._points3d = None
        self._flatten_3d_points = None
        self._length = None
        self._parabola = None
        self._parameters = None


    @property
    def surface(self):
        if self._surface is None:
            self._fit_plane_to_points()
        return self._surface

    @property
    def triangulated_3d_points(self):
        if self._triangulated_3d_points is None:
            self._triangulate_3d_points()
        return self._triangulated_3d_points

    @property
    def flatten_3d_points(self):
        if self._flatten_3d_points is None:
            self._get_flatten_3d_points()
        return self._flatten_3d_points

    @property
    def length(self):
        if self._length is None:
            self._get_track_length()
        return self._length

    @property
    def parabola(self):
        if self._parabola is None:
            self._get_parabola()
        return self._parabola

    @property
    def parameters(self):
        if self._parameters is None:
            self._get_parameters()
        return self._parameters
    
    @property
    def V0(self):
        if self._parameters is None:
            self._get_parameters()
        return self._parameters[3]

    @property
    def Alpha(self):
        if self._parameters is None:
            self._get_parameters()
        return self._parameters[4]


    def _fit_plane_to_points(self):
        points = self._triangulated_3d_points[:,0,:]
        [rows, cols] = points.shape
        G = np.ones((rows, 3))
        G[:,0] = points[:,0]  #X
        G[:,1] = points[:,1]  #Y
        Z = points[:,2]
        (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None) 
        normal = (a, b, -1)
        nn = np.linalg.norm(normal)
        normal = normal / nn
        self._surface = (a, b, c)
    

    # def _triangulate_3d_points(self):
    #     corresponding_points = [[], []]
    #     component1, component2 = self.component1, self.component2
    #     for i in range(len(component1.center_line)):
    #         p1_left = component1.center_line_left[i]
    #         p1_right = component1.center_line_right[i]
    #         p2_left = component2.center_line_left[i]
    #         p2_right = component2.center_line_right[i]
    #         corresponding_points[0].append([p1_left[0] + component1.left, p1_left[1] + component1.top])
    #         corresponding_points[0].append([p1_right[0] + component1.left, p1_right[1] + component1.top])
    #         corresponding_points[1].append([p2_left[0] + component2.left, p2_left[1] + component2.top])
    #         corresponding_points[1].append([p2_right[0] + component2.left, p2_right[1] + component2.top])
    #     points1 = np.array(corresponding_points[0], dtype=float)
    #     points2 = np.array(corresponding_points[1], dtype=float)
    #     points_4d = cv2.triangulatePoints(self.P1, self.P2, points1.T, points2.T)
    #     points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
    #     self._triangulated_3d_points = points_3d



    def _triangulate_3d_points(self):
        corresponding_points = [[], []]

        #roi_height = min(component1.height, component2.height)
        component1, component2 = self.component1, self.component2
        
        # for p1, p2 in zip(component1.center_line, component2.center_line):
        #     corresponding_points[0].append([p1[0] + component1.left, p1[1] + component1.top])
        #     corresponding_points[1].append([p2[0] + component2.left, p2[1] + component2.top])

        for p1, p2 in zip(component1.center_line_left, component2.center_line_left):
            corresponding_points[0].append([p1[0] + component1.left, p1[1] + component1.top])
            corresponding_points[1].append([p2[0] + component2.left, p2[1] + component2.top])

        corresponding_points[0].append([component1.top_x + component1.left, component1.top_y + component1.top])
        corresponding_points[1].append([component2.top_x + component2.left, component2.top_y + component2.top])

        for p1, p2 in zip(component1.center_line_right, component2.center_line_right):
            corresponding_points[0].append([p1[0] + component1.left, p1[1] + component1.top])
            corresponding_points[1].append([p2[0] + component2.left, p2[1] + component2.top])

        #projmtx1 = np.dot(self.P1, np.hstack((np.identity(3), np.zeros((3,1)))))
        #projmtx2 = np.dot(self.P2, np.hstack((self.R, self.T)))
        points1 = np.array(corresponding_points[0], dtype=float)
        points2 = np.array(corresponding_points[1], dtype=float)
        points_4d = cv2.triangulatePoints(self.P1, self.P2, points1.T, points2.T)
        points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
        
        self._triangulated_3d_points = points_3d

    def _project_point_to_plane(self, point):
        """
        Projects the points with coordinates x, y, z onto the plane
        defined by a*x + b*y + c*z = 1
        """
        a, b, c = self.surface

        vector_norm = a*a + b*b + c*c
        normal_vector = np.array([a, b, c]) / np.sqrt(vector_norm)
        point_in_plane = np.array([a, b, c]) / vector_norm

        points = np.column_stack((point[0], point[1], point[2]))
        points_from_point_in_plane = points - point_in_plane
        proj_onto_normal_vector = np.dot(points_from_point_in_plane, normal_vector)
        proj_onto_plane = (points_from_point_in_plane - proj_onto_normal_vector[:, None] * normal_vector)

        point_on_surface = point_in_plane + proj_onto_plane
        point_on_surface[0][2] = a * point_on_surface[0][0] + b * point_on_surface[0][1] + c
        return point_on_surface

    def _get_flatten_3d_points(self):
        flatten_points = [] 
        for point in self.triangulated_3d_points[:,0,:]:
            flatten_points.append(self._project_point_to_plane(point))

        self._flatten_3d_points = np.array(flatten_points)

    def _get_track_length(self):
        length = 0.0
        points = self.flatten_3d_points[:,0,:]

        for i in range(1, len(points)):
            length += ((points[i-1][0] - points[i][0])**2 + (points[i-1][1] - points[i][1])**2 + (points[i-1][2] - points[i][2])**2)**0.5
        self._length = length

    def _get_parabola(self):
        x = [point[0] for point in self.flatten_3d_points[:,0,:]]
        y = [point[1] for point in self.flatten_3d_points[:,0,:]]
        z, residuals, _, _, _ = np.polyfit(x, y, 2, full=True)
        
        self._parabola = (*z, residuals)

    def _get_parameters(self):
        grid_height = self.grid_height
        g = 9.81 # m/s^2

        xxs = np.roots((self.parabola[0], self.parabola[1], self.parabola[2] - grid_height))

        V_y0 = (2 * g * np.abs(grid_height * 10**-3 - np.min(self.flatten_3d_points[:,0,1]) * 10**-3))**0.5 
        V_x = (g / (2 * self.parabola[0] / 10**-3))**0.5
        V_y = V_x * self.parabola[1] + (xxs[0] * 10**-3 * g) / V_x
        V0 = (V_x**2 + V_y**2)**0.5

        if np.isreal(V_y):
            alpha = np.rad2deg(np.arctan2(V_y, V_x))
        else:
            alpha = 90 
        #V_exp = self.length * 10**-3 / exposure_time

        r = self.particle_radius
        ro = self.particle_density # kg/m^3
        vol = 1/3 * np.pi * r**3
        m = ro*vol

        d = grid_height # 10 * 10**-3 # m
        U = self.voltage 
        # E = U / d

        q = (m * V0 ** 2 / 2 + m * g * d) / U

        valid = np.isreal(V_y) and np.isreal(V0) and np.isreal(q) 

        self._parameters = (V_y0, V_y, V_x, V0, alpha, r, m, q, valid)


    def draw_3d_figure(self):
        points = self.flatten_3d_points

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        ax.scatter(points[:,0,0], points[:,0,1], points[:,0,2], marker='o')

        ax.set_zlim(min(points[:,0,2]) - 2, max(points[:,0,2]) + 2)

        ax.set_xlabel('X , mm')
        ax.set_ylabel('Y , mm')
        ax.set_zlabel('Z , mm')

        ax.view_init(-70, -90)
        
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='png', bbox_inches='tight')
        plt.close()
        image_stream.seek(0)
        return image_stream

    def draw_2d_figure(self):
        points = self.flatten_3d_points

        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        ax.scatter(points[:,0,0], points[:,0,1], marker='o', facecolors='none', edgecolors='r')

        x = np.linspace(min(points[:,0,0]), max(points[:,0,0]), 1000)
        y = self.parabola[0]*x**2 + self.parabola[1]*x + self.parabola[2]

        ax.plot(x, y)

        ax.set_xlabel('X, mm')
        ax.set_ylabel('Y, mm')
        ax.grid(True)
        ax.invert_yaxis()
        
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='png', bbox_inches='tight')
        plt.close()
        image_stream.seek(0)
        return image_stream
    
    