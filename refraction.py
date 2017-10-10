import trimesh
import numpy as np
import math
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main():
    transform = np.eye(4)
    transform[1,3]=0.12
    print(transform)
    mesh_phantom = trimesh.primitives.Box(extents=np.array([0.12675,0.0579,0.0579]), transform=transform)
    # mesh_phantom.show()

    camera_a_origin = np.array([0,0.12,0.12])
    camera_b_origin = np.array([0,0,0])

    point_observed = np.array([0.05,0.12 + 0.025,0-0.025])

    modeler = RefractionModeler(camera_a_origin, camera_b_origin, mesh_phantom, 1.2, 1.0)

    real_point = modeler.solve_real_point_from_refracted(point_observed)
    print("Observed Point", point_observed, "Real Point", real_point)
    modeler.make_plot()
#     ray_origin = np.array([[0,0,0]])
#     ray_direction = normalize(np.array([[1,0.5,0.0]]))
#     ray_origins = []
#     ray_directions = []
#     ray_origins.append(ray_origin)
#     ray_directions.append(ray_direction)
#
#
#     ret = mesh.ray.intersects_location(ray_origin, ray_direction)
#     triangles, rays, locations = mesh.ray.intersects_id(ray_origin, ray_direction, return_locations=True)
#
#     distances = []
#     for i, location in enumerate(locations):
#         distance = np.linalg.norm(location - ray_origin)
#         distances.append(distance)
#
#     intersections_sorted = sorted(zip(distances, triangles, locations), key=lambda x: x[0], reverse=False)
#
#     triangle_nearest = intersections_sorted[0][1]
#     location_nearest = intersections_sorted[0][2]
#     normal_nearest = mesh.face_normals[triangle_nearest]
#
#     print("Location " +str(location_nearest) + " normal direction " + str(normal_nearest))
#
#     axis_rotation = np.cross(ray_direction, normal_nearest)
#     axis_rotation_norm = axis_rotation/np.linalg.norm(axis_rotation)
#     print("Rotation axis: " + str(axis_rotation_norm))
#
#
#     angle_incident = math.acos(np.dot(ray_direction, -normal_nearest) / (np.linalg.norm(ray_direction) * np.linalg.norm(-normal_nearest)))
#     angle = calculate_refraction_angle(angle_incident, 1.0, 1.2)
#     angle_diff = angle - angle_incident
#     print("Angle 1: " + str(angle_incident*180/np.pi) + " Angle 2: " + str(angle*180/np.pi) + " Angle delta: " + str(angle_diff*180/np.pi))
#
#     # rotation_euler = trimesh.transformations.euler_from_quaternion(trimesh.transformations.quaternion_about_axis(np.pi/4, axis_rotation_norm.T))
#     rotation_mat = trimesh.transformations.rotation_matrix(angle_diff, axis_rotation_norm.T)
#     print("Rotation mat: ", rotation_mat[0:3,0:3])
#     print("Ray direction: ", ray_direction.T)
#     ray_direction_refracted =  np.dot(rotation_mat[0:3,0:3], ray_direction.T)
#     print("Original direction: " + str(ray_direction) + " Rotated direction: " + str(ray_direction_refracted))
#     print(np.linalg.norm(ray_direction_refracted))
#
#     ray_origins.append(np.array([location_nearest]))
#     ray_directions.append(ray_direction_refracted.T)
#
#     print("origins: ", ray_origins)
#     print("directions: ", ray_directions)
#
#     x = []
#     y = []
#     z = []
#
#     x.append(ray_origins[0][0][0])
#     x.append(ray_origins[1][0][0])
#     x.append(ray_origins[1][0][0] + ray_directions[1][0][0])
#     print(x)
#
#     y.append(ray_origins[0][0][1])
#     y.append(ray_origins[1][0][1])
#     y.append(ray_origins[1][0][1] + ray_directions[1][0][1])
#     print(y)
#
#     z.append(ray_origins[0][0][2])
#     z.append(ray_origins[1][0][2])
#     z.append(ray_origins[1][0][2] + ray_directions[1][0][2])
#     print(z)
#
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot(x, y, z)
#     plt.axis('equal')
#     plt.show()
#
#     point, _ = rays_closest_point(np.array([0,0,0]), np.array([1,0,0]), np.array([-1,-1,0]), np.array([0,1,0]))
#     print("Point",point)
#
# def calculate_refraction_angle(angle_in, index_outer, index_inner):
#     angle_out = math.asin((index_outer/index_inner)*math.sin(angle_in))
#     print("Angle out", angle_out)
#     return angle_out
#
# def normalize(input):
#     return input/np.linalg.norm(input)

# def rays_closest_point(ray1_origin, ray1_direction, ray2_origin, ray2_direction):
#     # http://morroworks.com/Content/Docs/Rays%20closest%20point.pdf
#     c = ray2_origin- ray1_origin
#     D = ray1_origin + ray1_direction*((-np.dot(ray1_direction, ray2_direction)*np.dot(ray2_direction, c) + np.dot(ray1_direction, c)*np.dot(ray2_direction, ray2_direction)) /
#                                       np.dot(ray1_direction, ray1_direction)*np.dot(ray2_direction,ray2_direction) - np.dot(ray1_direction,ray2_direction)*np.dot(ray1_direction,ray2_direction))
#     E = ray2_origin + ray2_direction * ((np.dot(ray1_direction, ray2_direction) * np.dot(ray1_direction, c) - np.dot(ray2_direction, c) * np.dot(ray1_direction, ray1_direction)) /
#                                         np.dot(ray1_direction, ray1_direction) * np.dot(ray2_direction,ray2_direction) - np.dot(ray1_direction, ray2_direction) * np.dot(ray1_direction, ray2_direction))
#     return (D+E)*0.5, D, E

class RefractionModeler(object):
    def __init__(self, camera_a_origin, camera_b_origin, mesh_phantom, refractive_index_phantom, refractive_index_ambient):
        self.camera_a_origin = camera_a_origin
        self.camera_b_origin = camera_b_origin
        self.mesh_phantom = mesh_phantom
        self.refractive_index_phantom = refractive_index_phantom
        self.refractive_index_ambient = refractive_index_ambient

    def solve_real_point_from_refracted(self, point_observed):
        print("Origin A", self.camera_a_origin)
        print("Origin B", self.camera_b_origin)
        self.point_observed = point_observed

        camera_a_direction = self._normalize(point_observed - self.camera_a_origin)
        camera_b_direction = self._normalize(point_observed - self.camera_b_origin)

        print("Dir A",camera_a_direction)
        print("Dir B",camera_b_direction)

        location_nearest_a, normal_nearest_a = self._get_closest_intersection(self.camera_a_origin, camera_a_direction)
        location_nearest_b, normal_nearest_b = self._get_closest_intersection(self.camera_b_origin, camera_b_direction)

        print("Inter A", location_nearest_a)
        print("Inter B", location_nearest_b)

        print("Norm A", normal_nearest_a)
        print("Norm B", normal_nearest_b)

        camera_a_direction_refracted = self._get_refracted_direction(camera_a_direction, normal_nearest_a)
        camera_b_direction_refracted = self._get_refracted_direction(camera_b_direction, normal_nearest_b)

        print("New Dir A", camera_a_direction_refracted)
        print("New Dir B", camera_b_direction_refracted)

        self.real_point, _, _, delta = self._rays_closest_point(location_nearest_a, camera_a_direction_refracted, location_nearest_b, camera_b_direction_refracted)
        self.lines_a = np.concatenate(([self.camera_a_origin], [location_nearest_a],
                                 [location_nearest_a + 0.15*camera_a_direction_refracted]), axis=0)
        self.lines_b = np.concatenate(([self.camera_b_origin], [location_nearest_b],
                                 [location_nearest_b + 0.15 * camera_b_direction_refracted]), axis=0)

        print("Refraction error", np.linalg.norm(self.real_point - self.point_observed))

        return self.real_point

    def make_plot(self):
        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(self.lines_a[:, 0], self.lines_a[:, 1], self.lines_a[:, 2])
        ax.plot3D(self.lines_b[:, 0], self.lines_b[:, 1], self.lines_b[:, 2])
        ax.scatter(self.camera_a_origin[0], self.camera_a_origin[1], self.camera_a_origin[2])
        ax.scatter(self.camera_b_origin[0], self.camera_b_origin[1], self.camera_b_origin[2])
        ax.scatter(self.real_point[0], self.real_point[1], self.real_point[2], 'r')
        ax.scatter(self.point_observed[0], self.point_observed[1], self.point_observed[2], 'g')
        plt.axis('equal')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def _get_closest_intersection(self, ray_origin, ray_direction):
        # print(np.reshape(ray_origin,(3,1)))
        # print(np.reshape(ray_direction,(3,1)))
        triangles, rays, locations = self.mesh_phantom.ray.intersects_id(np.reshape(ray_origin,(1,3)), np.reshape(ray_direction,(1,3)), return_locations=True)
        # print(len(triangles))
        distances = []
        for i, location in enumerate(locations):
            distance = np.linalg.norm(location - ray_origin)
            distances.append(distance)
        intersections_sorted = sorted(zip(distances, triangles, locations), key=lambda x: x[0], reverse=False)
        print("Intersections sorted", intersections_sorted)
        triangle_nearest = intersections_sorted[0][1]
        location_nearest = intersections_sorted[0][2]
        print("Loc Nearest", location_nearest)
        normal_nearest = self.mesh_phantom.face_normals[triangle_nearest]
        return location_nearest, normal_nearest

    def _get_refracted_direction(self, ray_direction, normal_nearest):
        axis_rotation = np.cross(ray_direction, normal_nearest)
        axis_rotation_norm = axis_rotation / np.linalg.norm(axis_rotation)
        angle_incident = math.acos(
            np.dot(ray_direction, -normal_nearest) / (np.linalg.norm(ray_direction) * np.linalg.norm(-normal_nearest)))
        angle = self._calculate_refraction_angle(angle_incident, self.refractive_index_ambient, self.refractive_index_phantom)
        angle_diff = angle - angle_incident
        print(angle_diff)
        if abs(angle_diff) > 0.0:
            rotation_mat = trimesh.transformations.rotation_matrix(angle_diff, axis_rotation_norm.T)
        else:
            rotation_mat = np.eye(3)
        ray_direction_refracted = np.dot(rotation_mat[0:3, 0:3], ray_direction.T)
        return ray_direction_refracted

    def _rays_closest_point(self, ray1_origin, ray1_direction, ray2_origin, ray2_direction):
        # http://morroworks.com/Content/Docs/Rays%20closest%20point.pdf
        c = ray2_origin - ray1_origin
        D = ray1_origin + ray1_direction * (-np.dot(ray1_direction, ray2_direction) * np.dot(ray2_direction, c)
                                             + np.dot(ray1_direction, c) * np.dot(ray2_direction, ray2_direction)) / \
                          (np.dot(ray1_direction, ray1_direction) * np.dot(ray2_direction, ray2_direction)
                                            - np.dot(ray1_direction, ray2_direction) * np.dot(ray1_direction, ray2_direction))
        E = ray2_origin + ray2_direction * (np.dot(ray1_direction, ray2_direction) * np.dot(ray1_direction, c) - np.dot(
                                                ray2_direction, c) * np.dot(ray1_direction, ray1_direction)) / \
                          (np.dot(ray1_direction, ray1_direction) * np.dot(ray2_direction,ray2_direction)
                           - np.dot(ray1_direction, ray2_direction) * np.dot(ray1_direction, ray2_direction))
        return (D + E) * 0.5, D, E, np.linalg.norm(E-D)

    def _normalize(self, input):
        return np.array(input / np.linalg.norm(input))

    def _calculate_refraction_angle(self, angle_in, index_outer, index_inner):
        angle_out = math.asin((index_outer / index_inner) * math.sin(angle_in))
        # print("Angle out", angle_out)
        return angle_out


if __name__ == '__main__':
    main()
