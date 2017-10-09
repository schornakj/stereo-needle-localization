import trimesh
import numpy as np
import math
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main():
    transform = np.eye(4)
    transform[0,3]=1
    mesh = trimesh.primitives.Box(extents=np.array([1,2,3]), transform=transform)

    ray_origin = np.array([[0,0,0]])
    ray_direction = normalize(np.array([[1,0.5,0.0]]))
    ray_origins = []
    ray_directions = []
    ray_origins.append(ray_origin)
    ray_directions.append(ray_direction)


    ret = mesh.ray.intersects_location(ray_origin, ray_direction)
    triangles, rays, locations = mesh.ray.intersects_id(ray_origin, ray_direction, return_locations=True)

    distances = []
    for i, location in enumerate(locations):
        distance = np.linalg.norm(location - ray_origin)
        distances.append(distance)

    intersections_sorted = sorted(zip(distances, triangles, locations), key=lambda x: x[0], reverse=False)

    triangle_nearest = intersections_sorted[0][1]
    location_nearest = intersections_sorted[0][2]
    normal_nearest = mesh.face_normals[triangle_nearest]

    print("Location " +str(location_nearest) + " normal direction " + str(normal_nearest))

    axis_rotation = np.cross(ray_direction, normal_nearest)
    axis_rotation_norm = axis_rotation/np.linalg.norm(axis_rotation)
    print("Rotation axis: " + str(axis_rotation_norm))


    angle_incident = math.acos(np.dot(ray_direction, -normal_nearest) / (np.linalg.norm(ray_direction) * np.linalg.norm(-normal_nearest)))
    angle = calculate_refraction_angle(angle_incident, 1.0, 1.2)
    angle_diff = angle - angle_incident
    print("Angle 1: " + str(angle_incident*180/np.pi) + " Angle 2: " + str(angle*180/np.pi) + " Angle delta: " + str(angle_diff*180/np.pi))

    # rotation_euler = trimesh.transformations.euler_from_quaternion(trimesh.transformations.quaternion_about_axis(np.pi/4, axis_rotation_norm.T))
    rotation_mat = trimesh.transformations.rotation_matrix(angle_diff, axis_rotation_norm.T)
    print("Rotation mat: ", rotation_mat[0:3,0:3])
    print("Ray direction: ", ray_direction.T)
    ray_direction_refracted =  np.dot(rotation_mat[0:3,0:3], ray_direction.T)
    print("Original direction: " + str(ray_direction) + " Rotated direction: " + str(ray_direction_refracted))
    print(np.linalg.norm(ray_direction_refracted))

    ray_origins.append(np.array([location_nearest]))
    ray_directions.append(ray_direction_refracted.T)

    print("origins: ", ray_origins)
    print("directions: ", ray_directions)

    x = []
    y = []
    z = []

    x.append(ray_origins[0][0][0])
    x.append(ray_origins[1][0][0])
    x.append(ray_origins[1][0][0] + ray_directions[1][0][0])
    print(x)

    y.append(ray_origins[0][0][1])
    y.append(ray_origins[1][0][1])
    y.append(ray_origins[1][0][1] + ray_directions[1][0][1])
    print(y)

    z.append(ray_origins[0][0][2])
    z.append(ray_origins[1][0][2])
    z.append(ray_origins[1][0][2] + ray_directions[1][0][2])
    print(z)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z)
    plt.axis('equal')
    plt.show()

    point, _ = rays_closest_point(np.array([0,0,0]), np.array([1,0,0]), np.array([-1,-1,0]), np.array([0,1,0]))
    print("Point",point)

def calculate_refraction_angle(angle_in, index_outer, index_inner):
    angle_out = math.asin((index_outer/index_inner)*math.sin(angle_in))
    print("Angle out", angle_out)
    return angle_out

def normalize(input):
    return input/np.linalg.norm(input)

# def rays_closest_point(ray1_origin, ray1_direction, ray2_origin, ray2_direction):
#     # http://morroworks.com/Content/Docs/Rays%20closest%20point.pdf
#     c = ray2_origin- ray1_origin
#     D = ray1_origin + ray1_direction*((-np.dot(ray1_direction, ray2_direction)*np.dot(ray2_direction, c) + np.dot(ray1_direction, c)*np.dot(ray2_direction, ray2_direction)) /
#                                       np.dot(ray1_direction, ray1_direction)*np.dot(ray2_direction,ray2_direction) - np.dot(ray1_direction,ray2_direction)*np.dot(ray1_direction,ray2_direction))
#     E = ray2_origin + ray2_direction * ((np.dot(ray1_direction, ray2_direction) * np.dot(ray1_direction, c) - np.dot(ray2_direction, c) * np.dot(ray1_direction, ray1_direction)) /
#                                         np.dot(ray1_direction, ray1_direction) * np.dot(ray2_direction,ray2_direction) - np.dot(ray1_direction, ray2_direction) * np.dot(ray1_direction, ray2_direction))
#     return (D+E)*0.5, D, E

class RefractionModeler(object):
    def __init__(self, camera_a_tform, camera_b_tform, phantom_tform, phantom_dims, phantom_refractive_index):
        self.camera_a_origin = camera_a_tform[0:2,3]
        self.camera_b_origin = camera_b_tform[0:2, 3]
        self.mesh_phantom = trimesh.primitives.Box(extents=phantom_dims, transform=phantom_tform)
        self.index_refraction = phantom_refractive_index

    def solve_real_point_from_refracted(self, point_observed):
        camera_a_direction = self._normalize(point_observed - self.camera_a_origin)
        camera_b_direction = self._normalize(point_observed - self.camera_b_origin)

        location_nearest_a, location_nearest_a = self._get_closest_intersection(self.camera_a_origin, camera_a_direction)
        location_nearest_b, location_nearest_b = self._get_closest_intersection(self.camera_b_origin, camera_b_direction)


        return 0 # not yet implemented

    def _get_closest_intersection(self, ray_origin, ray_direction):
        triangles, rays, locations = self.mesh_phantom.ray.intersects_id(ray_origin, ray_direction, return_locations=True)
        distances = []
        for i, location in enumerate(locations):
            distance = np.linalg.norm(location - ray_origin)
            distances.append(distance)
        intersections_sorted = sorted(zip(distances, triangles, locations), key=lambda x: x[0], reverse=False)
        triangle_nearest = intersections_sorted[0][1]
        location_nearest = intersections_sorted[0][2]
        normal_nearest = self.mesh_phantom.face_normals[triangle_nearest]
        return location_nearest, normal_nearest

    def _rays_closest_point(self, ray1_origin, ray1_direction, ray2_origin, ray2_direction):
        # http://morroworks.com/Content/Docs/Rays%20closest%20point.pdf
        c = ray2_origin - ray1_origin
        D = ray1_origin + ray1_direction * ((-np.dot(ray1_direction, ray2_direction) * np.dot(ray2_direction,
                                                                                              c) + np.dot(
            ray1_direction, c) * np.dot(ray2_direction, ray2_direction)) /
                                            np.dot(ray1_direction, ray1_direction) * np.dot(ray2_direction,
                                                                                            ray2_direction) - np.dot(
            ray1_direction, ray2_direction) * np.dot(ray1_direction, ray2_direction))
        E = ray2_origin + ray2_direction * ((
                                            np.dot(ray1_direction, ray2_direction) * np.dot(ray1_direction, c) - np.dot(
                                                ray2_direction, c) * np.dot(ray1_direction, ray1_direction)) /
                                            np.dot(ray1_direction, ray1_direction) * np.dot(ray2_direction,
                                                                                            ray2_direction) - np.dot(
            ray1_direction, ray2_direction) * np.dot(ray1_direction, ray2_direction))
        return (D + E) * 0.5, D, E

    def _normalize(self, input):
        return input / np.linalg.norm(input)


if __name__ == '__main__':
    main()
