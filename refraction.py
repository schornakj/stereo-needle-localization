import trimesh
import numpy as np

transform = np.eye(4)
transform[0,3]=1
mesh = trimesh.primitives.Box(extents=np.array([1,2,3]), transform=transform)

ray_origin = np.array([[0,0,0]])
ray_direction = np.array([[1,0,0]])

ret = mesh.ray.intersects_location(ray_origin, ray_direction)
triangles, rays, locations = mesh.ray.intersects_id(ray_origin, ray_direction, return_locations=True)

distances = []
for i, location in enumerate(locations):
    distance = np.linalg.norm(location - ray_origin)
    distances.append(distance)

intersections_sorted = sorted(zip(distances, triangles, locations), key=lambda x: x[0], reverse=False)

triangle_nearest = intersections_sorted[0][1]
location_nearest = intersections_sorted[0][2]

print("Location " +str(location_nearest) + " normal direction " + str(mesh.face_normals[triangle_nearest]))
