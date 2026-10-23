import trimesh, json, numpy as np

m = trimesh.load('Duck.dae', force='mesh')
print(m.vertices.shape, m.faces.shape, m.bounds)

uv = getattr(m.visual, 'uv', None)
if uv is None:
    uv = np.zeros((len(m.vertices), 2))
uv = np.asarray(uv).copy()
uv[:, 1] = 1.0 - uv[:, 1]   # flip v: image origin is top-left

out = {
    "vertices": m.vertices.tolist(),
    "faces": m.faces.tolist(),
    "vertex_colors": np.ones((len(m.vertices), 3)).tolist(),
    "vertex_normals": m.vertex_normals.tolist(),
    "vertex_uvs": uv.tolist(),
}
json.dump(out, open('duck_mesh.json', 'w'))
