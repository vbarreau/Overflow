from maillage import *

#                         0     1     2     3     4     5     6     7
nodes_ligne = np.array([[0,0],[1,0],[0,1],[1,1],[2,0],[2,1],[3,0],[3,1]])
faces_ligne = []

faces_ligne += [Mesh.Face(0,[0,1],[0],nodes_ligne)  ]# 0
faces_ligne += [Mesh.Face(1,[1,3],[0,1],nodes_ligne)]# 1
faces_ligne += [Mesh.Face(2,[0,2],[0],nodes_ligne)  ]# 2
faces_ligne += [Mesh.Face(3,[2,3],[0],nodes_ligne)  ]# 3
faces_ligne += [Mesh.Face(4,[1,4],[1],nodes_ligne)  ]# 4
faces_ligne += [Mesh.Face(5,[3,5],[1],nodes_ligne)  ]# 5
faces_ligne += [Mesh.Face(6,[4,5],[1,2],nodes_ligne)]# 6
faces_ligne += [Mesh.Face(7,[4,6],[2],nodes_ligne)  ]# 7
faces_ligne += [Mesh.Face(8,[5,7],[2],nodes_ligne)  ]# 8
faces_ligne += [Mesh.Face(9,[6,7],[2],nodes_ligne)  ]# 9
faces_ligne = np.array(faces_ligne)

cells_ligne = []
cells_ligne += [Mesh.Cell(0,[0,1,2,3],[0,1,2,3],[1],nodes_ligne)]
cells_ligne += [Mesh.Cell(1,[1,4,5,6],[1,3,4,5],[0,2],nodes_ligne)]
cells_ligne += [Mesh.Cell(2,[6,7,8,9],[4,5,6,7],[1],nodes_ligne)]
cells_ligne = np.array(cells_ligne)

mesh_ligne = Mesh(nodes=nodes_ligne, faces=faces_ligne,cells=cells_ligne)


if __name__ == "__main__":
    from inspect_mesh import MeshInspector
    inspector = MeshInspector(mesh_ligne)
    inspector.run()