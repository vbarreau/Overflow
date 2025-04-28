import numpy as np 
import matplotlib.pyplot as plt 

def surface_triangle(node1, node2, node3):
    """Calcul de la surface d'un triangle par le produit vectoriel"""
    a = node2 - node1
    b = node3 - node1
    return 0.5 * (a[0] * b[1] - a[1] * b[0])


def centre(nodes):  
    """Calcul du centre de gravité d'un ensemble de points"""
    return np.mean(nodes, axis=0)

def get_nodes_from_faces(faces_index:list)->list:
    """Récupération des noeuds à partir des faces"""
    nodes = []
    for i in range(len(faces_index)):
        for j in range(len(faces_index[i].nodes)):
            if faces_index[i].nodes[j] not in nodes:
                nodes.append(faces_index[i].nodes[j])
    return nodes



class Mesh:
    def __init__(self, filename):
        """Initialize the Mesh object and read nodes, faces, and cells."""
        self.nodes = self.mesh_read_nodes(filename)
        self.faces = self.mesh_read_faces(filename, self.nodes)
        self.cells = self.mesh_read_cells(filename, self.faces, self.nodes)

    def mesh_read_nodes(self, filename: str) -> np.array:
        """Read nodes from the mesh file."""
        lines = open(filename, 'r').readlines()
        nodes = []
        for line in lines:
            if line[0] == 'n':
                nodes.append([float(x) for x in line[1:].split()])
        return np.array(nodes)

    def mesh_read_faces(self, filename: str, nodes: np.array) -> list:
        """Read faces from the mesh file."""
        lines = open(filename, 'r').readlines()
        faces = []
        for i, line in enumerate(lines):
            if line[0] == 'f':
                index = len(faces)
                nodes_index = [int(x) for x in line[1:].split()]
                cells = []
                for j in range(i, len(lines)):
                    l = lines[j]
                    if l[0] == 'c' and f" {index} " in l:
                        cells.append(int(l[1]))
                faces.append(self.Face(index, nodes_index, cells, nodes))
        return faces

    def mesh_read_cells(self, filename: str, faces: list, nodes: np.array) -> list:
        """Read cells from the mesh file."""
        lines = open(filename, 'r').readlines()
        cells = []
        for i, line in enumerate(lines):
            if line[0] == 'c':
                index = len(cells)
                faces_index = [int(x) for x in line[2:].split()]
                nodes_index = []
                voisins = []
                for fi in faces_index:
                    f = faces[fi]
                    nodes_index += f.nodes.tolist()
                    for j in range(len(f.cells)):
                        if f.cells[j] != index:
                            voisins.append(f.cells[j])
                nodes_index = list(set(nodes_index))
                voisins = list(set(voisins))
                cells.append(self.Cell(index, faces_index, nodes_index, voisins, nodes))
        return cells

    class Face:
        def __init__(self, i: int, nodes_index: list, cells: list, nodes: np.array):
            self.indice_global = i
            self.nodes = np.array(nodes_index)
            self.centroid = centre(nodes[nodes_index])
            self.surface = self.length(nodes) * self.get_normal(nodes)
            self.cells = np.array(cells)

        def __repr__(self):
            return f"Face {self.indice_global} : \n{self.nodes}\n"

        def length(self, nodes):
            """Calculate the length of the face."""
            face_nodes = nodes[self.nodes]
            return np.linalg.norm(face_nodes[0] - face_nodes[1])

        def get_normal(self, nodes):
            """Calculate the normal vector of the face."""
            face_nodes = nodes[self.nodes]
            face_vector = (face_nodes[1] - face_nodes[0])
            normal = np.array([-face_vector[1], face_vector[0]])
            normal /= np.linalg.norm(normal)
            return normal

    class Cell:
        def __init__(self, i: int, faces_index: list, nodes_index: list, voisins: list, nodes: np.array):
            self.indice_global = i
            self.faces = np.array(faces_index)
            self.nodes = np.array(nodes_index)
            self.voisins = np.array(voisins)
            self.volume = self.get_volume(nodes)
            self.centroid = np.mean(nodes[self.nodes], axis=0)
            self.sort_nodes(nodes)
            self.T = 0
            self.v = np.array([0, 0])
            self.p = 0

        def __repr__(self):
            return f"Cell {self.indice_global} : \nfaces : {self.faces}\nNoeuds : {self.nodes}\nVoisins : {self.voisins}\nParametres: {self.T} K, {self.v} m/s, {self.p} Pa\n"

        def sort_nodes(self, nodes):
            """Sort the nodes of the cell in counterclockwise order."""
            cell_nodes = nodes[self.nodes]
            c_node = centre(cell_nodes)
            angles = np.arctan2(cell_nodes[:, 1] - c_node[1], cell_nodes[:, 0] - c_node[0])
            sorted_indices = np.argsort(angles)
            self.nodes = self.nodes[sorted_indices]

        def get_volume(self, nodes):
            """Calculate the volume of the cell."""
            cell_nodes = nodes[self.nodes]
            V = surface_triangle(cell_nodes[0], cell_nodes[1], cell_nodes[2])
            for i in range(3, len(cell_nodes)):
                V += surface_triangle(cell_nodes[0], cell_nodes[i - 1], cell_nodes[i])
            return V


# Example usage
if __name__ == "__main__":
    mesh = Mesh("D:/OneDrive/Documents/11-Codes/overflow/02_RANS/mesh.dat")
    print(mesh.cells[0])  # Print the first cell