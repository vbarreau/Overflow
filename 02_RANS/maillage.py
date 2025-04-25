import numpy as np 
import matplotlib.pyplot as plt 

def get_nodes_from_faces(faces_index:list)->list:
    """Récupération des noeuds à partir des faces"""
    nodes = []
    for i in range(len(faces_index)):
        for j in range(len(faces_index[i].nodes)):
            if faces_index[i].nodes[j] not in nodes:
                nodes.append(faces_index[i].nodes[j])
    return nodes

def  mesh_read_nodes(filename:str)->np.array:
    """Lecture du maillage à partir d'un fichier"""
    lines = open(filename, 'r').readlines()
    nodes = []

    for line in lines:
        if line[0]=='n':
            nodes.append([float(x) for x in line[1:].split()])
    nodes = np.array(nodes)

    return nodes

global_nodes = mesh_read_nodes("D:/OneDrive/Documents/11-Codes/overflow/02_RANS/mesh.dat")


def surface_triangle(node1, node2, node3):
    """Calcul de la surface d'un triangle par le produit vectoriel"""
    a = node2 - node1
    b = node3 - node1
    return np.linalg.norm(np.cross(a,b)) / 2


def centre(nodes):  
    """Calcul du centre de gravité d'un ensemble de points"""
    return np.mean(nodes, axis=0)

class Face():
    def __init__(self, i:int, nodes_index:list, owner_cell:int, neighbour_cell:int):
        self.indice_global = i
        self.nodes = np.array(nodes_index)
        self.owner = owner_cell
        self.neighbour = neighbour_cell
        self.centroid = centre(global_nodes[nodes_index])
        self.surface = self.length()


    def __repr__(self):
        return f"Face {self.indice_global} : \n{global_nodes[self.nodes]}, #owner {self.owner}"
    
    def length(self):
        """Calcul de la longueur de la face"""
        nodes = global_nodes[self.nodes]
        return np.linalg.norm(nodes[0] - nodes[1])






class Cell():
    def __init__(self,i:int,faces_index:list,nodes_index:list,voisins:list):
        self.indice_global = i
        self.faces = np.array(faces_index)
        self.nodes = np.array(nodes_index)
        self.voisins = np.array(voisins)
        self.volume = self.get_volume()
        self.centroid = np.mean(global_nodes[self.nodes], axis=0)
        self.sort_nodes()

    def __repr__(self):
        return f"Cell {self.indice_global} : {self.faces} {self.nodes} {self.voisins}"
    
    def sort_nodes(self):
        """Tri des noeuds de la cellule dans le sens trigonométrique"""
        # On prend le premier noeud comme référence
        nodes = global_nodes[self.nodes]
        c_node = centre(nodes)

        # On calcule les angles
        angles = np.zeros(len(nodes))
        for i in range(len(nodes)):
            # On calcule l'angle entre le noeud et le centre de la cellule
            angles[i] = np.arctan2(nodes[i][1]-c_node[1], nodes[i][0]-c_node[0])
        # On trie les noeuds en fonction des angles
        sorted_indices = np.argsort(angles)
        self.nodes = self.nodes[sorted_indices]

    # def sort_faces(self):
    #     """Tri des faces de la cellule dans le sens trigonométrique"""
    #     new_faces = []
    #     for i in range(len(self.nodes)-1):
    #         i_f = np.where( nodes[i] in face.nodes)


    def get_volume(self):
        nodes = global_nodes[self.nodes]
        V = surface_triangle(nodes[0], nodes[1], nodes[2])
        for i in range(3, len(nodes)):
            V += surface_triangle(nodes[0], nodes[i-1], nodes[i])
        return V


class Mesh():
    def __init__(self, cells : np.array):
        self.cells = cells
        self.size = cells.shape[0]
     

if __name__ ==  "__main__" :
    # Exemple d'utilisation
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    face1 = Face(0, [0,1], 0, 1)
    face2 = Face(1, [1,2], 0, 2)
    face3 = Face(2, [0,2], 0, 3)
    cell = Cell(0, [1,2], [1,2,0], [0,1,2,3])
    print( "Volume cellule : " , cell.volume)  # Affiche la surface du triangle
    print("centre de la cellule : ",cell.centroid)  # Affiche le centre de gravité du triangle

    print(face1.__repr__())  # Affiche la face
    print(face1.surface)  # Affiche la surface de la face
    print(face2.centroid)  # Affiche le centre de gravité de la face