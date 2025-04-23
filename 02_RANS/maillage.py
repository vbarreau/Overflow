import numpy as np 
import matplotlib.pyplot as plt 

def surface_triangle(node1, node2, node3):
    """Calcul de la surface d'un triangle par le produit vectoriel"""
    a = node2 - node1
    b = node3 - node1
    return np.linalg.norm(np.cross(a,b)) / 2


class Cell():
    def __init__(self,i:int,faces:list,nodes:list,voisins:list):
        self.indice_global = i
        self.faces = faces
        self.nodes = nodes
        self.voisins = voisins
        self.volume = self.get_volume()
        self.centroid = np.mean(self.nodes, axis=0)
        self.sort_nodes()

    def __repr__(self):
        return f"Cell {self.indice_global} : {self.faces} {self.nodes} {self.voisins}"
    
    def sort_nodes(self):
        """Tri des noeuds de la cellule dans le sens trigonométrique"""
        # On prend le premier noeud comme référence
        ref = self.nodes[0]
        # On calcule les angles
        angles = np.arctan2(self.nodes[:,1] - ref[1], self.nodes[:,0] - ref[0])
        # On trie les noeuds en fonction des angles
        sorted_indices = np.argsort(angles)
        self.nodes = self.nodes[sorted_indices]


    def get_volume(self):
        V = surface_triangle(self.nodes[0], self.nodes[1], self.nodes[2])
        for i in range(3, len(self.nodes)):
            V += surface_triangle(self.nodes[0], self.nodes[i-1], self.nodes[i])
        return V


class Mesh():
    def __init__(self, cells : np.array):
        self.cells = cells
        self.size = cells.shape[0]
     