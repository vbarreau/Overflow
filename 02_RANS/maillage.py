import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm

def surface_triangle(node1, node2, node3):
    """Calcul de la surface d'un triangle par le produit vectoriel"""
    a = node2 - node1
    b = node3 - node1
    return 0.5 * (a[0] * b[1] - a[1] * b[0])


def centre(nodes):  
    """Calcul du centre de gravité d'un ensemble de points"""
    return np.mean(nodes, axis=0)

# def get_nodes_from_faces(faces_index:list,)->list:
#     """Récupération des noeuds à partir des faces"""
#     nodes = []
#     for i in range(len(faces_index)):
#         for j in range(len(faces_index[i].nodes)):
#             if faces_index[i].nodes[j] not in nodes:
#                 nodes.append(faces_index[i].nodes[j])
#     return nodes

def get_paraph(meshdat) :
    n = None
    f = None
    c = None    
    for i in range(len(meshdat)):
        if meshdat[i][0] == 'n' and n is None:
            n = i 
        elif meshdat[i][0] == 'f' and f is None:
            f = i
        elif meshdat[i][0] == 'c' and c is None:
            c = i
    return n,f,c




class Mesh():

    def __init__(self, **kwargs):
        """Initialize the Mesh object and read nodes, faces, and cells."""
        if len(kwargs) == 1 and isinstance(kwargs['filename'], str):
            filename = kwargs['filename']
            self.nodes, self.faces, self.cells = self.read_mesh(filename)

        elif isinstance(kwargs['cells'], np.ndarray):
            self.nodes = kwargs['nodes']
            self.faces = kwargs['faces']
            self.cells = kwargs['cells']
        else:
            raise ValueError("Invalid arguments. Provide either a filename or cells, nodes, and faces arrays.")

        self.set_mesh_volume()
        for face in self.faces:
            face.set_owner(self.nodes, self.cells)
        self.size = len(self.cells)

    def read_mesh(self, filename:str)->None:
        lines = open(filename, 'r').readlines()
        n,f,c = get_paraph(lines)
        nodes = self.mesh_read_nodes(filename)
        faces_ref = np.zeros((c-f+1, 2 ), dtype=int)
        cells_of_faces = np.zeros((c-f+1, 2 ), dtype=int)
        cells_of_faces[:,:] = -1
        cells_ref = np.zeros((len(lines)-c+1, 3), dtype=int)
        n_count = 0
        f_count = 0
        for i in tqdm(range(len(lines)), desc="Lecture du maillage"):
            line = lines[i]
            if line[0]=='n' :
                point = [float(x) for x in line[1:].split()]
                nodes[n_count,0] = point[0]
                nodes[n_count,1] = point[1]
                n_count += 1
            elif line[0]=='f':
                words = line.split()
                faces_ref[f_count,0] = int(words[1])
                faces_ref[f_count,1] = int(words[2])
                f_count += 1
            elif line[0]=='c':
                words = line.split()
                c_count = int(words[0][1:])
                cells_ref[c_count,0] = int(words[1])
                cells_ref[c_count,1] = int(words[2])
                cells_ref[c_count,2] = int(words[3])
                for j in range(3):
                    cells_of_faces[cells_ref[c_count,j]] = c_count

                c_count += 1

        # On enlève les lignes vides
        faces_ref = faces_ref[~np.all(faces_ref == 0, axis=1)]
        faces = np.zeros((len(faces_ref)), dtype=self.Face)
        for i in range(len(faces_ref)):
            faces[i] = self.Face(i, faces_ref[i],cells_of_faces[i] ,nodes)

        cells_ref = cells_ref[~np.all(cells_ref == 0, axis=1)]
        cells = np.zeros((len(cells_ref)), dtype=self.Cell)
        for i in range(len(cells_ref)):
            nodes_of_cell = []
            for j_face in cells_ref[i]:
                # On récupère les indices des noeuds de la face
                nodes_of_cell += faces[j_face].nodes_index.tolist()
            list(set(nodes_of_cell))
            # On cherche les voisines de la cellule
            voisins = []
            for j in range(3):
                if cells_of_faces[cells_ref[i,j]][0] != -1:
                    voisins.append(cells_of_faces[cells_ref[i,j]][0])
            cells[i] = self.Cell(i, cells_ref[i], nodes_of_cell,voisins,nodes)
        return nodes, faces, cells



    def  mesh_read_nodes(self,filename:str)->np.array:
        """Lecture du maillage à partir d'un fichier"""
        lines = open(filename, 'r').readlines()
        nodes = []

        for i in range(len(lines)):
            line = lines[i]
            if line[0]=='n':
                nodes.append([float(x) for x in line[1:].split()])
        nodes = np.array(nodes)

        return nodes
    


    def set_mesh_volume(self)->None:
        """Calcul du volume de chaque cellule"""
        self.cell_volume = np.zeros(len(self.cells))
        for i in range(len(self.cells)):
            cell = self.cells[i]
            self.cell_volume[i] = cell.volume
    
    class Face():
        def __init__(self, i:int, nodes_index:list, cells_index,nodes):

            self.indice_global = i
            self.nodes_index = np.array(nodes_index)
            self.centroid = centre(nodes[nodes_index])
            self.surface = self.length(nodes) * self.get_normal(nodes)
            self.cells = np.array(cells_index,dtype=int)
            if len(cells_index) == 0 :
                raise MeshError("La face n'a pas de cellules contiguës")
            assert len(nodes_index) == 2 , "La face n'a pas 2 noeuds"

        def __repr__(self):
            return f"Face {self.indice_global} : \n{self.nodes_index}\nCentre : {self.centroid}\nSurface : {self.surface}\n"
        
        def length(self,nodes:np.array)->float:
            """Calcul de la longueur de la face"""
            nodes = nodes[self.nodes_index]
            return np.linalg.norm(nodes[0] - nodes[1])
        
        def get_normal(self,nodes:np.array)->np.array:
            """Calcul de la normale à la face"""
            face_nodes = nodes[self.nodes_index]
            face_vector = (face_nodes[1] - face_nodes[0])
            # On calcule le vecteur normal à la face
            normal = np.array([-face_vector[1], face_vector[0]],dtype=float)
            # On normalise le vecteur
            normal /= np.linalg.norm(normal)
            return normal
        
        def set_owner(self,nodes:np.array,cells)->None:
            """Set le propriétaire de la face"""
            self.owner = -1
            try : 
                self.neighbour = self.cells[0]
            except :
                raise MeshError("La face n'a pas de cellule contiguë")
            
            if len(self.cells) == 1:
                self.owner = self.cells[0]
                self.neighbour = -1
            elif len(self.cells) == 2:
                cell0 = cells[self.cells[0]]
                cell1 = cells[self.cells[1]]
                c0 = cell0.centroid
                c1 = cell1.centroid

                vec = self.centroid + self.get_normal(nodes) * 0.1
                d0 = np.linalg.norm(c0 - vec)
                d1 = np.linalg.norm(c1 - vec)
                if d0 < d1:
                    self.owner = self.cells[1]
                    self.neighbour = self.cells[0]
                else:
                    self.owner = self.cells[0]
                    self.neighbour = self.cells[1]


    class Cell():
        def __init__(self,i:int,faces_index:list,nodes_index:list,voisins:list,nodes:np.array):
            self.indice_global = i
            self.faces = np.array(faces_index)
            self.nodes = np.array(nodes_index)
            self.voisins = np.array(voisins)
            self.centroid = np.mean(nodes[self.nodes], axis=0)
            self.sort_nodes(nodes)
            self.volume = self.get_volume(nodes)


        def __repr__(self):
            return f"Cell {self.indice_global} : \nfaces : {self.faces}\nNoeuds : {self.nodes}\nVoisins : {self.voisins}\n"
        
        def sort_nodes(self,nodes:np.array)->None:
            """Tri des noeuds de la cellule dans le sens trigonométrique"""
            # On prend le premier noeud comme référence
            nodes = nodes[self.nodes]
            c_node = centre(nodes)

            # On calcule les angles
            angles = np.zeros(len(nodes))
            for i in range(len(nodes)):
                # On calcule l'angle entre le noeud et le centre de la cellule
                angles[i] = np.arctan2(nodes[i][1]-c_node[1], nodes[i][0]-c_node[0])
            # On trie les noeuds en fonction des angles
            sorted_indices = np.argsort(angles)
            self.nodes = self.nodes[sorted_indices]


        def get_volume(self,nodes:np.array)->float:
            nodes = nodes[self.nodes]
            V = surface_triangle(nodes[0], nodes[1], nodes[2])
            for i in range(3, len(nodes)):
                V += surface_triangle(nodes[0], nodes[i-1], nodes[i])
            return V
        
    def span(self):
        """Calcul de l'étendue du maillage"""
        x_min = np.min(self.nodes[:,0])
        x_max = np.max(self.nodes[:,0])
        y_min = np.min(self.nodes[:,1])
        y_max = np.max(self.nodes[:,1])
        return x_min, x_max, y_min, y_max
    
    def plot_mesh(self,ax=None)->None:
        """Affichage du maillage"""
        if ax is None:
            fig, ax = plt.subplots()
        for i in range(len(self.cells)):
            cell = (self.cells[i])
            faces = self.faces[cell.faces]
            for f in faces :
                nodes = self.nodes[f.nodes_index]
                ax.plot(nodes[:,0], nodes[:,1], 'k-')
                ax.fill(nodes[:,0], nodes[:,1], alpha=0.2)
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            ax.plot(node[0], node[1], 'ro', markersize=1)
        ax.set_aspect('equal', adjustable='box')
        return ax   
    
    def complete_plot(self) -> None :
        ax = self.plot_mesh()
        ax.set_title("Maillage")
        for face in self.faces:
            normal = face.get_normal(self.nodes)
            centroid = face.centroid
            ax.quiver(centroid[0], centroid[1], normal[0], normal[1], angles='xy', scale_units='xy', scale=1, color='blue')
        return ax

class MeshError(Exception):
    """Classe d'erreur pour le maillage"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


if __name__ ==  "__main__" :
    mesh = Mesh("D:/OneDrive/Documents/11-Codes/overflow/02_RANS/circle_mesh.dat")
    cell = mesh.cells[0]
    print( "Volume cellule : " , cell.volume)  # Affiche la surface du triangle
    print("centre de la cellule : ",cell.centroid)  # Affiche le centre de gravité du triangle
    
    face1 = mesh.faces[0]
    print(face1.__repr__())  
    print(f"normale 1 : {face1.get_normal(mesh.nodes)}")
    print(f"Owner face 1 : {face1.owner}\n")  
    face2 = mesh.faces[1]
    print(face2.__repr__())
    print(f"normale 2 : {face2.get_normal(mesh.nodes)}")
    print(f"Owner face 2 : {face2.owner}")
    print()
    
    print(cell.__repr__())  # Affiche la cellule

    mesh.plot_mesh()  # Affiche le maillage