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

def points_toward_cell(centre_cell,centre_face,normale) :
    vec = centre_cell -centre_face
    return np.dot(vec,normale)>0


class MeshError(Exception):
    """Classe d'erreur pour le maillage"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class GeomObject() :
    def __init__(self,i, nodes_index,nodes:np.array):
        """Initialisation de l'objet géométrique"""
        self.indice_global = i 
        self.nodes_index = np.array(nodes_index)
        self.centroid = np.mean(nodes[nodes_index], axis=0)

class Face(GeomObject) :
    def __init__(self, i, nodes_index,cells_index,nodes:np.array):
        assert len(nodes_index) == 2 , "Face must have exactly 2 nodes in a 2D mesh"
        if len(cells_index) == 0 :
            raise MeshError("La face n'a pas de cellules contiguës")
        super().__init__(i, nodes_index,nodes)
        self.normal = self.get_normal(nodes)
        self.surface = self.length(nodes) * self.get_normal(nodes)
        self.cells = np.array(cells_index,dtype=int)


    def __repr__(self):
        return f"Face {self.indice_global} with nodes {self.nodes_index} and normal {self.normal}"
    
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
        normale = self.get_normal(nodes)
        if len(self.cells) == 1:
            self.owner = self.cells[0]
            self.neighbour = -1
            # Ensure the surface vector points outward the owner cell
            
            if points_toward_cell(cells[self.cells[0]].centroid,self.centroid,normale):
                self.surface = -self.length(nodes) * self.get_normal(nodes)
        elif len(self.cells) == 2:
            cell0 = cells[self.cells[0]]
            cell1 = cells[self.cells[1]]
            c0 = cell0.centroid
            c1 = cell1.centroid

            if points_toward_cell(cells[self.cells[0]].centroid,self.centroid,normale):
                self.owner = self.cells[1]
                self.neighbour = self.cells[0]
            else:
                self.owner = self.cells[0]
                self.neighbour = self.cells[1]

    def length(self,nodes:np.array)->float:
        """Calcul de la longueur de la face"""
        nodes = nodes[self.nodes_index]
        return np.linalg.norm(nodes[0] - nodes[1])
    

class Cell(GeomObject) :

    def __init__(self,i:int,faces_index:list,nodes_index:list,voisins:list,nodes:np.array):
        super().__init__(i, nodes_index,nodes)
        self.faces = np.array(faces_index,dtype=int)
        self.voisins = np.array(voisins,dtype=int)
        self.sort_nodes(nodes)
        self.volume = self.get_volume(nodes)
        self.is_boundary = len(self.voisins) < len(self.faces)

    def __repr__(self):
        return f"Cell {self.indice_global} : \nfaces : {self.faces}\nNoeuds : {self.nodes_index}\nVoisins : {self.voisins}\n"
    
    def sort_nodes(self,nodes:np.array)->None:
        """Tri des noeuds de la cellule dans le sens trigonométrique"""
        # On prend le premier noeud comme référence
        nodes = nodes[self.nodes_index]
        c_node = centre(nodes)

        # On calcule les angles
        angles = np.zeros(len(nodes))
        for i in range(len(nodes)):
            # On calcule l'angle entre le noeud et le centre de la cellule
            angles[i] = np.arctan2(nodes[i][1]-c_node[1], nodes[i][0]-c_node[0])
        # On trie les noeuds en fonction des angles
        sorted_indices = np.argsort(angles)
        self.nodes_index = self.nodes_index[sorted_indices]

    def get_volume(self,nodes:np.array)->float:
        nodes = nodes[self.nodes_index]
        V = surface_triangle(nodes[0], nodes[1], nodes[2])
        for i in range(3, len(nodes)):
            V += surface_triangle(nodes[0], nodes[i-1], nodes[i])
        return V
    
    def contains(self, x,y,nodes)->bool:
        """Vérifie si le point (x,y) est dans la cellule"""
        node = np.array([x,y]) 
        angles = np.zeros(len(self.nodes_index))
        for i in range(len(self.nodes_index)):
            i_node = self.nodes_index[i]
            # On calcule l'angle entre le noeud, le centre de la cellule et son horizontale
            angles[i] = np.arctan2(node[1]-nodes[i_node][1], node[0]-nodes[i_node][0])
        delta = abs(max(angles) - min(angles))
        if delta >= np.pi:
            return True
        else:
            return False


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
        step = 0
        for face in self.faces:
            step += np.linalg.norm(face.surface)
        self.mean_step = step/len(self.faces)

    def read_mesh(self, filename:str,faces_per_cell = 4)->None:
        lines = open(filename, 'r').readlines()
        n,f,c = get_paraph(lines)
        n_faces = c-f+1
        n_cells = len(lines)-c+1

        nodes = self.mesh_read_nodes(filename)

        faces_ref = np.zeros((n_faces, 2 ), dtype=int)
        cells_of_faces = np.zeros((n_faces, 2 ), dtype=int)
        cells_of_faces[:,:] = -1
        cells_ref = np.zeros((n_cells, faces_per_cell), dtype=int)
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
                for j in range(faces_per_cell):
                    face_index = int(words[j+1])
                    cells_ref[c_count,j] = face_index
                    cells_of_faces[face_index][np.where(cells_of_faces[face_index]==-1)[0][0]]=c_count
                        

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
    

    def span(self):
        """Calcul de l'étendue du maillage"""
        x_min = np.min(self.nodes[:,0])
        x_max = np.max(self.nodes[:,0])
        y_min = np.min(self.nodes[:,1])
        y_max = np.max(self.nodes[:,1])
        return x_min, x_max, y_min, y_max
    
    def plot_mesh(self, ax=None) -> None:
        """Enhanced visualization of the mesh."""
        offset = 0.02
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))  # Larger figure size for better visibility

        # Plot cells and faces
        for i, cell in enumerate(self.cells):
            faces = self.faces[cell.faces]
            for f in faces:
                nodes = self.nodes[f.nodes_index]
                centre = f.centroid
                ax.plot(nodes[:, 0], nodes[:, 1], color="black", linewidth=1)  # Cell boundaries
                ax.text(centre[0]-offset, centre[1]-offset, f"{f.indice_global}", color="k", fontsize=8, ha="center", va="center")  # face labels

        # Plot nodes
        for i, node in enumerate(self.nodes):
            ax.plot(node[0], node[1], 'ro', markersize=4)  # Nodes as red dots
            ax.text(node[0]+offset, node[1]+offset, f"{i}", color="red", fontsize=8, ha="center", va="center")  # Node labels

        # Set aspect ratio and grid
        ax.set_aspect('equal', adjustable='box')
        ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Mesh Visualization", fontsize=14, fontweight="bold")
        return ax

    def complete_plot(self,ax=None) -> None:
        """Enhanced complete plot with normals and labels."""
        ax = self.plot_mesh(ax=ax)
        ax.set_title("Mesh with Normals", fontsize=14, fontweight="bold")
        offset = 0.05

        # Add normals to faces
        for face in self.faces:
            normal = face.get_normal(self.nodes)
            centroid = face.centroid
            ax.quiver(
                centroid[0], centroid[1], normal[0], normal[1],
                angles='xy', scale_units='xy', scale=2, color='blue', alpha=0.5, 
                label="Face Normal", width=0.005
            )

        # Add cell centroids
        for i, cell in enumerate(self.cells):
            ax.plot(cell.centroid[0], cell.centroid[1], 'kx', markersize=6)  # Centroids as black crosses
            ax.text(cell.centroid[0]+offset, cell.centroid[1]+offset, f"C{i}", color="green", fontsize=10, ha="center", va="center")  # Cell labels

        plt.legend(["Cell Boundaries", "Face Normals", "Nodes", "Cell Centroids"], loc="upper right", fontsize=8)
    
    def find_cell(self,x:float,y:float)->int:
        for i in range(len(self.cells)):
            cell = self.cells[i]
            if cell.contains(x,y,self.nodes):
                return i


if __name__ ==  "__main__" :
    
    node1 = np.array([0, 0])
    node2 = np.array([1, 0])
    node3 = np.array([0, 1])
    nodes_ligne = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [2, 1], [2, 0]])
    f0 = Face(0, [0, 1], [0], nodes_ligne)
    f1 = Face(1, [1, 2], [0, 1], nodes_ligne)
    f2 = Face(2, [0, 2], [0], nodes_ligne)
    f3 = Face(3, [1, 3], [1, 3], nodes_ligne)
    f4 = Face(4, [2, 3], [1, 2], nodes_ligne)
    f5 = Face(5, [2, 4], [2], nodes_ligne)
    f6 = Face(6, [4, 5], [2], nodes_ligne)
    f7 = Face(7, [3, 5], [2], nodes_ligne)
    f8 = Face(8, [3, 6], [3], nodes_ligne)
    f9 = Face(9, [6, 7], [3], nodes_ligne)
    f10 = Face(10, [1, 7], [3], nodes_ligne)

    faces_ligne = np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])

    cell0 = Cell(0, [0, 1, 2], [0, 1, 2], [1], nodes_ligne)
    cell1 = Cell(1, [1, 3, 4], [1, 3, 2], [0, 2], nodes_ligne)
    cell2 = Cell(2, [4, 5, 6, 7], [2, 3, 4, 5], [1], nodes_ligne)
    cell3 = Cell(3, [3, 8, 9, 10], [1, 3, 6, 7], [1], nodes_ligne)

    cells_ligne = np.array([cell0, cell1, cell2, cell3])

    mesh = Mesh(cells=cells_ligne, nodes=nodes_ligne, faces=faces_ligne)

    mesh.plot_mesh()  # Affiche le maillage
    plt.show()