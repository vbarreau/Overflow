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
        self.distToWall = None


    def next_object(self,dir : np.ndarray, mesh) -> int:
        """Renvoie l'indice de l'objet voisin dans la direction donnée"""
        # TODO
        pass

    def distance_from(self,other)->float:
        """Calcul de la distance entre deux objets géométriques"""
        return np.linalg.norm(self.centroid - other.centroid)

    


class Face(GeomObject) :
    def __init__(self, i, nodes_index,cells_index,nodes:np.array):
        """Initialaze a Face object.\n
        **i** : global index for the face\n
        **nodes_index**: list of indices of the nodes forming the face\n
        **cells_index**: list of indices of the cells adjacent to the face.\n
        **nodes**: array of all nodes' coordinates in the mesh"""
        assert len(nodes_index) == 2 , "Face must have exactly 2 nodes in a 2D mesh"
        if len(cells_index) == 0 :
            raise MeshError("La face n'a pas de cellules contiguës")
        super().__init__(i, nodes_index,nodes)
        self.normal = self.get_normal(nodes)
        self.surface = self.length(nodes) * self.get_normal(nodes)
        self.cells = np.array(cells_index,dtype=int)
        self.owner = cells_index[0]
        try :
            self.neighbour = cells_index[1]
        except :
            self.neighbour = None
        # TODO : définir si la face est une frontiere, une paroi ou juste une face de cellule
        self.isWall = False
        self.isBoundary = False


    def __repr__(self):
        return f"Face {self.indice_global} with nodes {self.nodes_index} and normal {self.normal}"
    
    def get_normal(self,nodes:np.array)->np.array:
        """Calcul de la normale à la face"""
        face_nodes = nodes[self.nodes_index]
        face_vector = (face_nodes[1] - face_nodes[0])
        # Compute the normal vector by rotating the face vector 90 degrees counter-clockwise
        normal = np.array([-face_vector[1], face_vector[0]],dtype=float)
        # normalize the normal vector
        normal /= np.linalg.norm(normal)
        return normal

    def set_owner(self,nodes:np.array,cells)->None:
        """Set le propriétaire de la face"""
        self.owner = None
        try : 
            self.neighbour = self.cells[0]
        except :
            raise MeshError("La face n'a pas de cellule contiguë")
        normale = self.get_normal(nodes)
        if len(self.cells) == 1:
            self.owner = self.cells[0]
            self.neighbour = None
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
    
    def stream(self,mesh)-> tuple: 
        """Renvoie les cellules de double amont à double aval"""
        # Dans le cas d'une frontière, on renvoie seulement la cellule frontière
        # On obtiendra un gradient nul
        if self.owner is None:
            D = mesh.cells[self.neighbour]
            return D,D,D,D,D
        elif self.neighbour is None:
            C = mesh.cells[self.owner]
            return C,C,C,C,C
        
        C = mesh.cells[self.owner]
        D = mesh.cells[self.neighbour]
        vec_CD = C.centroid - D.centroid
        xy_DD = D.centroid + vec_CD
        DD = mesh.find_cell(xy_DD[0], xy_DD[1])
        if DD is None:
            DD = D  # Si la cellule voisine n'existe pas, on prolonge la cellule D
        xy_U = C.centroid - vec_CD
        U = mesh.find_cell(xy_U[0], xy_U[1])
        if U is None:
            U = C
            UU = C 
        else :
            xy_UU = U.centroid - vec_CD
            UU = mesh.find_cell(xy_UU[0], xy_UU[1])
            if UU is None:
                UU = U
        return UU,U,C,D,DD
    
    def getCD(self,cells)-> tuple:
        """Returns the cenroids of the owner and neighbour cells."""
        if self.owner is None :
            C = self.centroid
            D = cells[self.neighbour].centroid
        elif self.neighbour is None:
            D = self.centroid
            C = cells[self.owner].centroid
        else :
            C = cells[self.owner].centroid
            D = cells[self.neighbour].centroid
        return C,D
    
    def Ef_Tf(self,cells:np.array) -> tuple:
        """Décompose le vecteur surface dans le repère formé par la droite reliant le centre des cellules et la tangente à la face"""
        C,D = self.getCD(cells)
        e = np.linalg.norm(C - D) # Vecteur unitaire entre les centres des cellules
        if e == 0:
            raise MeshError(f"Les centres des cellules {self.owner} et {self.neighbour} sont identiques, impossible de calculer la direction de la face")
        Ef = np.dot(e,self.surface) * e  # Projection de la surface sur la direction entre les centres des cellules
        Tf = self.surface - Ef
        return  Ef, Tf
    
    def distance_des_centres(self,cells:np.array)->float:
        """Calcul de la distance entre les centres des cellules"""
        C,D = self.getCD(cells)
        return np.linalg.norm(C - D)
    
    def normal_distance(self,obj:GeomObject) -> float:
        """Calcul de la distance normale entre la face et un objet géométrique"""
        c = obj.centroid
        vec_d = c - self.centroid
        return np.abs(np.dot(vec_d, self.normal))  # Distance normale à la face car ||n||=1
    
    def gDiff_f(self, cells:GeomObject)->float :
        """Calcul la valeur E_f / d_CF, appelée gDiff_f dans le bouquin (voir page 246)\n
        Input: array of Cell objects\n
        Output: float"""
        return np.linalg.norm(self.Ef_Tf(cells)[0]) / self.distance_des_centres(cells)
    
    def next_object(self,dir : np.ndarray, mesh) -> int:
        """Renvoie l'indice de la face voisine dans la direction donnée"""
        face_match = -1
        angle_match = -1
        list_of_faces = mesh.cells[self.owner].faces.tolist() + mesh.cells[self.neighbour].faces.tolist()
        list_of_faces.remove(self.indice_global)  # On enlève la face actuelle pour ne pas la considérer
        for face_index in list_of_faces:
            face = mesh.face[face_index]
            c = face.centroid
            if np.dot(c - self.centroid, dir) > angle_match:
                angle_match = np.dot(c - self.centroid, dir)
                face_match = face_index
        return mesh.cells[face_match].indice_global if face_match != -1 else None
    

        

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


    
    def next_object(self,dir : np.ndarray, mesh) -> int:
        """Renvoie l'indice de la cellule voisine dans la direction donnée"""
        face_match = -1
        angle_match = -1
        for face_index in self.faces:
            face = mesh.face[face_index]
            c = face.centroid
            if np.dot(c - self.centroid, dir) > angle_match:
                angle_match = np.dot(c - self.centroid, dir)
                face_match = face_index
        return mesh.cells[face_match].indice_global if face_match != -1 else None


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
        self.mat = self.matrice_voisins()
        self.setBonudaries()
        self.setNormalDistanceToWall()

    def copy(self):
        """Renvoie une copie du maillage"""
        new_mesh = Mesh(nodes=self.nodes.copy(), faces=self.faces.copy(), cells=self.cells.copy())
        return new_mesh

    def read_mesh(self, filename:str,faces_per_cell = 4)-> tuple[np.array, np.array, np.array] :
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
                # Reading the coordinates for each point
                point = [float(x) for x in line[1:].split()]
                nodes[n_count,0] = point[0]
                nodes[n_count,1] = point[1]
                n_count += 1
            elif line[0]=='f':
                # Reading the indices of the nodes for each face
                words = line.split()
                faces_ref[f_count,0] = int(words[1])
                faces_ref[f_count,1] = int(words[2])
                f_count += 1
            elif line[0]=='c':
                words = line.split()
                c_count = int(words[0][1:]) # The number of the cell is explicitly written
                # Reading the indices of the faces for each cell
                for j in range(len(words)-1):
                    face_index = int(words[j+1])
                    cells_ref[c_count,j] = face_index
                    # Adding the cell index to the corresponding face, at the first empty position
                    try:
                       cells_of_faces[face_index][np.where(cells_of_faces[face_index]==-1)[0][0]]=c_count
                    except IndexError:
                        pass                    

                c_count += 1

        # On enlève les lignes vides
        faces_ref = faces_ref[~np.all(faces_ref == 0, axis=1)]
        faces = np.zeros((len(faces_ref)), dtype=Face)
        for i in range(len(faces_ref)):
            cells_index = [idx for idx in cells_of_faces[i] if idx != -1]
            faces[i] = Face(i, faces_ref[i],cells_index ,nodes)

        cells_ref = cells_ref[~np.all(cells_ref == 0, axis=1)]
        cells = np.zeros((len(cells_ref)), dtype=Cell)
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
            cells[i] = Cell(i, cells_ref[i], nodes_of_cell,voisins,nodes)
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
    
    def setBonudaries(self):
        """Sets faces as walls or boundaries """
        xmin, xmax, ymin, ymax = self.span()
        for i,f in enumerate(self.faces):
            if f.owner is None or f.neighbour is None:
                x1,y1 = self.nodes[f.nodes_index[0]]
                if (np.abs(x1 - xmin) <1e-5 ) or (np.abs(y1 - ymin)<1e-5 ) or (np.abs(x1 - xmax) <1e-5 ) or (np.abs(y1 - ymax)<1e-5 )  :
                    self.faces[i].isBoundary = True
                else:
                    self.faces[i].isWall = True
    
    def plot_mesh(self, ax=None,plotnodes=False) -> None:
        """Enhanced visualization of the mesh."""
        offset = 0.02
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))  # Larger figure size for better visibility

        # Plot cells and faces
        for i in tqdm(range(len(self.faces)),desc="Plotting..."):
            f = self.faces[i]
            nodes = self.nodes[f.nodes_index]
            c = "orange" if f.isWall else "blue" if f.isBoundary else "gray"
            ax.plot(nodes[:, 0], nodes[:, 1], color=c, linewidth=1)  # Cell boundaries
            # centre = f.centroid
            # ax.text(centre[0]-offset, centre[1]-offset, f"{f.indice_global}", color="k", fontsize=8, ha="center", va="center")  # face labels

        # Plot nodes
        if plotnodes:
            for i, node in enumerate(self.nodes):
                ax.plot(node[0], node[1], 'ro', markersize=4)  # Nodes as red dots

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
    
    def find_cell_index(self,x:float,y:float)->int:
        for i in range(len(self.cells)):
            cell = self.cells[i]
            if cell.contains(x,y,self.nodes):
                return i
    
    def find_cell(self,x:float,y:float)->Cell:
        """Renvoie la cellule contenant le point (x,y)"""
        i = self.find_cell_index(x,y)
        if i is not None:
            return self.cells[i]
        else:
            return None

    def matrice_voisins(self)->np.array:
        """Calcul de la matrice des voisins"""
        n = len(self.cells)
        mat = np.zeros((n,n), dtype=int)
        for i in range(n):
            cell = self.cells[i]
            for j in cell.voisins:
                mat[i,j] = 1
        return mat
    
    def getWalls(self)->list:
        """Renvoie la liste des indices des faces de paroi"""
        walls = []
        for face in self.faces:
            if face.isWall:
                walls.append(face.indice_global)
        return walls
    
    def setNormalDistanceToWall(self):
        # TODO 
        walls = self.getWalls()
        if len(walls) == 0:
            for c in self.cells:
                c.distToWall = np.inf
            for f in self.faces:
                f.distToWall = np.inf
            return
        for f_i in walls:
            face = self.faces[f_i]
            face.distToWall = 0
        # Iterating on the cells to find the closest wall face
        for i in range(len(self.cells)):
            self.cells[i].distToWall = np.inf
            for fwi in walls :
                d = self.cells[i].distance_from(self.faces[fwi])
                if d < self.cells[i].distToWall:
                    self.cells[i].distToWall = d
        for fi in range(len(self.faces)):
            if self.faces[fi].owner is not None and self.faces[fi].neighbour is not None:
                self.faces[fi].distToWall = (self.cells[self.faces[fi].owner].distToWall + self.cells[self.faces[fi].neighbour].distToWall)/2
            elif fi not in walls:
                owner_i = self.faces[fi].owner
                neighbour_i = self.faces[fi].neighbour
                if self.faces[fi].owner is not None:
                    self.faces[fi].distToWall = self.cells[self.faces[fi].owner].distToWall + self.faces[fi].distance_from(self.cells[owner_i])
                elif self.faces[fi].neighbour is not None:
                    self.faces[fi].distToWall = self.cells[self.faces[fi].neighbour].distToWall + self.faces[fi].distance_from(self.cells[neighbour_i])
                

    def plot_distance_to_wall_heatmap(self, ax=None):
        """Affiche une heatmap de la distance à la paroi pour chaque cellule."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Récupère les coordonnées des centres et la distance à la paroi
        x_c = [cell.centroid[0] for cell in self.cells]
        y_c = [cell.centroid[1] for cell in self.cells]
        d_c = [getattr(cell, 'distToWall') for cell in tqdm(self.cells)]
        x_f = [face.centroid[0] for face in self.faces]
        y_f = [face.centroid[1] for face in self.faces]
        d_f = [getattr(face, 'distToWall') for face in tqdm(self.faces)]

        x= x_c + x_f
        y = y_c + y_f 
        d = d_c + d_f

        sc = ax.scatter(x, y, c=d, cmap='viridis', s=100)
        plt.colorbar(sc, ax=ax, label="Distance à la paroi")
        ax.set_title("Heatmap distance à la paroi")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        return ax
        


### Fonctions ext ###

def face_commune(c1 : Cell, c2 : Cell, mesh:Mesh)->Face:
    """Renvoie la face commune entre deux cellules"""
    faces1 = c1.faces
    faces2 = c2.faces
    for f1 in faces1:
        for f2 in faces2:
            if f1 == f2:
                return mesh.faces[f1]
    return None

def all_neighbouring_faces(cells_index : list, mesh:Mesh)->list:
    """Renvoie la liste des indices de toutes les faces voisines des cellules dont les indices sont listé.\n
    Input:\n
    - cells_index : liste d'indices de cellules\n
    - mesh : objet Mesh\n
    Output:\n
    - liste d'indices de faces"""
    neighbours = []
    for i in cells_index:
        cell = mesh.cells[i]
        for f_i in cell.faces:
            if f_i not in neighbours:
                neighbours.append(f_i)
    return neighbours

def all_neighbouring_cells(faces_index : list, mesh:Mesh)->list:
    """Renvoie la liste des indices de toutes les faces voisines des cellules dont les indices sont listé.\n
    Input:\n
    - faces_index : liste d'indices de faces\n
    - mesh : objet Mesh\n
    Output:\n
    - liste d'indices de cellules"""
    neighbours = []
    for i in faces_index:
        face = mesh.faces[i]
        for c_i in face.cells:
            if c_i not in neighbours:
                neighbours.append(c_i)
    return neighbours
        

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
    mesh.plot_distance_to_wall_heatmap()
    plt.show()