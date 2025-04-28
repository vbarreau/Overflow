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

class Parametres():
    """Classe contenant les paramètres physiques d'un point"""
    def __init__(self, T:float = 300, p:float = 1e5,V:np.array=np.array([0,0])):
        self.T = T
        self.p = p
        self.v = V


class Mesh():

    def __init__(self, filename):
        """Initialize the Mesh object and read nodes, faces, and cells."""
        self.nodes = self.mesh_read_nodes(filename)
        self.faces = self.mesh_read_faces(filename, self.nodes)
        self.cells = self.mesh_read_cells(filename, self.faces, self.nodes)
        for face in self.faces:
            face.set_owner(self.nodes, self.cells)
            face.set_mean_param(self.cells)

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
    
    def mesh_read_faces(self, filename:str,nodes)->list:
        """Lecture du maillage à partir d'un fichier"""
        lines = open(filename, 'r').readlines()
        faces = []
        for i in range(len(lines)):
            line = lines[i]
            if line[0]=='f':
                index = len(faces)
                if index == 3 :
                    a=1
                # On récupère les indices des noeuds de la face
                nodes_index = [int(x) for x in line[1:].split()]
                cells = []
                for j in range(i,len(lines)):
                    l = lines[j]
                    if l[0]=='c' and f" {index}" in l:
                        # On récupère les indices des cellules de la face
                        cells.append(int(l[1]))
                # On crée la face
                faces.append(self.Face(index, nodes_index, cells,nodes))
        return faces

    def mesh_read_cells(self,filename:str,faces,nodes)->list:
        """Lecture du maillage à partir d'un fichier"""
        lines = open(filename, 'r').readlines()
        cells = []

        for i in range(len(lines)):
            line = lines[i]
            if line[0]=='c':
                index = len(cells)
                # On récupère les indices des noeuds de la cellule
                faces_index = [int(x) for x in line[2:].split()]
                nodes_index = []
                voisins = []
                for fi in faces_index:
                    # On récupère les indices des noeuds de la face
                    f = faces[fi]
                    nodes_index += f.nodes_index.tolist()
                    # On récupère les indices des cellules voisines de la face
                    for j in range(len(f.cells)):
                        if f.cells[j] != index:
                            voisins.append(f.cells[j])
                # On enlève les doublons
                nodes_index = list(set(nodes_index))
                voisins = list(set(voisins))
                cells.append(self.Cell(index, faces_index, nodes_index, voisins,nodes))

        return cells

    class Face():
        def __init__(self, i:int, nodes_index:list, cells_index,nodes):
            self.indice_global = i
            self.nodes_index = np.array(nodes_index)
            self.centroid = centre(nodes[nodes_index])
            self.surface = self.length(nodes) * self.get_normal(nodes)
            self.cells = np.array(cells_index)


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
            normal = np.array([-face_vector[1], face_vector[0]])
            # On normalise le vecteur
            normal /= np.linalg.norm(normal)
            return normal
        
        def set_owner(self,nodes:np.array,cells)->None:
            """Set le propriétaire de la face"""
            self.owner = -1
            self.neighbour = self.cells[0]
            vec0 = self.centroid + self.get_normal(nodes)
            vec1 = self.centroid - self.get_normal(nodes)
            for i in range(len(self.cells)):
                c = cells[self.cells[i]]
                c_centre = c.centroid
                d1 = np.linalg.norm(vec0 - c_centre)
                d2 = np.linalg.norm(vec1 - c_centre)
                if d1 > d2:
                    self.owner = c.indice_global
                    if c.indice_global != self.cells[i]:
                        self.neighbour = c.indice_global
                    elif len(self.cells) > 1:
                        self.neighbour = self.cells[1-i]
                    break
        
        def set_mean_param(self,cells) : 
            Vo = cells[self.owner].volume
            Vn = cells[self.neighbour].volume
            gn = Vo/(Vo + Vn)
            go = 1-gn
            T = gn*cells[self.neighbour].param.T + go*cells[self.owner].param.T
            v = gn*cells[self.neighbour].param.v + go*cells[self.owner].param.v
            p = gn*cells[self.neighbour].param.p + go*cells[self.owner].param.p
            self.param = Parametres(T,p,v)






    class Cell():
        def __init__(self,i:int,faces_index:list,nodes_index:list,voisins:list,nodes:np.array):
            self.indice_global = i
            self.faces = np.array(faces_index)
            self.nodes = np.array(nodes_index)
            self.voisins = np.array(voisins)
            self.volume = self.get_volume(nodes)
            self.centroid = np.mean(nodes[self.nodes], axis=0)
            self.sort_nodes(nodes)
            self.param = Parametres()


        def __repr__(self):
            return f"Cell {self.indice_global} : \nfaces : {self.faces}\nNoeuds : {self.nodes}\nVoisins : {self.voisins}\nParametres: {self.T} K, {self.v} m/s, {self.p} Pa\n"
        
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

        # def sort_faces(self):
        #     """Tri des faces de la cellule dans le sens trigonométrique"""
        #     new_faces = []
        #     for i in range(len(self.nodes)-1):
        #         i_f = np.where( nodes[i] in face.nodes)


        def get_volume(self,nodes:np.array)->float:
            nodes = nodes[self.nodes]
            V = surface_triangle(nodes[0], nodes[1], nodes[2])
            for i in range(3, len(nodes)):
                V += surface_triangle(nodes[0], nodes[i-1], nodes[i])
            return V
        
        def grad(self, var,G_face:list)->np.array:
            """Calcul du gradient d'une variable dans la cellule"""
            # On calcule le gradient de la variable dans la cellule
            # On utilise la méthode des différences finies
            grad = np.zeros(2)
            for i in range(len(self.faces)):
                f = G_face[self.faces[i]]
                sign = f.owner == self.indice_global
                grad += sign * f.surface * f.get_var(var)
            grad /= self.volume
                
            return grad / len(self.nodes)

    def compute_gradient(self, var:str)->np.array:
        """Calcul du gradient d'une variable dans la cellule"""
        # On calcule le gradient de la variable dans la cellule
        # On utilise la méthode des différences finies
        grad = np.zeros(len(self.cells),2)
        for i in range(len(self.faces)):
            f = self.faces[i]
            flux_f = f.get_flux(var)*f.surface
            if f.owner != -1:
                grad[f.owner] += flux_f
                grad[f.neighbour] -= flux_f
            else:
                grad[f.neighbour] -= flux_f
        grad /= self.volume
                
        return grad / len(self.nodes)
    
    def plot_mesh(self)->None:
        """Affichage du maillage"""
        plt.figure()
        for i in range(len(self.faces)):
            f = self.faces[i]
            nodes = self.nodes[f.nodes_index]
            plt.plot(nodes[:,0], nodes[:,1], 'k-')
            plt.fill(nodes[:,0], nodes[:,1], alpha=0.2)
        plt.show()
     

if __name__ ==  "__main__" :
    mesh = Mesh("D:/OneDrive/Documents/11-Codes/overflow/02_RANS/mesh.dat")
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
    cell.T = 300
    cell.v = np.array([1,0])
    cell.p = 101325
    print(cell.__repr__())  # Affiche la cellule

    mesh.plot_mesh()  # Affiche le maillage