from tqdm import tqdm  # Import tqdm for progress bars
import numpy as np

# Parameters
circle_radius = 2.0
x_span = 40.0  # X-axis span
y_span = 20
num_circle_points = 100  # Number of points on the circle boundary
default_step = 0.5  # Default step size for mesh generation

XC = x_span / 2  
YC = y_span / 2  

def belong_to_circle(x, y):
    """Check if the point (x, y) belongs to the circle."""
    return (x - XC) ** 2 + (y - YC) ** 2 <= circle_radius ** 2 + 1e-6

def centre(nodes):  
    """Calcul du centre de gravité d'un ensemble de points"""
    return np.mean(nodes, axis=0)

def sort_nodes(nodes:np.array)->None:
    """Tri des noeuds de la cellule dans le sens trigonométrique"""
    c_node = centre(nodes)
    angles = np.arctan2(nodes[:, 1] - c_node[1], nodes[:, 0] - c_node[0])
    sorted_indices = np.argsort(angles)
    return nodes[sorted_indices]

def is_square(nodes:np.array):
    """Vérifie si les noeuds forment un carré"""
    if len(nodes) != 4:
        return False,0
    # Check if the nodes are in a square shape
    dists = np.linalg.norm(nodes[:, None] - nodes, axis=2)
    return np.all(np.isclose(dists, dists[0, 1], atol=1e-6)), dists[0, 1]

def parallel(f1, f2):
    """Check if two faces are parallel"""
    # Check if the faces are parallel
    v1 = f1[1] - f1[0]
    v2 = f2[1] - f2[0]
    parallel = (v1[0]*v2[1] - v1[1]*v2[0]<1e-6)
    return parallel 

def share_node(fi1, fi2):
    """Check if two faces share a node"""
    n1, n2 = fi1
    n3, n4 = fi2
    return (n1 == n3) or (n1 == n4) or (n2 == n3) or (n2 == n4)


def find_faces(nodes_index, faces_ij):
    """Find the faces that are formed by the given nodes."""
    face_indices = []
    for i in range(len(faces_ij)):
        face = faces_ij[i]
        
        if face[0] in nodes_index and face[1] in nodes_index:
            face_indices.append(i)

    return np.array(face_indices)

def generate_nodes(X,Y):
    nodes = []
    print("Generating nodes...")
    for x in tqdm(X, desc="Nodes (X-axis)"):
        for y in Y:
            if not belong_to_circle(x, y):
                nodes.append((x, y))
    return np.array(nodes)

def generate_faces(nodes):
    faces_ij = []
    faces_xy = []
    print("Generating faces...")
    for i in tqdm(range(len(nodes)), desc="Faces (Outer Loop)"):
        for j in range(i + 1, len(nodes)):
            x, y = nodes[j]
            d = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))
            if d <= default_step * 1.001 :
                faces_ij.append(np.array((i, j)))
                faces_xy.append(np.array((nodes[i], nodes[j])))
    return np.array(faces_ij),np.array(faces_xy)

def generate_cells(nodes,faces_ij,faces_xy):
    cells = []
    print("Generating cells...")
    face_is_used = np.zeros(len(faces_ij), dtype=int)
    for i in tqdm(range(len(faces_ij)),desc="Cells"):
        if face_is_used[i] >= 2:
            continue
        f1 = faces_ij[i]
        n1, n2 = f1
        c1 = centre(nodes[f1])
        # create a subset of surrounding faces
        index_face_of_interest = np.arange(len(faces_ij))
        is_vertical = np.abs(c1[0] - nodes[f1[0]][0]) < 1e-6
        if is_vertical:     
            subset_index0 = (faces_xy[:,0,0]==faces_xy[:,1,0])
            subset_index1 = (np.abs(faces_xy[:,0,0]-c1[0])<default_step*1.001)
            subset_index = np.logical_and(subset_index0,subset_index1)
            subset_xy = faces_xy[subset_index]
            subset_ij = faces_ij[subset_index]
            index_face_of_interest = index_face_of_interest[subset_index]

            subset_index0 = np.abs(subset_xy[:,0,1]-c1[1])<default_step*1.001
            subset_index1 = np.abs(subset_xy[:,1,1]-c1[1])<default_step*1.001
            subset_index  = np.logical_and(subset_index0,subset_index1)
            subset_xy = subset_xy[subset_index]
            subset_ij = subset_ij[subset_index]
            index_face_of_interest = index_face_of_interest[subset_index]

        else :
            subset_index0 = (faces_xy[:,0,1]==faces_xy[:,1,1])
            subset_index1 = (np.abs(faces_xy[:,0,1]-c1[1])<default_step*1.001)
            subset_index = np.logical_and(subset_index0,subset_index1)
            subset_xy = faces_xy[subset_index]
            subset_ij = faces_ij[subset_index]
            index_face_of_interest = index_face_of_interest[subset_index]

            subset_index0 = np.abs(subset_xy[:,0,0]-c1[0])<default_step*1.001
            subset_index1 = np.abs(subset_xy[:,1,0]-c1[0])<default_step*1.001
            subset_index  = np.logical_and(subset_index0,subset_index1)
            subset_xy = subset_xy[subset_index]
            subset_ij = subset_ij[subset_index]
            index_face_of_interest = index_face_of_interest[subset_index]

        if len(subset_ij) == 0:
            raise ValueError("No surrounding faces found for the given face.")
        if len(subset_ij) > 3 :
            raise ValueError("More than 3 surrounding faces found for the given face.")
        for j in range(len(subset_ij)):
            # Check if face j is the same as face i
            if (subset_ij[j] == faces_ij[i]).all() :
                continue
            if face_is_used[index_face_of_interest[j]] >= 2:
                continue
            n3, n4 = subset_ij[j]
            if is_square(nodes[np.array([n1, n2, n3, n4])]):
                candidate_face_index = find_faces([n1,n2,n3,n4], faces_ij)
                if len(candidate_face_index)==4:
                    face_is_used[candidate_face_index] += 1
                    cells.append(candidate_face_index)

    return cells,face_is_used

# Open file to write the mesh
with open("D:/OneDrive/Documents/11-Codes/overflow/02_RANS/circle_mesh.dat", "w") as file:

    X = np.linspace(0, x_span, int(x_span / default_step) + 1)
    Y = np.linspace(0, y_span, int(y_span / default_step) + 1)
    nodes = generate_nodes(X,Y)
    
    print("Writing nodes to file...")
    for i, (x, y) in tqdm(enumerate(nodes), desc="Writing Nodes", total=len(nodes)):
        file.write(f"n {x:.2f} {y:.2f}\n")

    faces_ij, faces_xy = generate_faces(nodes)

    cells , face_is_used = generate_cells(nodes,faces_ij,faces_xy)    
    # Remove duplicate cells
    print("Removing duplicate cells...")
    unique_cells = []
    seen = set()
    for cell in cells:
        sorted_cell = tuple(sorted(cell))
        if sorted_cell not in seen:
            seen.add(sorted_cell)
            unique_cells.append(cell)
    cells = unique_cells
    assert np.all(face_is_used >= 1), f"Only {np.count_nonzero(face_is_used)} faces out of {len(face_is_used)} are used in any cell."

    # Write faces to file
    print("Writing faces to file...")
    for i, (i, j) in tqdm(enumerate(faces_ij), desc="Writing Faces", total=len(faces_ij)):
        file.write(f"f {i} {j}\n")

    # Write cells to file
    print("Writing cells to file...")
    for i, cell in tqdm(enumerate(cells), desc="Writing Cells", total=len(cells)):
        file.write(f"c{i} {' '.join(map(str, cell))}\n")

    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of faces: {len(faces_ij)}")
    print(f"Number of cells: {len(cells)}")