from tqdm import tqdm  # Import tqdm for progress bars
import numpy as np

# Parameters
circle_radius = 2.0
square_size = 20.0  # Square domain size (side length)
num_circle_points = 100  # Number of points on the circle boundary
default_step = 1  # Default step size for mesh generation

def belong_to_circle(x, y):
    """Check if the point (x, y) belongs to the circle."""
    return (x - square_size / 2) ** 2 + (y - square_size / 2) ** 2 <= circle_radius ** 2 + 1e-6

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


def find_faces(nodes_index, faces,faces_index):
    """Find the faces that are formed by the given nodes."""
    face_indices = []
    for i in range(len(faces)):
        face = faces[i]
        
        if face[0] in nodes_index and face[1] in nodes_index:
            face_indices.append(faces_index[i])

    return face_indices

def generate_nodes(x,nodes):
    print("Generating nodes...")
    for x in tqdm(XY, desc="Nodes (X-axis)"):
        for y in XY:
            if not belong_to_circle(x, y):
                nodes.append((x, y))

def generate_faces(nodes,faces):
    print("Generating faces...")
    for i in tqdm(range(len(nodes)), desc="Faces (Outer Loop)"):
        for j in range(i + 1, len(nodes)):
            x, y = nodes[j]
            d = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))
            if d <= default_step * 1.001 :
                faces.append(np.array((i, j)))
    

def generate_cells(nodes,faces, cells):
    test = False
    print("Generating cells...")
    face_is_used = np.zeros(len(faces), dtype=int)
    for i in tqdm(range(len(faces)),desc="Cells"):
        if face_is_used[i] >= 2:
            continue
        f1 = faces[i]
        n1, n2 = f1
        c1 = centre(nodes[f1])
        # create a subset of surrounding faces
        index_face_of_interest = [i]
        for j in range(len(faces)):
            if face_is_used[j] >= 2:
                continue
            c2 = centre(nodes[faces[j]])
            if np.linalg.norm(c1 - c2) <= default_step * 1.001:
                index_face_of_interest.append(j)
        index_face_of_interest = list(set(index_face_of_interest))
        assert len(index_face_of_interest) >1, f"Only {len(index_face_of_interest)} faces found for cell {i}."
        for j in index_face_of_interest:
            if parallel(nodes[faces[i]],nodes[faces[j]]) and not share_node(faces[i],faces[j]):
                n1, n2 = faces[j]
                n3,n4 = faces[i]
                cell = find_faces([n1,n2,n3,n4],faces[index_face_of_interest],index_face_of_interest)
                face_is_used[np.array(cell)] += 1
                cells.append(cell)

    return face_is_used

# Open file to write the mesh
with open("D:/OneDrive/Documents/11-Codes/overflow/02_RANS/circle_mesh.dat", "w") as file:
    nodes = []
    faces = []
    cells = []

    XY = np.linspace(0, square_size, int(square_size / default_step) + 1)
    generate_nodes(XY,nodes)
    nodes = np.array(nodes)
    print("Writing nodes to file...")
    for i, (x, y) in tqdm(enumerate(nodes), desc="Writing Nodes", total=len(nodes)):
        file.write(f"n {x:.2f} {y:.2f}\n")

    generate_faces(nodes,faces)
    faces = np.array(faces)

    face_is_used = generate_cells(nodes,faces,cells)    
    assert np.all(face_is_used >= 1), f"Only {np.count_nonzero(face_is_used)} faces out of {len(face_is_used)} are used in any cell."

    # Write faces to file
    print("Writing faces to file...")
    for i, (i, j) in tqdm(enumerate(faces), desc="Writing Faces", total=len(faces)):
        file.write(f"f {i} {j}\n")

    # Write cells to file
    print("Writing cells to file...")
    for i, cell in tqdm(enumerate(cells), desc="Writing Cells", total=len(cells)):
        file.write(f"c{i} {' '.join(map(str, cell))}\n")

    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of faces: {len(faces)}")
    print(f"Number of cells: {len(cells)}")