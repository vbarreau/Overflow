Ce modeste code de simulation à simplement pour objectif de simuler des écoulement 2D avec une méthode RANS et un modèle à deux équations, k-oméga. Il n'a aucune autre ambition que d'entretenir mes compétences en code et en mécanique des fluides.

Les seules bibliothèques utilisées sont Numpy et Matplotlib. Par conséquent, le projet contient des codes de création et lecture de maillage.
Un premier code permet de simuler une diffusion convection puis la simulation RANS est dans le dossier eponyme.

La simulation RANS est structurée en différentes classes :
- La classe Face contient les coordonnée des deux noeuds formant la face ;
- La classe Cell contient les références des faces formant la cellule ;
- Mesh est le maillage, soit un ensemble de noeuds, faces et cellules ;
- Param est une classe permettant d'associer un ensemble de paramètres physiques à une cellule ;
- Sim est la simulation : elle contient un Mesh et une liste de Param.
