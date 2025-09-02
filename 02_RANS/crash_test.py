import sys
import matplotlib.pyplot as plt
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from geom import *
print("running crash_test.py")

if __name__=="__main__":

    circle_mesh = Mesh(filename = r'D:\OneDrive\Documents\11-Codes\overflow\02_RANS\circle_mesh.dat')
    circle_mesh.plot_distance_to_wall_heatmap()
    plt.show()
