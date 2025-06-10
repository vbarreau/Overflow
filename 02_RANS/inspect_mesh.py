import sys
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from geom import Mesh


class MeshInspector:
    def __init__(self, mesh):
        self.mesh = mesh
        self.current_cell_index = 0

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Mesh Inspector")

        # Create a frame for the plot
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a frame for controls
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Add buttons and entry box
        self.prev_button = ttk.Button(self.control_frame, text="Previous", command=self.show_previous_cell)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.cell_entry = ttk.Entry(self.control_frame, width=10)
        self.cell_entry.pack(side=tk.LEFT, padx=5, pady=5)
        self.cell_entry.bind("<Return>", self.go_to_cell)

        self.next_button = ttk.Button(self.control_frame, text="Next", command=self.show_next_cell)
        self.next_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.quit_button = ttk.Button(self.control_frame, text="Quit", command=self.root.quit)
        self.quit_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Add a label to display cell information
        self.info_label = ttk.Label(self.control_frame, text="")
        self.info_label.pack(side=tk.LEFT, padx=10, pady=5)

        # Initialize the plot
        self.figure, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(0, 20)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Show the first cell
        self.plot_cell(self.current_cell_index)

    def plot_cell(self, cell_index):
        """Plot a single cell with its nodes and centroid."""
        self.ax.clear()
        cell = self.mesh.cells[cell_index]
        nodes = self.mesh.nodes[cell.nodes_index]

        self.ax.plot(nodes[:, 0], nodes[:, 1], 'bo-', label="Cell Nodes")
        self.ax.fill(nodes[:, 0], nodes[:, 1], alpha=0.3, label="Cell Area")
        self.ax.plot(cell.centroid[0], cell.centroid[1], 'rx', label="Centroid")
        self.ax.set_title(f"Cell {cell_index}")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        
        self.ax.legend()
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(0, 20)
        self.ax.grid(True)

        self.canvas.draw()

        # Update cell information
        self.info_label.config(
            text=f"Cell {cell_index} | Faces: {cell.faces} | Centroid: {cell.centroid} | Volume: {cell.volume:.2f}"
        )

    def show_next_cell(self):
        """Show the next cell."""
        self.current_cell_index = (self.current_cell_index + 1) % len(self.mesh.cells)
        self.plot_cell(self.current_cell_index)

    def show_previous_cell(self):
        """Show the previous cell."""
        self.current_cell_index = (self.current_cell_index - 1) % len(self.mesh.cells)
        self.plot_cell(self.current_cell_index)

    def go_to_cell(self, event):
        """Go to a specific cell."""
        try:
            cell_index = int(self.cell_entry.get())
            if 0 <= cell_index < len(self.mesh.cells):
                self.current_cell_index = cell_index
                self.plot_cell(self.current_cell_index)
            else:
                self.info_label.config(text="Invalid cell index!")
        except ValueError:
            self.info_label.config(text="Please enter a valid number!")

    def run(self):
        """Run the Tkinter main loop."""
        
        self.root.mainloop()


if __name__ == "__main__":
    # Load the mesh
    mesh_file = "D:/OneDrive/Documents/11-Codes/overflow/02_RANS/circle_mesh.dat"
    mesh = Mesh(filename=mesh_file)

    # Start the inspector
    inspector = MeshInspector(mesh)
    inspector.run()