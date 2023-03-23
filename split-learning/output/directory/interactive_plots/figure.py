# from logging import root
# import matplotlib
# from matplotlib.backend_bases import NavigationToolbar2
# import matplotlib.pyplot as plt
# import pickle
# matplotlib.use('TkAgg')
# # matplotlib.use('Qt5Agg')
# from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
# import tkinter as tk


# # with open('batch_size_client_num_in_total.fig.pkl', 'rb') as f:
# #     fig = pickle.load(f)
# # with open('batch_size_cut_layer.fig.pkl', 'rb') as f:
# #     fig = pickle.load(f)
# # with open('client_num_in_total_cut_layer.fig.pkl', 'rb') as f:
# #     fig = pickle.load(f)

# # # plt.ion()  # Turn on interactive mode
# # plt.show(block=True)  # Show the figure and block the script until the window is closed


# # Define a custom NavigationToolbar2Tk class
# class CustomToolbar(NavigationToolbar2Tk):
#     def __init__(self, canvas, parent):
#         self.toolitems = (
#             ('Home', 'Reset original view', 'home', 'home'),
#             ('Back', 'Back to previous view', 'back', 'back'),
#             ('Forward', 'Forward to next view', 'forward', 'forward'),
#             ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
#             ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
#             ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
#         )
#         self._buttons = {}
#         NavigationToolbar2.__init__(self, canvas)
#         self.parent = parent


# # Load the .fig.pkl file
# file_path = 'batch_size_client_num_in_total.fig.pkl'
# with open(file_path, 'rb') as f:
#     print(matplotlib.get_backend())
#     fig = pickle.load(f)

# # Create a Tkinter window and add the figure to it
# root = tk.Tk()
# canvas = plt.figure(figsize=(6, 4)).canvas
# canvas.draw()
# canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# # Create a NavigationToolbar2Tk object
# # toolbar = NavigationToolbar2Tk(canvas, root)
# toolbar = CustomToolbar(canvas, root)
# plt.rcParams['toolbar'] = 'toolmanager'
# toolbar.update()

# # Display the interactive plot
# plt.show()

from logging import root
import matplotlib
from matplotlib.backend_bases import NavigationToolbar2
import matplotlib.pyplot as plt
import pickle
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import tkinter as tk

# Define a custom NavigationToolbar2Tk class
class CustomToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, parent):
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Back', 'Back to previous view', 'back', 'back'),
            ('Forward', 'Forward to next view', 'forward', 'forward'),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
            ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        )
        self._buttons = {}
        NavigationToolbar2.__init__(self, canvas)
        self.parent = parent


# Load the .fig.pkl file
file_path1 = 'batch_size_client_num_in_total.fig.pkl'
with open(file_path1, 'rb') as f:
    print(matplotlib.get_backend())
    fig = pickle.load(f)

file_path2 = 'batch_size_cut_layer.fig.pkl'
with open(file_path2, 'rb') as f:
    print(matplotlib.get_backend())
    fig = pickle.load(f)

file_path3 = 'client_num_in_total_cut_layer.fig.pkl'
with open(file_path3, 'rb') as f:
    print(matplotlib.get_backend())
    fig = pickle.load(f)


# Create a Tkinter window and add the figure to it
root = tk.Tk()
canvas = plt.figure(figsize=(6, 4)).canvas
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Create a NavigationToolbar2Tk object
# toolbar = NavigationToolbar2Tk(canvas, root)
toolbar = CustomToolbar(canvas, root)
plt.rcParams['toolbar'] = 'toolmanager'
toolbar.update()

# Display the interactive plot
plt.show()