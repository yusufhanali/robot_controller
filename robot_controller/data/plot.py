import numpy as np
from matplotlib import pyplot as plt

def plot_data(data, title, xlabel, ylabel):
    
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    
def plot_3d_data(data, title, xlabel, ylabel, zlabel):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[0], data[1], data[2])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()
    
def all_positive(data):
    for i in range(len(data)):
        if data[i] < 0:
            data[i] *= -1
    return data
    
error_x_nocomp = np.load("/home/kovan/USTA/src/robot_controller/robot_controller/data/breathe_gaze_error/no_compensation/error_x.npy")
error_y_nocomp = np.load("/home/kovan/USTA/src/robot_controller/robot_controller/data/breathe_gaze_error/no_compensation/error_y.npy")
breathe = np.load("/home/kovan/USTA/src/robot_controller/robot_controller/data/breathe_gaze_error/no_compensation/breathe.npy")

error_x_comp = np.load("/home/kovan/USTA/src/robot_controller/robot_controller/data/breathe_gaze_error/compensation/error_x.npy")
error_y_comp = np.load("/home/kovan/USTA/src/robot_controller/robot_controller/data/breathe_gaze_error/compensation/error_y.npy")

mean_error_x_nocomp = np.mean(all_positive(error_x_nocomp.copy()))
mean_error_y_nocomp = np.mean(all_positive(error_y_nocomp.copy()))
mean_error_x_comp =   np.mean(all_positive(error_x_comp.copy()))
mean_error_y_comp =   np.mean(all_positive(error_y_comp.copy()))

plt.plot(error_x_nocomp, label=f"Error X No Comp (Mean: {mean_error_x_nocomp*100} cm)")
plt.plot(error_y_nocomp, label=f"Error Y No Comp (Mean: {mean_error_y_nocomp*100} cm)")
plt.plot(error_x_comp, label=f"Error X Comp (Mean: {mean_error_x_comp*100} cm)")
plt.plot(error_y_comp, label=f"Error Y Comp (Mean: {mean_error_y_comp*100} cm)")
plt.plot(breathe, label="Breathe")
plt.title("Error X and Y")
plt.xlabel("Time")
plt.ylabel("Error")
plt.legend()
plt.show()