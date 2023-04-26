import matplotlib.pyplot as plt

def plot(T, meas, traj, kfiltered, ind_to_vis):
    plt.figure()
    fig, ax = plt.subplots()
    linemeas, = ax.plot(range(T), meas[ind_to_vis, :, 0], label="Measured")
    linetraj, = ax.plot(range(T), traj[ind_to_vis, :, 0], label="Trajectory")
    linefltr, = ax.plot(range(T), kfiltered[ind_to_vis, :, 0], label="Filtered", color='g')
    plt.legend()
    plt.show()
