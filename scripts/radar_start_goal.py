import numpy as np

num_start = 8
goal_per_start = 250

q = np.array([0.0, -0.3, 0.3, 0.0, 0.0, 0.0])

radar_start = np.tile(q, (num_start*goal_per_start, 1))

for i in range(num_start):
    for j in range(goal_per_start):
        radar_start[i*goal_per_start + j, 0] = (i*(2*np.pi))/num_start
        radar_start[i*goal_per_start + j, 0] -= np.pi # bringing to -pi to pi

radar_start_file = '../examples/manipulation/resources/shield/radar_starts.txt'
with open(radar_start_file, "w") as f:
    np.savetxt(f, radar_start, delimiter=' ')


