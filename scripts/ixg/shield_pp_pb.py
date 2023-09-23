import os
import random
import mujoco as mp
from mujoco import MjData, MjModel
import mujoco_viewer
from time import sleep
import numpy as np

import pickle

nq = 6
dt = 1e-2
new_traj_dt = 1
nviz = 5

model_dir = '/home/gaussian/cmu_ri_phd/phd_research/parallel_search/third_party/mujoco-2.3.2/model/abb/irb_1600/'
mjcf = 'irb1600_6_12_realshield_pp_anim.xml'

# just using arm model for calculating ee traj
arm_model = MjModel.from_xml_path(os.path.join(model_dir, mjcf))
arm_data = MjData(arm_model)
viewer = mujoco_viewer.MujocoViewer(arm_model, arm_data)

# Directory containing your files
directory = '/home/gaussian/cmu_ri_phd/phd_research/parallel_search/logs/insat_logs_obs/paths_library/'

# Prefix to filter files
prefix = 'path_'

# List all files in the directory that start with the specified prefix
matching_files = [file for file in os.listdir(directory) if file.startswith(prefix)]


def loadPathFromLib(idx=-1):
    # Check if there are any matching files
    if not matching_files:
        print(f"No files found in {directory} starting with '{prefix}'.")
    else:
        # Select a random file from the matching files
        if idx == -1:
            random_file = random.sample(matching_files, k=nviz)
        else:
            random_file = 'path_' + str(idx)

        pp_traj = []
        for rf in random_file:
            # Load the file as a NumPy matrix, excluding the first line
            file_content = np.loadtxt(os.path.join(directory, rf), skiprows=1)

            # Now you have the content of the random file in the 'file_content' variable
            print(f"Random file selected: {rf}")
            pp_traj.append(file_content)

        return pp_traj

def padMultipleTrajectories(pp_traj):
    # Determine the maximum dimensions
    N = len(pp_traj)
    max_rows = max(matrix.shape[0] for matrix in pp_traj)
    max_cols = max(matrix.shape[1] for matrix in pp_traj)

    max_rows *= 2

    # Initialize the banded matrix with zeros
    banded_matrix = np.zeros((N * max_rows, N * max_cols))

    # Copy the elements from each input matrix to the banded matrix
    row_offset = 0
    offset_add = int(new_traj_dt/dt)
    for i, matrix in enumerate(pp_traj):
        rows, cols = matrix.shape
        col_offset = i * max_cols
        banded_matrix[row_offset:row_offset + rows, col_offset:col_offset + cols] = matrix
        banded_matrix[row_offset + rows:, col_offset:col_offset + cols] = matrix[-1, :]
        row_offset += offset_add

    return banded_matrix

pp_traj = loadPathFromLib()
banded_traj = padMultipleTrajectories(pp_traj)


for i in range(np.shape(banded_traj)[0]):
    viz_traj = np.hstack((np.zeros(nq), banded_traj[i, :]))

    if viewer.is_alive:
        arm_data.qpos[:] = viz_traj[:]
        mp.mj_step(arm_model, arm_data)
        viewer.render()
        sleep(dt)
    else:
        break

# close
# viewer.close()

