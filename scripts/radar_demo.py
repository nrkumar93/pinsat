
import os
import mujoco as mp
from mujoco import MjData, MjModel
import mujoco_viewer
from time import sleep
import numpy as np
from numpy import genfromtxt

# planner_name = 'test'
# planner_name = 'insat'
# planner_name = 'pinsat'
# planner_name = 'rrt'
# planner_name = 'rrtconnect'
# planner_name = 'epase'
planner_name = 'gepase'

static_planner = True if not (planner_name=='insat' or planner_name=='pinsat' or planner_name=='test') else False

if static_planner:
  # dt = 1e-2
  # dt = 0.05
  dt = 2e-3
  # dt = 6e-3
else:
  # dt = 6e-3
  dt = 5e-3
  # dt = 0.5
  # dt = 1

model_dir = '../third_party/mujoco-2.3.2/model/abb/irb_1600'
mjcf = 'irb1600_6_12.xml'
mjcf_arm = 'irb1600_6_12_shield.xml'
traj_file = '../logs/demo/' + planner_name + '_abb_traj.txt'
starts_file = '../logs/demo/' + planner_name + '_abb_starts.txt'
goals_file = '../logs/demo/' + planner_name + '_abb_goals.txt'
traj = genfromtxt(traj_file, delimiter=' ' if static_planner else ',')
starts = genfromtxt(starts_file, delimiter=',')
goals = genfromtxt(goals_file, delimiter=',')

# just using arm model for calculating ee traj
arm_model = MjModel.from_xml_path(os.path.join(model_dir, mjcf_arm))
arm_data = MjData(arm_model)
viewer = mujoco_viewer.MujocoViewer(arm_model, arm_data)

qpos = np.zeros((6,))
qpos[1] = -0.3
qpos[2] = 0.3
qvel = np.zeros((6,))
qvel[0] = 2*np.pi

num_frames = 5000

i=0
while i <= num_frames:

  if viewer.is_alive:
    arm_data.qpos[:] = qpos[:]
    arm_data.qvel[:] = qvel[:]

    mp.mj_inverse(arm_model, arm_data)
    arm_data.qfrc_applied[:] = arm_data.qfrc_inverse[:]

    mp.mj_step(arm_model, arm_data)

    qpos[:] = arm_data.qpos[:]

    viewer.render()
    # sleep(dt)
  else:
      break
  i+=1

# close
viewer.close()

