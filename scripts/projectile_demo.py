
import os
import mujoco as mp
from mujoco import MjData, MjModel
import mujoco_viewer
from time import sleep
import numpy as np
from numpy import genfromtxt

# planner_name = 'test'
# planner_name = 'insat'
planner_name = 'pinsat'
# planner_name = 'rrt'
# planner_name = 'rrtconnect'
# planner_name = 'epase'

static_planner = True if not (planner_name=='insat' or planner_name=='pinsat' or planner_name=='test') else False

if static_planner:
  # dt = 1e-2
  # dt = 0.05
  # dt = 6e-3
  dt = 5e-4
else:
  # dt = 6e-3
  dt = 9e-4
  # dt = 0.5
  # dt = 1

model_dir = '/home/gaussian/cmu_ri_phd/phd_research/parallel_search/third_party/mujoco-2.3.2/model/abb/irb_1600'
mjcf_arm = 'irb1600_6_12_shield.xml'
mjcf_arm_demo = 'irb1600_6_12_shield_projectile.xml'
traj_file = '/home/gaussian/cmu_ri_phd/phd_research/parallel_search/logs/demo/' + planner_name + '_abb_traj.txt'
starts_file = '/home/gaussian/cmu_ri_phd/phd_research/parallel_search/logs/demo/' + planner_name + '_abb_starts.txt'
goals_file = '/home/gaussian/cmu_ri_phd/phd_research/parallel_search/logs/demo/' + planner_name + '_abb_goals.txt'
traj = genfromtxt(traj_file, delimiter=' ' if static_planner else ',')
starts = genfromtxt(starts_file, delimiter=',')
goals = genfromtxt(goals_file, delimiter=',')
# just using arm model for calculating ee traj
arm_model = MjModel.from_xml_path(os.path.join(model_dir, mjcf_arm))
arm_data = MjData(arm_model)
# for demo
arm_model_demo = MjModel.from_xml_path(os.path.join(model_dir, mjcf_arm_demo))
arm_data_demo = MjData(arm_model_demo)
viewer = mujoco_viewer.MujocoViewer(arm_model_demo, arm_data_demo)


def balltraj(end_pos, len, max_len, dx):
  xy_end = np.array([end_pos[0], end_pos[1], 0])
  u = end_pos/np.linalg.norm(xy_end)
  ball_traj = np.tile(end_pos, [max_len, 1])
  for i in range(len):
    ball_traj[i] = end_pos + (len-i)*u*dx
  quat_traj = np.tile([1,0,0,0], [max_len, 1])
  ball_traj = np.hstack((ball_traj, quat_traj))
  return ball_traj

def processLogToTrajectory(log, sts, gos):
  log_len = np.shape(log)[0]
  idx = np.zeros((log_len, 1))
  for st in range(np.shape(sts)[0]):
    st_mat = np.repeat([sts[st,:]], repeats=log_len, axis=0)
    diff = log - st_mat
    dnorm = np.linalg.norm(diff, axis=1)
    idx[st] = np.argmin(dnorm)

  traj = list()
  for i in range(np.shape(idx)[0]-1):
    traj.append(log[int(idx[i][0]):int(idx[i+1][0])-1,:])

  return traj

def getBallGoal(goals):
  goal_len = np.shape(goals)[0]
  ball_goals = np.zeros((goal_len,3))
  for go in range(goal_len):
    arm_data.qpos[:] = goals[go,:]
    mp.mj_step(arm_model, arm_data)
    ball_goals[go,:] = arm_data.xpos[-1]
  return ball_goals

ind_traj = processLogToTrajectory(traj, starts, goals)
ball_goals = getBallGoal(goals)


# ball_traj = list()
# for i in range(len(ind_traj)):
#   traj = ind_traj[i]
#   ball_dx = .02 #m
#   tot_len = traj.shape[0]
#   knot = int((tot_len-1)/5)
#   bt = balltraj(ball_goals[i], np.shape(ind_traj[i])[0], knot+1, ball_dx)
#   ball_traj.append(bt)

for i in range(len(ind_traj)):
  traj = ind_traj[i]
  # bt = ball_traj[i]

  for j in range(np.shape(traj)[0]):
    if viewer.is_alive:
      arm_data.qpos[:] = traj[j,:]
      # arm_data.qpos[7:] = bt[j,:]

      mp.mj_step(arm_model, arm_data)
      viewer.render()
      # sleep(dt)
    else:
        break

  sleep(2)

# close
viewer.close()


