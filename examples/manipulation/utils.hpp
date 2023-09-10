/*
 * Copyright (c) 2023, Ramkumar Natarajan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Mellon University nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/*!
 * \file   utils.hpp
 * \author Ramkumar Natarajan (rnataraj@cs.cmu.edu)
 * \date   9/6/23
 */


#ifndef MANIPULATION_UTILS_HPP
#define MANIPULATION_UTILS_HPP

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <numeric>
#include <boost/functional/hash.hpp>
#include <drake/math/matrix_util.h>
#include <planners/insat/InsatPlanner.hpp>
#include <planners/insat/PinsatPlanner.hpp>
#include <planners/RrtPlanner.hpp>
#include <planners/RrtConnectPlanner.hpp>
#include <planners/EpasePlanner.hpp>
#include <planners/GepasePlanner.hpp>
#include <planners/WastarPlanner.hpp>
#include <planners/BFSPlanner.hpp>
#include "ManipulationActions.hpp"
#include "mujoco/mujoco.h"
#include "bfs3d.h"
#include "planners/insat/opt/BSplineOpt.hpp"

#define TERMINATION_DIST 0.1
#define BFS_DISCRETIZATION 0.01
#define DISCRETIZATION 0.05

namespace ps {

  enum class HeuristicMode
  {
    EUCLIDEAN = 0,
    LOS,
    SHIELD,
    SMPL_BFS,
    EE
  };

  enum class GoalCheckerMode
  {
    CSPACE = 0,
    EE
  };

  enum class PPMode
  {
    NONE = 0,
    CONTROLPT,
    WAYPT
  };

  struct Config
  {
    std::vector<double> goal;
    Vec3f goal_ee_pos;

    int dof;
    VecDf discretization;

    // Mujoco
    mjModel* global_m = nullptr;
    mjData* global_d = nullptr;
    /// LoS heuristic
    std::unordered_map<size_t, double> heuristic_cache;
    /// shield heuristic weight
    VecDf shield_h_w;

    // BFS
    /// BFS model Mj Handles
    mjModel* global_bfs_m = nullptr;
    mjData* global_bfs_d = nullptr;
    /// State Map for BFS heuristic
    ps::Planner::StatePtrMapType bfs_state_map;
    std::shared_ptr<smpl::BFS_3D> bfs3d;

    // Modes
    HeuristicMode h_mode = HeuristicMode::LOS;
    GoalCheckerMode goal_mode = GoalCheckerMode::CSPACE;
    PPMode pp_mode = PPMode::WAYPT;
  };

  static Config config_;

  static double roundOff(double value, unsigned char prec)
  {
    double pow_10 = pow(10.0, (double)prec);
    return round(value * pow_10) / pow_10;
  }

  static Vec3f getEEPosition(const VecDf& state)
  {
    mju_copy(config_.global_d->qpos, state.data(), config_.global_m->nq);
    mj_fwdPosition(config_.global_m, config_.global_d);

    VecDf ee_pos(3);
    mju_copy(ee_pos.data(), config_.global_d->xpos + 3 * (config_.global_m->nbody - 1), 3);

    return ee_pos;
  }

  static Vec3f getEEPosition(const StateVarsType& state_vars)
  {
    Eigen::Map<const VecDf> state(&state_vars[0], state_vars.size());
    return getEEPosition(state);
  }

  static Vec4f getEERotation(const VecDf& state)
  {
    mju_copy(config_.global_d->qpos, state.data(), config_.global_m->nq);
    mj_fwdPosition(config_.global_m, config_.global_d);

    VecDf ee_rot(4);
    mju_copy(ee_rot.data(), config_.global_d->xquat + 4 * (config_.global_m->nbody - 1), 4);

    return ee_rot;
  }

  static Vec4f getEERotation(const StateVarsType& state_vars)
  {
    Eigen::Map<const VecDf> state(&state_vars[0], state_vars.size());
    return getEERotation(state);
  }

  static bool isGoalState(const StateVarsType& state_vars, double dist_thresh)
  {
    /// Joint-wise threshold
    for (int i=0; i < config_.dof; ++i)
    {
      if (fabs(config_.goal[i] - state_vars[i]) > dist_thresh)
      {
        return false;
      }
    }
    return true;

    /// Euclidean threshold
//    return (computeHeuristic(state_vars) < dist_thresh);
  }

  static bool isEEGoalState(const StateVarsType& state_vars, double dist_thresh)
  {
    Vec3f ee_pos = getEEPosition(state_vars);

    /// Joint-wise threshold
    for (int i=0; i < 3; ++i)
    {
      if (fabs(config_.goal_ee_pos[i] - ee_pos[i]) > dist_thresh)
      {
        return false;
      }
    }
    return true;
  }


  static bool isBFS3DGoalState(const StateVarsType& state_vars, double dist_thresh)
  {
    return false;
  }

  static size_t StateKeyGenerator(const StateVarsType& state_vars)
  {
    size_t seed = 0;
    for (int i=0; i < config_.dof; ++i)
    {
      boost::hash_combine(seed, state_vars[i]);
    }
    return seed;
  }

  static size_t BFS3DStateKeyGenerator(const StateVarsType& state_vars)
  {
    size_t seed = 0;
    for (int i=0; i < 3; ++i)
    {
      boost::hash_combine(seed, state_vars[i]);
    }
    return seed;
  }

  static size_t EdgeKeyGenerator(const EdgePtrType& edge_ptr)
  {
    int controller_id;
    auto action_ptr = edge_ptr->action_ptr_;

    controller_id = std::stoi(action_ptr->GetType());

    size_t seed = 0;
    boost::hash_combine(seed, edge_ptr->parent_state_ptr_->GetStateID());
    boost::hash_combine(seed, controller_id);

    return seed;
  }

  static double computeHeuristicStateToState(const StateVarsType& state_vars_1, const StateVarsType& state_vars_2)
  {
    double dist = 0.0;
    for (int i=0; i < config_.dof; ++i)
    {
      dist += pow(state_vars_2[i]-state_vars_1[i], 2);
    }
    return std::sqrt(dist);
  }

  static double zeroHeuristic(const StateVarsType& state_vars)
  {
    return 0.0;
  }

  static double computeHeuristic(const StateVarsType& state_vars)
  {
    return computeHeuristicStateToState(state_vars, config_.goal);
  }

  static double computeEEHeuristic(const StateVarsType& state_vars)
  {
    Eigen::Map<const VecDf> state(&state_vars[0], state_vars.size());
    Vec3f ee_pos = getEEPosition(state);

    return (config_.goal_ee_pos - ee_pos).norm();
  }

  static double computeLoSHeuristic(const StateVarsType& state_vars)
  {
    size_t state_key = StateKeyGenerator(state_vars);
    if (config_.heuristic_cache.find(state_key) != config_.heuristic_cache.end())
    {
      return config_.heuristic_cache[state_key];
    }

    double h = computeHeuristic(state_vars);
    config_.heuristic_cache[state_key] = h;
    int N = static_cast<int>(h)/9e-1;
    Eigen::Map<const VecDf> p1(&state_vars[0], state_vars.size());
    Eigen::Map<const VecDf> p2(&config_.goal[0], config_.goal.size());

    for (int i=0; i<N; ++i)
    {
      double j = i/static_cast<double>(N);
      VecDf intp_pt = p1*(1-j) + p2*j;

      mju_copy(config_.global_d->qpos, intp_pt.data(), config_.global_m->nq);
      mj_fwdPosition(config_.global_m, config_.global_d);

      if (config_.global_d->ncon > 0)
      {
        config_.heuristic_cache[state_key] = 100;
        break;
      }
    }

    return config_.heuristic_cache[state_key];
  }

  static double computeShieldHeuristic(const StateVarsType& state_vars)
  {

//    double cost = shield_h_w(0) * pow((goal[0] - state_vars[0]),2) +
//                  shield_h_w(1) * pow((state_vars[1]),2) +
//                  shield_h_w(2) * pow((goal[2] - state_vars[2]),2) +
//                  shield_h_w(3) * pow((goal[3] - state_vars[3]),2) +
//                  shield_h_w(4) * pow((goal[4] - state_vars[4]),2) +
//                  shield_h_w(5) * pow((goal[5] - state_vars[5]),2);

    double cost = config_.shield_h_w(0) * pow((config_.goal[0] - state_vars[0]), 2) +
                  config_.shield_h_w(1) * pow((state_vars[1]), 2) +
                  config_.shield_h_w(2) * pow((-M_PI / 2 - state_vars[2]), 2);
    return std::sqrt(cost);
  }

// double computeBFSHeuristic(const StateVarsType& state_vars)
// {
//   size_t state_key = StateKeyGenerator(state_vars);
//   return rm::bfs_state_map[state_key]->GetFValue();
// }

  static void initializeBFS(int length, int width, int height, std::vector<std::vector<int>> occupied_cells)
  {
    config_.bfs3d = std::make_shared<smpl::BFS_3D>(length, width, height);
    for (auto& c : occupied_cells)
    {
      config_.bfs3d->setWall(c[0], c[1], c[2]);
    }
  }

  static void setupSmplBFS()
  {
    Vec3f lwh;
    lwh << config_.global_bfs_m->numeric_data[3] - config_.global_bfs_m->numeric_data[0],
        config_.global_bfs_m->numeric_data[4] - config_.global_bfs_m->numeric_data[1],
        config_.global_bfs_m->numeric_data[5] - config_.global_bfs_m->numeric_data[2];

    int length = static_cast<int>(lwh(0)/BFS_DISCRETIZATION)+1;
    int width = static_cast<int>(lwh(1)/BFS_DISCRETIZATION)+1;
    int height = static_cast<int>(lwh(2)/BFS_DISCRETIZATION)+1;

    std::vector<std::vector<int>> occupied_cells;
    for (int i=0; i<length; ++i)
    {
      for (int j=0; j<width; ++j)
      {
        for (int k=0; k<height; ++k)
        {
          Vec3f xyz;
          xyz << i*BFS_DISCRETIZATION + config_.global_bfs_m->numeric_data[0],
              j*BFS_DISCRETIZATION + config_.global_bfs_m->numeric_data[1],
              k*BFS_DISCRETIZATION + config_.global_bfs_m->numeric_data[2];

          VecDf fullstate(7);
          fullstate << xyz(0), xyz(1), xyz(2), 1, 0, 0, 0;
          mju_copy(config_.global_bfs_d->qpos, fullstate.data(), config_.global_bfs_m->nq);
          mj_fwdPosition(config_.global_bfs_m, config_.global_bfs_d);

          if (config_.global_bfs_d->ncon > 0)
          {
            std::vector<int> occupied_cell;
            occupied_cell.emplace_back(i);
            occupied_cell.emplace_back(j);
            occupied_cell.emplace_back(k);

            occupied_cells.push_back(occupied_cell);
          }
        }
      }
    }

    initializeBFS(length, width, height, occupied_cells);

    std::cout << "Finished setting up SMPL bfs3d environment of size " <<  length << "x" << width << "x" << height
              << " cells containing " << occupied_cells.size() << " occupied cells." << std::endl;
  }

  static void recomputeBFS()
  {
    int x = static_cast<int>((config_.goal_ee_pos(0) - config_.global_bfs_m->numeric_data[0]) / BFS_DISCRETIZATION);
    int y = static_cast<int>((config_.goal_ee_pos(1) - config_.global_bfs_m->numeric_data[1]) / BFS_DISCRETIZATION);
    int z = static_cast<int>((config_.goal_ee_pos(2) - config_.global_bfs_m->numeric_data[2]) / BFS_DISCRETIZATION);

    config_.bfs3d->run(x, y, z);
  }

  static double computeBFSHeuristic(const StateVarsType& state_vars)
  {
    Vec3f ee_pos = getEEPosition(state_vars);

    int x = static_cast<int>((ee_pos(0) - config_.global_bfs_m->numeric_data[0]) / BFS_DISCRETIZATION);
    int y = static_cast<int>((ee_pos(1) - config_.global_bfs_m->numeric_data[1]) / BFS_DISCRETIZATION);
    int z = static_cast<int>((ee_pos(2) - config_.global_bfs_m->numeric_data[2]) / BFS_DISCRETIZATION);
    double cost_per_cell = 1;

    if (!config_.bfs3d->inBounds(x, y, z)) {
      return DINF;
    }
    else if (config_.bfs3d->getDistance(x, y, z) == smpl::BFS_3D::WALL) {
      return DINF;
    }
    else {
      return cost_per_cell * config_.bfs3d->getDistance(x, y, z);
    }
  }

  static void postProcess(std::vector<PlanElement>& path, double& cost, double allowed_time, const std::shared_ptr<Action>& act, BSplineOpt& opt)
  {
    std::cout << "Post processing with timeout: " << allowed_time << std::endl;
    std::shared_ptr<InsatAction> ins_act = std::dynamic_pointer_cast<InsatAction>(act);
    opt.postProcess(path, cost, allowed_time, ins_act.get());
  }

  static void postProcessWithControlPoints(std::vector<PlanElement>& path, double& cost, double allowed_time, const std::shared_ptr<Action>& act, BSplineOpt& opt)
  {
    std::cout << "Post processing with timeout: " << allowed_time << std::endl;
    std::shared_ptr<InsatAction> ins_act = std::dynamic_pointer_cast<InsatAction>(act);
    opt.postProcessWithControlPoints(path, cost, allowed_time, ins_act.get());
  }

  static void setupMujoco(mjModel **m, mjData **d, std::string modelpath)
  {
    *m = nullptr;
    if (std::strlen(modelpath.c_str()) > 4 && !strcmp(modelpath.c_str() + std::strlen(modelpath.c_str()) - 4, ".mjb"))
    {
      *m = mj_loadModel(modelpath.c_str(), nullptr);
    }
    else
    {
      *m = mj_loadXML(modelpath.c_str(), nullptr, nullptr, 0);
    }
    if (!m)
    {
      mju_error("Cannot load the model");
    }
    *d = mj_makeData(*m);
  }

  static MatDf loadMPrims(std::string mprim_file)
  {
    if (!config_.global_m)
    {
      std::runtime_error("Attempting to load motion primitives before Mujoco model. ERROR!");
    }

    /// Load input prims
    MatDf mprims = loadEigenFromFile<MatDf>(mprim_file, ' ');

    /// Input prims contain only one direction. Flip the sign for adding prims in the other direction
    int num_input_prim = mprims.rows();
    mprims.conservativeResize(2*mprims.rows(), mprims.cols());
    mprims.block(num_input_prim, 0, num_input_prim, mprims.cols()) =
        -1*mprims.block(0, 0, num_input_prim, mprims.cols());

    return mprims;
  }

  static void constructActions(std::vector<std::shared_ptr<Action>>& action_ptrs,
                        ParamsType& action_params,
                        std::string& mj_modelpath, std::string& mprimpath,
                        ManipulationAction::OptVecPtrType& opt,
                        int num_threads)
  {
    /// Vectorize simulator handle
    ManipulationAction::MjModelVecType m_vec;
    ManipulationAction::MjDataVecType d_vec;
    for (int i=0; i<num_threads; ++i)
    {
      mjModel* act_m= nullptr;
      mjData * act_d= nullptr;
      setupMujoco(&act_m, &act_d, mj_modelpath);
      m_vec.push_back(act_m);
      d_vec.push_back(act_d);
    }

    /// Load mprims
    auto mprims = loadMPrims(mprimpath);
    mprims *= (M_PI/180.0); /// Input is in degrees. Convert to radians
    action_params["length"] = mprims.rows();

    for (int i=0; i<=action_params["length"]; ++i)
    {
      if (i == action_params["length"])
      {
        auto one_joint_action = std::make_shared<OneJointAtATime>(std::to_string(i), action_params,
                                                                  DISCRETIZATION, mprims,
                                                                  opt, m_vec, d_vec, num_threads, 1);
        action_ptrs.emplace_back(one_joint_action);
      }
      else
      {
        bool is_expensive = (action_params["planner_type"] == 1) ? 1 : 0;
        auto one_joint_action = std::make_shared<OneJointAtATime>(std::to_string(i), action_params,
                                                                  DISCRETIZATION, mprims,
                                                                  opt, m_vec, d_vec, num_threads, is_expensive);
        action_ptrs.emplace_back(one_joint_action);
      }
    }

    // So that the adaptive primitive is tried first
    std::reverse(action_ptrs.begin(), action_ptrs.end());
  }


  static void constructBFSActions(std::vector<std::shared_ptr<Action>>& action_ptrs,
                           ParamsType& action_params,
                           std::string& mj_modelpath, std::string& mprimpath,
                           int num_threads)
  {
    /// Vectorize simulator handle
    ManipulationAction::MjModelVecType m_vec;
    ManipulationAction::MjDataVecType d_vec;
    for (int i=0; i<num_threads; ++i)
    {
      mjModel* act_m= nullptr;
      mjData * act_d= nullptr;
      setupMujoco(&act_m, &act_d, mj_modelpath);
      m_vec.push_back(act_m);
      d_vec.push_back(act_d);
    }

    /// Load mprims
    auto mprims = loadMPrims(mprimpath);
    action_params["length"] = mprims.rows();

    for (int i=0; i<action_params["length"]; ++i)
    {
      auto one_joint_action = std::make_shared<TaskSpaceAction>(std::to_string(i), action_params,
                                                                BFS_DISCRETIZATION, mprims,
                                                                m_vec, d_vec, num_threads, 0);
      action_ptrs.emplace_back(one_joint_action);
    }
  }


  static void constructPlanner(std::string planner_name, std::shared_ptr<Planner>& planner_ptr, std::vector<std::shared_ptr<Action>>& action_ptrs, ParamsType& planner_params, ParamsType& action_params, BSplineOpt& opt)
  {
    if (planner_name == "epase")
      planner_ptr = std::make_shared<EpasePlanner>(planner_params);
    else if (planner_name == "gepase")
      planner_ptr = std::make_shared<GepasePlanner>(planner_params);
    else if (planner_name == "insat")
      planner_ptr = std::make_shared<InsatPlanner>(planner_params);
    else if (planner_name == "pinsat")
      planner_ptr = std::make_shared<PinsatPlanner>(planner_params);
    else if (planner_name == "rrt")
      planner_ptr = std::make_shared<RrtPlanner>(planner_params);
    else if (planner_name == "rrtconnect")
      planner_ptr = std::make_shared<RrtConnectPlanner>(planner_params);
    else if (planner_name == "wastar")
      planner_ptr = std::make_shared<WastarPlanner>(planner_params);
    else
      throw std::runtime_error("Planner type not identified!");

    /// Heuristic
    if (config_.h_mode == HeuristicMode::EUCLIDEAN)
    {
      planner_ptr->SetHeuristicGenerator(std::bind(computeHeuristic, std::placeholders::_1));
    }
    else if (config_.h_mode == HeuristicMode::LOS)
    {
      planner_ptr->SetHeuristicGenerator(std::bind(computeLoSHeuristic, std::placeholders::_1));
    }
    else if (config_.h_mode == HeuristicMode::SHIELD)
    {
      planner_ptr->SetHeuristicGenerator(std::bind(computeShieldHeuristic, std::placeholders::_1));
    }
    else if (config_.h_mode == HeuristicMode::SMPL_BFS)
    {
      planner_ptr->SetHeuristicGenerator(std::bind(computeBFSHeuristic, std::placeholders::_1));
    }
    else if (config_.h_mode == HeuristicMode::EE)
    {
      planner_ptr->SetHeuristicGenerator(std::bind(computeEEHeuristic, std::placeholders::_1));
    }


    planner_ptr->SetActions(action_ptrs);
    planner_ptr->SetStateMapKeyGenerator(bind(StateKeyGenerator, std::placeholders::_1));
    planner_ptr->SetEdgeKeyGenerator(bind(EdgeKeyGenerator, std::placeholders::_1));
    planner_ptr->SetStateToStateHeuristicGenerator(bind(computeHeuristicStateToState, std::placeholders::_1, std::placeholders::_2));

    /// Goal checker
    if (config_.goal_mode == GoalCheckerMode::CSPACE)
    {
      planner_ptr->SetGoalChecker(std::bind(isGoalState, std::placeholders::_1, TERMINATION_DIST));
    }
    else if (config_.goal_mode == GoalCheckerMode::EE)
    {
      planner_ptr->SetGoalChecker(std::bind(isEEGoalState, std::placeholders::_1, TERMINATION_DIST));
    }

    /// PP
    if ((planner_name != "pinsat") && (planner_name != "insat"))
    {
      if (config_.pp_mode == PPMode::WAYPT)
      {
        planner_ptr->SetPostProcessor(std::bind(postProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, action_ptrs[0], opt));
      }
      else if (config_.pp_mode == PPMode::CONTROLPT)
      {
        planner_ptr->SetPostProcessor(std::bind(postProcessWithControlPoints, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, action_ptrs[0], opt));
      }
    }
  }

  static MatDf sampleTrajectory(const drake::trajectories::BsplineTrajectory<double>& traj, double dt=1e-1)
  {
    MatDf sampled_traj;
    int i=0;
    for (double t=0.0; t<=traj.end_time(); t+=dt)
    {
      sampled_traj.conservativeResize(3*config_.dof, sampled_traj.cols() + 1);
      sampled_traj.col(i).head(config_.dof) = traj.value(t);
      sampled_traj.col(i).middleRows(config_.dof, config_.dof) = traj.EvalDerivative(t, 1);
      sampled_traj.col(i).middleRows(2*config_.dof, config_.dof) = traj.EvalDerivative(t, 1);
      ++i;
    }
    return sampled_traj;
  }

}


#endif //MANIPULATION_UTILS_HPP
