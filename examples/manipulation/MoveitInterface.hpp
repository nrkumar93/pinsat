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
 * \file   MoveitInterface.hpp
 * \author Ramkumar Natarajan (rnataraj@cs.cmu.edu)
 * \date   9/6/23
 */

#ifndef MANIPULATION_MOVEIT_INTERFACE_HPP
#define MANIPULATION_MOVEIT_INTERFACE_HPP


#include "utils.hpp"
#include <boost/filesystem.hpp>

// MoveIt
#include <moveit_msgs/MotionPlanRequest.h>
#include <moveit_msgs/MotionPlanResponse.h>
#include <moveit_msgs/PlanningScene.h>

namespace fs = boost::filesystem;

namespace ps {


  class ManipulationMoveitInterface {

  public:

    bool init(char* planner_name,
              std::string& model_dir,
              std::string& mprim_dir,
              int num_threads=1) {


      if (!strcmp(planner_name, "insat") &&
          !strcmp(planner_name, "wastar") &&
          !strcmp(planner_name, "pinsat") &&
          !strcmp(planner_name, "rrt") &&
          !strcmp(planner_name, "rrtconnect") &&
          !strcmp(planner_name, "epase") &&
          !strcmp(planner_name, "gepase"))
      {
        throw std::runtime_error("Planner " + std::string(planner_name) + " not identified");
      }

      planner_name_ = planner_name;


      /// Load MuJoCo model
      std::string modelpath = (fs::path(model_dir)/fs::path("irb1600_6_12_realshield.xml")).string();
      mjModel *m = nullptr;
      mjData *d = nullptr;

      setupMujoco(&m,&d,modelpath);
      setupMujoco(&config_.global_m, &config_.global_d, modelpath);
      config_.dof = m->nq;
      config_.shield_h_w.resize(config_.dof);
      config_.shield_h_w << 0, 10, 7, 0.1, 1, 0.1;



      // Experiment parameters
      std::vector<int> scale_vec = {5, 5, 5, 10, 5};

      // Define planner parameters
      planner_params_["num_threads"] = num_threads;
      planner_params_["heuristic_weight"] = 10;
      planner_params_["timeout"] = 20;
      planner_params_["adaptive_opt"] = 0;
      planner_params_["smart_opt"] = 1;
      planner_params_["min_exec_duration"] = 0.5;
      planner_params_["max_exec_duration"] = 0.9;
      planner_params_["num_ctrl_points"] = 7;
      planner_params_["min_ctrl_points"] = 4;
      planner_params_["max_ctrl_points"] = 7;
      planner_params_["spline_order"] = 4;
      planner_params_["sampling_dt"] = 4e-3;

      if ((planner_name_ == "rrt") || (planner_name_ == "rrtconnect"))
      {
        planner_params_["eps"] = 1.0;
        planner_params_["goal_bias_probability"] = 0.05;
        planner_params_["termination_distance"] = TERMINATION_DIST;
      }

      // Robot Params
      IRB1600 robot_params;
      // Insat Params
      InsatParams insat_params(config_.dof, 2 * config_.dof, config_.dof);
      // spline params
      BSplineOpt::BSplineOptParams spline_params(config_.dof,
                                                 planner_params_["num_ctrl_points"],
                                                 planner_params_["spline_order"],
                                                 planner_params_["min_exec_duration"],
                                                 planner_params_["max_exec_duration"],
                                                 BSplineOpt::BSplineOptParams::ConstraintMode::CONTROLPT);
      spline_params.setAdaptiveParams(planner_params_["min_ctrl_points"], planner_params_["max_ctrl_points"]);
      // discretization
      config_.discretization.resize(config_.dof);
      config_.discretization.setOnes();
      config_.discretization *= DISCRETIZATION;
      // discretization = (robot_params.max_q_ - robot_params.min_q_)/50.0;

      // create opt
      auto opt = BSplineOpt(insat_params, robot_params, spline_params, planner_params_);
      opt.SetGoalChecker(std::bind(isGoalState, std::placeholders::_1, TERMINATION_DIST));
      opt_vec_ptr_ = std::make_shared<ManipulationAction::OptVecType>(num_threads, opt);

      // Construct actions
      action_params_["planner_type"] = planner_name_ == "insat" || planner_name_ == "pinsat" ? 1 : -1;
      std::string mprimpath = (fs::path(mprim_dir)/fs::path("irb1600_6_12.mprim")).string();
      constructActions(action_ptrs_, action_params_,
                       modelpath,
                       mprimpath,
                       opt_vec_ptr_, num_threads);

      // Construct BFS actions
      std::string bfsmodelpath = (fs::path(model_dir)/fs::path("realshield_bfs_heuristic.xml")).string();
      setupMujoco(&config_.global_bfs_m, &config_.global_bfs_d, bfsmodelpath);
      std::string bfsmprimpath = (fs::path(mprim_dir)/fs::path("bfs3d.mprim")).string();
      std::vector<std::shared_ptr<Action>> bfs_action_ptrs;
      // constructBFSActions(bfs_action_ptrs, action_params,
      //                    bfsmodelpath, bfsmprimpath, num_threads);

      /// SMPL bfs3d
      if (config_.h_mode == HeuristicMode::SMPL_BFS)
      {
        setupSmplBFS();
      }

      for (auto& a : action_ptrs_)
      {
        std::shared_ptr<ManipulationAction> manip_action_ptr = std::dynamic_pointer_cast<ManipulationAction>(a);
        manip_action_ptrs.emplace_back(manip_action_ptr);
      }
      return true;
    }

    bool solve(moveit_msgs::PlanningScene& planning_scene,
               moveit_msgs::MotionPlanRequest& req,
               moveit_msgs::MotionPlanResponse& res) {
      planner_params_["timeout"] = req.allowed_planning_time;

      if (req.goal_constraints.size() != 1) {
        throw std::runtime_error("Multiple goals not supported");
      }

      if (req.goal_constraints[0].joint_constraints.empty()) {
        throw std::runtime_error("Goal constraints not specified in Joint Space");
      }

      // Set goal conditions
      for (auto& gc : req.goal_constraints[0].joint_constraints)
      {
        config_.goal.push_back(gc.position);
      }
      config_.goal_ee_pos = getEEPosition(config_.goal);
      /// Call SMPL bfs3d after updating ee goal
      if (config_.h_mode == HeuristicMode::SMPL_BFS)
      {
        recomputeBFS();
      }

      StateVarsType start;
      for (auto& sc : req.start_state.joint_state.position)
      {
        start.push_back(sc);
      }


      for (auto& op : *opt_vec_ptr_)
      {
        op.updateStartAndGoal(start, config_.goal);
      }

      for (auto& m : manip_action_ptrs)
      {
        m->setGoal(config_.goal);
      }

      /// Clear heuristic cache
      if (config_.h_mode == HeuristicMode::LOS)
      {
        config_.heuristic_cache.clear();
      }

      /// Set BFS heuristic
      std::shared_ptr<Planner> bfs_planner_ptr = std::make_shared<BFSPlanner>(planner_params_);
//        setBFSHeuristic(goals[run], bfs_planner_ptr, bfs_action_ptrs, planner_params);

      // Construct planner
      std::shared_ptr<Planner> planner_ptr;
      constructPlanner(planner_name_, planner_ptr, action_ptrs_, planner_params_, action_params_, opt_vec_ptr_->at(0));

      // Run experiments
      std::vector<double> time_vec, cost_vec;
      std::vector<int> num_edges_vec, threads_used_vec;
      std::vector<int> jobs_per_thread(planner_params_["num_threads"], 0);
      std::unordered_map<std::string, std::vector<double>> action_eval_times;

      std::cout << " | Planner: " << planner_name_
           << " | Heuristic weight: " << planner_params_["heuristic_weight"]
           << " | Number of threads: " << planner_params_["num_threads"]
           << std::endl;
      std::cout <<  "---------------------------------------------------" << std::endl;

      // print start and goal
      std::cout << "start: ";
      for (double i: start)
        std::cout << i << ' ';
      std::cout << std::endl;
      std::cout << "goal: ";
      for (double i: config_.goal)
        std::cout << i << ' ';
      std::cout << std::endl;


      // Set start state
      planner_ptr->SetStartState(start);
      if ((planner_name_ == "rrt") || (planner_name_ == "rrtconnect"))
      {
        planner_ptr->SetGoalState(config_.goal);
      }


      double t=0, cost=0;
      int num_edges=0;

      bool plan_found = planner_ptr->Plan();
      planner_stats_ = planner_ptr->GetStats();

      std::cout << " | Time (s): " << planner_stats_.total_time_
                << " | Cost: " << planner_stats_.path_cost_
                << " | Length: " << planner_stats_.path_length_
                << " | State expansions: " << planner_stats_.num_state_expansions_
                << " | State expansions rate: " << planner_stats_.num_state_expansions_ / planner_stats_.total_time_
                << " | Lock time: " << planner_stats_.lock_time_
                << " | Expand time: " << planner_stats_.cumulative_expansions_time_
                << " | Threads: " << planner_stats_.num_threads_spawned_ << "/" << planner_params_["num_threads"] << std::endl;

      for (auto& [action, times] : planner_stats_.action_eval_times_)
      {
        auto total_time = accumulate(times.begin(), times.end(), 0.0);
        std::cout << action << " mean time: " << total_time/times.size()
             << " | total: " << total_time
             << " | num: " << times.size()
             << std::endl;
      }
      // cout << endl << "------------- Jobs per thread -------------" << endl;
      // for (int tidx = 0; tidx < planner_params["num_threads"]; ++tidx)
      // {
      //     cout << "thread: " << tidx << " jobs: " << planner_stats_.num_jobs_per_thread_[tidx] << endl;
      // }
      // cout << "************************" << endl;
      // getchar();

      double exec_duration = -1;
      if (plan_found)
      {

        time_vec.emplace_back(planner_stats_.total_time_);
        cost_vec.emplace_back(planner_stats_.path_cost_);
        num_edges_vec.emplace_back(planner_stats_.num_evaluated_edges_);

        for (auto& [action, times] : planner_stats_.action_eval_times_)
        {
          action_eval_times[action].insert(action_eval_times[action].end(), times.begin(), times.end());
        }

        threads_used_vec.emplace_back(planner_stats_.num_threads_spawned_);
        for (int tidx = 0; tidx < planner_params_["num_threads"]; ++tidx) {
          jobs_per_thread[tidx] += planner_stats_.num_jobs_per_thread_[tidx];
        }

        std::cout << std::endl << "************************" << std::endl;
        std::cout << "Mean time: " << accumulate(time_vec.begin(), time_vec.end(), 0.0)/time_vec.size() << std::endl;
        std::cout << "Mean cost: " << accumulate(cost_vec.begin(), cost_vec.end(), 0.0)/cost_vec.size() << std::endl;
        std::cout << "Mean threads used: " << accumulate(threads_used_vec.begin(), threads_used_vec.end(), 0.0)/threads_used_vec.size() << "/" << planner_params_["num_threads"] << std::endl;
        std::cout << "Mean evaluated edges: " << roundOff(accumulate(num_edges_vec.begin(), num_edges_vec.end(), 0.0)/double(num_edges_vec.size()), 2) << std::endl;
        // cout << endl << "------------- Mean jobs per thread -------------" << endl;
        // for (int tidx = 0; tidx < planner_params["num_threads"]; ++tidx)
        // {
        //     cout << "thread: " << tidx << " jobs: " << jobs_per_thread[tidx]/num_success << endl;
        // }
        // cout << "************************" << endl;

        // cout << endl << "------------- Mean action eval times -------------" << endl;
        // for (auto [action, times] : action_eval_times)
        // {
        //     cout << action << ": " << accumulate(times.begin(), times.end(), 0.0)/times.size() << endl;
        // }
        // cout << "************************" << endl;

        /// track logs
//        start_log.conservativeResize(start_log.rows()+1, config_.dof);
//        goal_log.conservativeResize(goal_log.rows()+1, config_.dof);
//        for (int i=0; i < rm::dof; ++i)
//        {
//          Eigen::Map<const VecDf> svec(&start[0], config_.dof);
//          Eigen::Map<const VecDf> gvec(&start[0], config_.dof);
//          start_log.bottomRows(1) = svec.transpose();
//          goal_log.bottomRows(1) = gvec.transpose();
//        }


        if ((planner_name_ == "insat") || (planner_name_ == "pinsat"))
        {
          std::shared_ptr<InsatPlanner> insat_planner = std::dynamic_pointer_cast<InsatPlanner>(planner_ptr);
          auto soln_traj = insat_planner->getSolutionTraj();

          /// Saving sampled trajectory
          auto samp_traj = sampleTrajectory(soln_traj.traj_, planner_params_["sampling_dt"]);

          /// Save control points
//          auto soln_ctrl_pts =  drake::math::StdVectorToEigen(soln_traj.traj_.control_points());
//          ctrl_pt_log.conservativeResize(insat_params.lowD_dims_, ctrl_pt_log.cols()+soln_ctrl_pts.cols());
//          ctrl_pt_log.rightCols(soln_ctrl_pts.cols()) = soln_ctrl_pts;
//          ctrl_pt_log.conservativeResize(insat_params.lowD_dims_, ctrl_pt_log.cols()+1);
//          ctrl_pt_log.rightCols(1) = -1*VecDf::Ones(insat_params.lowD_dims_);

          std::cout << "Execution time: " << soln_traj.traj_.end_time() << std::endl;
          std::cout << "Traj converged in: " << soln_traj.story_ << std::endl;
          exec_duration = soln_traj.traj_.end_time();

          auto plan = planner_ptr->GetPlan();

          res.group_name = req.group_name;
          res.trajectory_start = req.start_state;
          res.planning_time = planner_stats_.total_time_;
          for (int i=0; i<samp_traj.cols(); ++i) {
            trajectory_msgs::JointTrajectoryPoint point;
            point.positions = std::vector<double>(samp_traj.col(i).data(), samp_traj.col(i).data()+config_.dof);
            point.velocities = std::vector<double>(samp_traj.col(i).data()+config_.dof, samp_traj.col(i).data()+2*config_.dof);
            point.accelerations = std::vector<double>(samp_traj.col(i).data()+2*config_.dof, samp_traj.col(i).data()+3*config_.dof);
            point.time_from_start = ros::Duration(i*planner_params_["sampling_dt"]);
            res.trajectory.joint_trajectory.points.push_back(point);
          }

        }
        else
        {
          auto plan = planner_ptr->GetPlan();
          res.group_name = req.group_name;
          res.trajectory_start = req.start_state;
          res.planning_time = planner_stats_.total_time_;
          for (int i=0; i<plan.size(); ++i) {
            trajectory_msgs::JointTrajectoryPoint point;
            point.positions = plan[i].state_;
            point.time_from_start = ros::Duration(i*planner_params_["sampling_dt"]);
            res.trajectory.joint_trajectory.points.push_back(point);
          }
        }
        return true;
      }
      else
      {
        res.error_code.val = moveit_msgs::MoveItErrorCodes::PLANNING_FAILED;
        std::cout << " | Plan not found!" << std::endl;
        return false;
      }
      return false;
    }

    /// @brief Return planning statistics from the last call to solve.
    ///
    /// Possible keys to statistics include:
    ///     "initial solution planning time"
    ///     "initial epsilon"
    ///     "initial solution expansions"
    ///     "final epsilon planning time"
    ///     "final epsilon"
    ///     "solution epsilon"
    ///     "expansions"
    ///     "solution cost"
    ///
    /// @return The statistics

    std::map<std::string, double> getPlannerStats() {
      std::map<std::string, double> stats;
      stats["initial solution planning time"] = planner_stats_.total_time_;
      stats["initial epsilon"] = planner_params_["heuristic_weight"];
      stats["initial solution expansions"] = planner_stats_.num_state_expansions_;
      stats["final epsilon planning time"] = planner_stats_.total_time_;
      stats["time"] = planner_stats_.total_time_;
      stats["final epsilon"] = planner_params_["heuristic_weight"];
      stats["solution epsilon"] = planner_params_["heuristic_weight"];
      stats["expansions"] = planner_stats_.num_state_expansions_;
      stats["state_expansions"] = planner_stats_.num_state_expansions_;
      stats["edge_expansions"] = planner_stats_.num_evaluated_edges_;
      stats["solution cost"] = planner_stats_.path_cost_;
      return stats;
    }

  private:
    std::string planner_name_;
    ParamsType planner_params_;
    ParamsType action_params_;
    PlannerStats planner_stats_;

    std::vector<std::shared_ptr<Action>> action_ptrs_;
    std::shared_ptr<ManipulationAction::OptVecType> opt_vec_ptr_;
    std::vector<std::shared_ptr<ManipulationAction>> manip_action_ptrs;


  };



}

#endif