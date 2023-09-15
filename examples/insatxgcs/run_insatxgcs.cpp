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
 * \file   run_manipulation.cpp
 * \author Ramkumar Natarajan (rnataraj@cs.cmu.edu)
 * \date   7/13/23
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <numeric>
#include <functional>
#include <boost/functional/hash.hpp>
#include <drake/math/matrix_util.h>
#include <planners/insat/InsatPlanner.hpp>
#include <planners/insat/INSATxGCS.hpp>
#include <planners/insat/PinsatPlanner.hpp>
#include <planners/RrtPlanner.hpp>
#include <planners/RrtConnectPlanner.hpp>
#include <planners/EpasePlanner.hpp>
#include <planners/GepasePlanner.hpp>
#include <planners/WastarPlanner.hpp>
#include <planners/BFSPlanner.hpp>
#include "INSATxGCSAction.hpp"
#include <mujoco/mujoco.h>
#include <planners/insat/opt/GCSOpt.hpp>
#include <common/insatxgcs/utils.hpp>

using namespace std;
using namespace ps;

#define TERMINATION_DIST 0.1
#define BFS_DISCRETIZATION 0.01
#define DISCRETIZATION 0.05

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

namespace rm
{
  vector<double> goal;
  Eigen::VectorXd goal_value;
  std::unordered_map<int64_t, GCSVertex*> viv;

  int dof;

  // Modes
  HeuristicMode h_mode = HeuristicMode::LOS;
  GoalCheckerMode goal_mode = GoalCheckerMode::CSPACE;
  PPMode pp_mode = PPMode::WAYPT;
}

double roundOff(double value, unsigned char prec)
{
  double pow_10 = pow(10.0, (double)prec);
  return round(value * pow_10) / pow_10;
}

bool isGoalState(const StateVarsType& state_vars, double dist_thresh)
{
  return rm::goal[0] == state_vars[0];
}

size_t StateKeyGenerator(const StateVarsType& state_vars)
{
  static std::hash<int> hasher;
  return hasher(static_cast<int>(state_vars[0]));
}

size_t EdgeKeyGenerator(const EdgePtrType& edge_ptr)
{
  static std::hash<int> hasher;
  return hasher(static_cast<int>(edge_ptr->parent_state_ptr_->GetStateVars()[0]));
  throw std::runtime_error("EdgeKeyGenerator is wrong");
}

double computeHeuristicStateToState(const StateVarsType& state_vars_1, const StateVarsType& state_vars_2)
{
  double dist = 0.0;
  dist += pow(state_vars_2[0]-state_vars_1[0], 2);
  return std::sqrt(dist);
}

double computeHeuristicStateToVec(const StateVarsType& state_vars_1, const Eigen::VectorXd& state_vars_2)
{
  double dist = 0.0;
  auto vtx = rm::viv[static_cast<int>(state_vars_1[0])];
  auto vtx_pt = vtx->set().MaybeGetFeasiblePoint();
  auto poi = vtx_pt.value().head(rm::dof);
  return (state_vars_2-poi).norm();
}

double zeroHeuristic(const StateVarsType& state_vars)
{
  return 0.0;
}

double computeHeuristic(const StateVarsType& state_vars)
{
  return computeHeuristicStateToVec(state_vars, rm::goal_value);
}

void constructActions(vector<shared_ptr<Action>>& action_ptrs,
                      ParamsType& action_params,
                      const std::vector<std::pair<int, int>>& edges_between_regions,
                      INSATxGCSAction::OptVecPtrType& opt,
                      int num_threads)
{
  action_params["length"] = 1;
  for (int i=0; i<=action_params["length"]; ++i)
  {
    auto insatxgcs_action = std::make_shared<INSATxGCSAction>(std::to_string(i),
                                                              action_params,
                                                              opt, 1);
    action_ptrs.emplace_back(insatxgcs_action);
  }
}



void constructPlanner(string planner_name, shared_ptr<Planner>& planner_ptr,
                      vector<shared_ptr<Action>>& action_ptrs,
                      ParamsType& planner_params)
{
  if (planner_name == "epase")
    planner_ptr = std::make_shared<EpasePlanner>(planner_params);
  else if (planner_name == "gepase")
    planner_ptr = std::make_shared<GepasePlanner>(planner_params);
  else if (planner_name == "insat")
    planner_ptr = std::make_shared<InsatPlanner>(planner_params);
  else if (planner_name == "insatxgcs")
    planner_ptr = std::make_shared<INSATxGCS>(planner_params);
  else if (planner_name == "pinsat")
    planner_ptr = std::make_shared<PinsatPlanner>(planner_params);
  else if (planner_name == "rrt")
    planner_ptr = std::make_shared<RrtPlanner>(planner_params);
  else if (planner_name == "rrtconnect")
    planner_ptr = std::make_shared<RrtConnectPlanner>(planner_params);
  else if (planner_name == "wastar")
    planner_ptr = std::make_shared<WastarPlanner>(planner_params);
  else
    throw runtime_error("Planner type not identified!");

  /// Heuristic
  planner_ptr->SetHeuristicGenerator(bind(computeHeuristic, placeholders::_1));
  planner_ptr->SetActions(action_ptrs);
  planner_ptr->SetStateMapKeyGenerator(bind(StateKeyGenerator, placeholders::_1));
  planner_ptr->SetEdgeKeyGenerator(bind(EdgeKeyGenerator, placeholders::_1));
  planner_ptr->SetStateToStateHeuristicGenerator(bind(computeHeuristicStateToState, placeholders::_1, placeholders::_2));

  /// Goal checker
  if (rm::goal_mode == GoalCheckerMode::CSPACE)
  {
    planner_ptr->SetGoalChecker(bind(isGoalState, placeholders::_1, TERMINATION_DIST));
  }
}


std::random_device rd;
std::mt19937 gen(0);  //here you could set the seed, but std::random_device already does that
std::uniform_real_distribution<float> dis(-1.0, 1.0);
VecDf genRandomVector(VecDf& low, VecDf& high, int size)
{
  VecDf range = high-low;
//    VecDf randvec = VecDf::Random(size);
  VecDf randvec = VecDf::NullaryExpr(size,1,[&](){return dis(gen);});
  randvec += VecDf::Constant(size, 1, 1.0);
  randvec /= 2.0;
  randvec = randvec.cwiseProduct(range);
  randvec += low;

  return randvec;
}

void loadStartsAndGoalsFromFile(vector<vector<double>>& starts,
                                vector<vector<double>>& goals,
                                const string& start_path, const string& goal_path)
{
  MatDf start_mat = loadEigenFromFile<MatDf>(start_path);
  MatDf goal_mat = loadEigenFromFile<MatDf>(goal_path);

  for (int i=0; i<start_mat.rows(); ++i)
  {
    std::vector<double> v_st, v_go;
    v_st.resize(rm::dof);
    v_go.resize(rm::dof);
    VecDf::Map(&v_st[0], start_mat.cols()) = start_mat.row(i);
    VecDf::Map(&v_go[0], goal_mat.cols()) = goal_mat.row(i);

    starts.emplace_back(v_st);
    goals.emplace_back(v_go);
  }
}


MatDf sampleTrajectory(const drake::trajectories::CompositeTrajectory<double>& traj, double dt=1e-1)
{
  MatDf sampled_traj;
  int i=0;
  for (double t=0.0; t<=traj.end_time(); t+=dt)
  {
    sampled_traj.conservativeResize(rm::dof, sampled_traj.cols() + 1);
    sampled_traj.col(i) = traj.value(t);
    ++i;
  }
  return sampled_traj;
}


int main(int argc, char* argv[])
{
  setenv("MOSEKLM_LICENSE_FILE", "/home/gaussian/Documents/softwares/mosektoolslinux64x86/mosek.lic", true);
  auto lic = drake::solvers::MosekSolver::AcquireLicense();

  int num_threads;

  if (!strcmp(argv[1], "insat") || !strcmp(argv[1], "insatxgcs") || !strcmp(argv[1], "wastar"))
  {
    if (argc != 2) throw runtime_error("Format: run_robot_nav_2d insat");
    num_threads = 1;
  }
  else if (!strcmp(argv[1], "pinsat") || !strcmp(argv[1], "rrt") || !strcmp(argv[1], "rrtconnect") || !strcmp(argv[1], "epase") || !strcmp(argv[1], "gepase"))
  {
    if (argc != 3) throw runtime_error("Format: run_robot_nav_2d pinsat [num_threads]");
    num_threads = atoi(argv[2]);
  }
  else
  {
    throw runtime_error("Planner " + string(argv[1]) + " not identified");
  }

  string planner_name = argv[1];


  std::vector<HPolyhedron> regions = utils::DeserializeRegions("/home/gaussian/cmu_ri_phd/phd_research/temp_INSATxGCS/INSATxGCS-Planner/src/data/maze.csv");
  auto edges_bw_regions = utils::DeserializeEdges("/home/gaussian/cmu_ri_phd/phd_research/temp_INSATxGCS/INSATxGCS-Planner/src/data/maze_edges.csv");

  int num_positions = 2;
  rm::dof = num_positions;
  int order = 3;
  double h_min = 1e-2;
  double h_max = 1;
  double path_len_weight = 5;
  double time_weight = 5;
  Eigen::VectorXd vel_lb = -5 * Eigen::VectorXd::Ones(num_positions);
  Eigen::VectorXd vel_ub = 5 * Eigen::VectorXd::Ones(num_positions);
  bool verbose = false;



  // Experiment parameters
  int num_runs;
  bool load_starts_goals_from_file = true;

  // Define planner parameters
  ParamsType planner_params;
  planner_params["num_threads"] = num_threads;
  planner_params["heuristic_weight"] = 10;
  planner_params["timeout"] = 50;
  planner_params["adaptive_opt"] = 0;
  planner_params["smart_opt"] = 1;
  planner_params["min_exec_duration"] = 0.5;
  planner_params["max_exec_duration"] = 0.9;
  planner_params["num_ctrl_points"] = 7;
  planner_params["min_ctrl_points"] = 4;
  planner_params["max_ctrl_points"] = 7;
  planner_params["spline_order"] = 4;
  planner_params["sampling_dt"] = 1e-2;

  ofstream log_file;

  if ((planner_params["smart_opt"] == 1) && ((planner_name == "insat") || (planner_name == "pinsat")))
  {
    log_file.open("../logs/" + planner_name + "_smart_" + to_string(num_threads) + ".txt");
  }
  else if ((planner_params["adaptive_opt"] == 1) && ((planner_name == "insat") || (planner_name == "pinsat")))
  {
    log_file.open("../logs/" + planner_name + "_adaptive_" + to_string(num_threads) + ".txt");
  }
  else
  {
    log_file.open("../logs/" + planner_name + "_" + to_string(num_threads) + ".txt");
  }

  if ((planner_name == "rrt") || (planner_name == "rrtconnect"))
  {
    planner_params["eps"] = 1.0;
    planner_params["goal_bias_probability"] = 0.05;
    planner_params["termination_distance"] = TERMINATION_DIST;
  }

  // Generate random starts and goals
  std::vector<vector<double>> starts, goals;
  if (load_starts_goals_from_file)
  {
    std::string starts_path = "../examples/insatxgcs/resources/maze2d/starts.txt";
    std::string goals_path = "../examples/insatxgcs/resources/maze2d/goals.txt";
    loadStartsAndGoalsFromFile(starts, goals, starts_path, goals_path);
  }

  // Insat Params
  InsatParams insat_params(rm::dof, 2 * rm::dof, rm::dof);

  vector<double> all_maps_time_vec, all_maps_cost_vec;
  vector<int> all_maps_num_edges_vec;
  unordered_map<string, vector<double>> all_action_eval_times;
  vector<double> all_execution_time;

  /// save logs
  MatDf start_log, goal_log, traj_log;
  std::string traj_path ="../logs/" + planner_name +"_maze2d_traj.txt";
  std::string starts_path ="../logs/" + planner_name + "_maze2d_starts.txt";
  std::string goals_path ="../logs/" + planner_name +"_maze2d_goals.txt";

//  auto opt = GCSOpt(regions, *edges_bw_regions,
//                    order, h_min, h_max, path_len_weight, time_weight,
//                    vel_lb, vel_ub, verbose);
//  auto opt_vec_ptr = std::make_shared<INSATxGCSAction::OptVecType>(num_threads, opt);
//
//  // Construct actions
//  ParamsType action_params;
//  action_params["planner_type"] = planner_name=="insat" || planner_name=="pinsat"? 1: -1;
//  vector<shared_ptr<Action>> action_ptrs;
//  constructActions(action_ptrs, action_params,
//                   *edges_bw_regions,
//                   opt_vec_ptr, num_threads);
//
//  std::vector<std::shared_ptr<INSATxGCSAction>> ixg_action_ptrs;
//  for (auto& a : action_ptrs)
//  {
//    std::shared_ptr<INSATxGCSAction> ixg_action_ptr = std::dynamic_pointer_cast<INSATxGCSAction>(a);
//    ixg_action_ptrs.emplace_back(ixg_action_ptr);
//  }

  int num_success = 0;
  vector<vector<PlanElement>> plan_vec;

  int run_offset = 0;
  num_runs = starts.size();
  for (int run = run_offset; run < run_offset+num_runs; ++run)
  {

    auto opt = GCSOpt(regions, *edges_bw_regions,
                      order, h_min, h_max, path_len_weight, time_weight,
                      vel_lb, vel_ub, verbose);
    auto opt_vec_ptr = std::make_shared<INSATxGCSAction::OptVecType>(num_threads, opt);

    // Construct actions
    ParamsType action_params;
    action_params["planner_type"] = planner_name=="insat" || planner_name=="pinsat"? 1: -1;
    vector<shared_ptr<Action>> action_ptrs;
    constructActions(action_ptrs, action_params,
                     *edges_bw_regions,
                     opt_vec_ptr, num_threads);

    std::vector<std::shared_ptr<INSATxGCSAction>> ixg_action_ptrs;
    for (auto& a : action_ptrs)
    {
      std::shared_ptr<INSATxGCSAction> ixg_action_ptr = std::dynamic_pointer_cast<INSATxGCSAction>(a);
      ixg_action_ptrs.emplace_back(ixg_action_ptr);
    }

    // Set goal conditions
    Eigen::VectorXd start_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(starts[run].data(), starts[run].size());
    Eigen::VectorXd goal_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(goals[run].data(), goals[run].size());

    VertexId start_vid, goal_vid;
    for (auto& op : *opt_vec_ptr)
    {
      start_vid = op.AddStart(start_vec);
      goal_vid = op.AddGoal(goal_vec);
      op.FormulateAndSetCostsAndConstraints();
    }
    StateVarsType start;
    start.push_back(start_vid.get_value());
    rm::goal.clear();
    rm::goal.push_back(goal_vid.get_value());
    rm::goal_value = goal_vec;

    for (auto& ixg_act : ixg_action_ptrs) {
      ixg_act->UpdateStateToSuccs();
    }

    rm::viv = (*opt_vec_ptr)[0].GetVertexIdToVertexMap();

    // Construct planner
    shared_ptr<Planner> planner_ptr;
    constructPlanner(planner_name, planner_ptr, action_ptrs, planner_params);

    // Run experiments
    vector<double> time_vec, cost_vec;
    vector<int> num_edges_vec, threads_used_vec;
    vector<int> jobs_per_thread(planner_params["num_threads"], 0);
    unordered_map<string, vector<double>> action_eval_times;

    cout << " | Planner: " << planner_name
         << " | Heuristic weight: " << planner_params["heuristic_weight"]
         << " | Number of threads: " << planner_params["num_threads"]
         << " | Number of runs: " << num_runs
         << endl;
    cout <<  "---------------------------------------------------" << endl;

    cout << "Experiment: " << run << endl;
    // print start and goal
    std::cout << "start: ";
    for (double i: starts[run])
      std::cout << i << ' ';
    std::cout << "start VId: " << start_vid.get_value();
    std::cout << std::endl;
    std::cout << "goal: ";
    for (double i: goals[run])
      std::cout << i << ' ';
    std::cout << "goal VId: " << goal_vid.get_value();
    std::cout << std::endl;


    // Set start state
    planner_ptr->SetStartState(start);
    if ((planner_name == "rrt") || (planner_name == "rrtconnect"))
    {
      planner_ptr->SetGoalState(rm::goal);
    }


    double t=0, cost=0;
    int num_edges=0;

    bool plan_found = planner_ptr->Plan();
    auto planner_stats = planner_ptr->GetStats();

    cout << " | Time (s): " << planner_stats.total_time_
         << " | Cost: " << planner_stats.path_cost_
         << " | Length: " << planner_stats.path_length_
         << " | State expansions: " << planner_stats.num_state_expansions_
         << " | State expansions rate: " << planner_stats.num_state_expansions_/planner_stats.total_time_
         << " | Lock time: " <<  planner_stats.lock_time_
         << " | Expand time: " << planner_stats.cumulative_expansions_time_
         << " | Threads: " << planner_stats.num_threads_spawned_ << "/" << planner_params["num_threads"] << endl;

    for (auto& [action, times] : planner_stats.action_eval_times_)
    {
      auto total_time = accumulate(times.begin(), times.end(), 0.0);
      cout << action << " mean time: " << total_time/times.size()
           << " | total: " << total_time
           << " | num: " << times.size()
           << endl;
    }

    double exec_duration = -1;
    if (plan_found)
    {

      time_vec.emplace_back(planner_stats.total_time_);
      all_maps_time_vec.emplace_back(planner_stats.total_time_);
      cost_vec.emplace_back(planner_stats.path_cost_);
      all_maps_cost_vec.emplace_back(planner_stats.path_cost_);
      num_edges_vec.emplace_back(planner_stats.num_evaluated_edges_);
      all_maps_num_edges_vec.emplace_back(planner_stats.num_evaluated_edges_);

      for (auto& [action, times] : planner_stats.action_eval_times_)
      {
        action_eval_times[action].insert(action_eval_times[action].end(), times.begin(), times.end());
        all_action_eval_times[action].insert(all_action_eval_times[action].end(), times.begin(), times.end());
      }

      threads_used_vec.emplace_back(planner_stats.num_threads_spawned_);
      for (int tidx = 0; tidx < planner_params["num_threads"]; ++tidx)
        jobs_per_thread[tidx] += planner_stats.num_jobs_per_thread_[tidx];

      num_success++;

      cout << endl << "************************" << endl;
      cout << "Number of runs: " << num_runs << endl;
      cout << "Mean time: " << accumulate(time_vec.begin(), time_vec.end(), 0.0)/time_vec.size() << endl;
      cout << "Mean cost: " << accumulate(cost_vec.begin(), cost_vec.end(), 0.0)/cost_vec.size() << endl;
      cout << "Mean threads used: " << accumulate(threads_used_vec.begin(), threads_used_vec.end(), 0.0)/threads_used_vec.size() << "/" << planner_params["num_threads"] << endl;
      cout << "Mean evaluated edges: " << roundOff(accumulate(num_edges_vec.begin(), num_edges_vec.end(), 0.0)/double(num_edges_vec.size()), 2) << endl;

      /// track logs
      start_log.conservativeResize(start_log.rows()+1, insat_params.lowD_dims_);
      goal_log.conservativeResize(goal_log.rows()+1, insat_params.lowD_dims_);
      for (int i=0; i < rm::dof; ++i)
      {
        Eigen::Map<const VecDf> svec(&starts[run][0], rm::dof);
        Eigen::Map<const VecDf> gvec(&goals[run][0], rm::dof);
        start_log.bottomRows(1) = svec.transpose();
        goal_log.bottomRows(1) = gvec.transpose();
      }

      if ((planner_name == "insat") || (planner_name == "pinsat") || (planner_name == "insatxgcs"))
      {
        std::shared_ptr<INSATxGCS> ixg_planner = std::dynamic_pointer_cast<INSATxGCS>(planner_ptr);
        auto soln_traj = ixg_planner->getSolutionTraj();

        /// Saving sampled trajectory
        auto samp_traj = sampleTrajectory(soln_traj.traj_, planner_params["sampling_dt"]);
        traj_log.conservativeResize(insat_params.lowD_dims_, traj_log.cols()+samp_traj.cols());
        traj_log.rightCols(samp_traj.cols()) = samp_traj;
        traj_log.conservativeResize(insat_params.lowD_dims_, traj_log.cols()+1);
        traj_log.rightCols(1) = -1*VecDf::Ones(insat_params.lowD_dims_);

        all_execution_time.push_back(soln_traj.traj_.end_time());
        cout << "Execution time: " << soln_traj.traj_.end_time() << endl;
        cout << "Traj converged in: " << soln_traj.story_ << endl;
        exec_duration = soln_traj.traj_.end_time();

        auto plan = planner_ptr->GetPlan();
        plan_vec.emplace_back(plan);
      }
      else
      {
        auto plan = planner_ptr->GetPlan();
        plan_vec.emplace_back(plan);
        exec_duration = plan_vec.size()*planner_params["sampling_dt"];
      }


    }
    else
    {
      cout << " | Plan not found!" << endl;
    }

//    for (auto& op : *opt_vec_ptr) {
//      op.CleanUp();
//    }


    log_file << run << " "
             << planner_stats.total_time_ << " "
             << planner_stats.path_cost_<< " "
             << planner_stats.path_length_<< " "
             << planner_stats.num_state_expansions_<< " "
             << planner_stats.num_evaluated_edges_<< " "
             << planner_stats.num_threads_spawned_<< " "
             << exec_duration<< " "
             << endl;
  }

  StateVarsType dummy_wp(6, -1);
  if ((planner_name == "insat") || (planner_name == "pinsat") || (planner_name == "insatxgcs"))
  {
    /// Dump traj to file
    traj_log.transposeInPlace();
    writeEigenToFile(traj_path, traj_log);

    ofstream traj_fout("../logs/" + planner_name + "_maze2d_path.txt");

    for (auto& p : plan_vec)
    {
      for (auto& wp : p)
      {
        for (auto& j : wp.state_)
        {
          traj_fout << j << " ";
        }
        traj_fout << endl;
      }

      for (auto& j : dummy_wp)
      {
        traj_fout << j << " ";
      }
      traj_fout << endl;
    }

    traj_fout.close();
  }
  else
  {
    ofstream traj_fout("../logs/" + planner_name + "_maze2d_traj.txt");

    for (auto& p : plan_vec)
    {
      for (auto& wp : p)
      {
        for (auto& j : wp.state_)
        {
          traj_fout << j << " ";
        }
        traj_fout << endl;
      }

      for (auto& j : dummy_wp)
      {
        traj_fout << j << " ";
      }
      traj_fout << endl;
    }

    traj_fout.close();
  }

  writeEigenToFile(starts_path, start_log);
  writeEigenToFile(goals_path, goal_log);


  cout << endl << "************ Global Stats ************" << endl;
  cout << "Success rate: " << double(num_success)/num_runs << endl;
  cout << "Mean time: " << accumulate(all_maps_time_vec.begin(), all_maps_time_vec.end(), 0.0)/all_maps_time_vec.size() << endl;
  cout << "Mean cost: " << accumulate(all_maps_cost_vec.begin(), all_maps_cost_vec.end(), 0.0)/all_maps_cost_vec.size() << endl;
  cout << "Mean evaluated edges: " << roundOff(accumulate(all_maps_num_edges_vec.begin(), all_maps_num_edges_vec.end(), 0.0)/double(all_maps_num_edges_vec.size()), 2) << endl;
  cout << "Mean trajectory duration: " << roundOff(reduce(all_execution_time.begin(), all_execution_time.end())/double(all_execution_time.size()), 2) << endl;
  cout << endl << "************************" << endl;

  cout << endl << "------------- Mean action eval times -------------" << endl;
  for (auto [action, times] : all_action_eval_times)
  {
    cout << action << ": " << accumulate(times.begin(), times.end(), 0.0)/times.size() << endl;
  }
  cout << "************************" << endl;

}

