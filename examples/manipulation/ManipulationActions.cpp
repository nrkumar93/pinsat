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
 * \file   ManipulationActions.cpp
 * \author Ramkumar Natarajan (rnataraj@cs.cmu.edu)
 * \date   2/19/23
 */

#include "ManipulationActions.hpp"
#include <common/GlorifiedAngles.h>


namespace ps
{
  ManipulationAction::ManipulationAction(const std::string& type,
                                         ParamsType params,
                                         std::string& mj_modelpath,
                                         VecDf ang_discretization,
                                         OptVecPtrType& opt,
                                         bool is_expensive) : InsatAction(type, params, is_expensive),
                                                          discretization_(ang_discretization),
                                                          opt_(opt)
  {
    m_ = mj_loadXML(mj_modelpath.c_str(), nullptr, nullptr, 0);
    d_ = mj_makeData(m_);

    // Caching discrete angles per DOF in the range -M_PI to M_PI
    for (int i=0; i<m_->nq; ++i)
    {
      double ang_lim = discretization_(i)*static_cast<int>(M_PI/discretization_(i));
      int num_angles = 1+(2*static_cast<int>(M_PI/discretization_(i)));
      discrete_angles_[i] = VecDf::LinSpaced(num_angles, -ang_lim, ang_lim);
    }
  }

  bool ManipulationAction::CheckPreconditions(StateVarsType state)
  {
    return true;
  }

  ActionSuccessor ManipulationAction::GetSuccessor(StateVarsType state_vars, int thread_id)
  {
    Eigen::Map<const VecDf> state(&state_vars[0], state_vars.size());
    std::vector<VecDf> successors = GetSuccessor(state);

    std::vector<std::pair<StateVarsType, double>> action_successors;
    for (auto s : successors)
    {
      StateVarsType succ;
      succ.resize(m_->nq);
      VecDf::Map(&succ[0], s.size()) = s;

      double cost = getCostToSuccessor(state, s);

      action_successors.emplace_back(succ, cost);
    }

    return ActionSuccessor(true, action_successors);
  }

  ActionSuccessor ManipulationAction::GetSuccessorLazy(StateVarsType state_vars, int thread_id)
  {

  }

  ActionSuccessor ManipulationAction::Evaluate(StateVarsType parent_state_vars, StateVarsType child_state_vars, int thread_id)
  {
    return GetSuccessor(parent_state_vars, thread_id);
  }

  /// Snap to Grid
  VecDf ManipulationAction::contToDisc(const VecDf & cont_state)
  {
    VecDf disc_state(m_->nq);
    for (int i=0; i<m_->nq; ++i)
    {
      // Normalize angle to -pi to pi. Should already be in that range.
      // cont_state(i) = angles::normalize_angle(cont_state(i));
      // Number of discrete angles in the DOF i
      int n_disc = discrete_angles_[i].size();
      // The offset to be added to find the bin because of negative to positive range
      int offset = (n_disc-1)/2;
      // One of the indexes of the bin (idx1)
      int idx1 = static_cast<int>(cont_state(i)/discretization_(i)) + offset;
      // The idx1 should not exceed bounds
      assert(idx1>=0 && idx1<config_->discrete_angles_[i].size());
      // The second index (idx2) based on idx1
      int idx2 = cont_state(i)>discrete_angles_[i](idx1)?idx1+1:idx1-1;
      idx2 = (idx2<0)?discrete_angles_[i].size()-1:idx2;
      idx2 = (idx2==discrete_angles_[i].size())?0:idx2;
      // The distance to the angles from two instances
      double d1 = fabs(angles::shortest_angular_distance(cont_state(i), discrete_angles_[i](idx1)));
      double d2 = fabs(angles::shortest_angular_distance(cont_state(i), discrete_angles_[i](idx2)));
      // The distance to the angles from two instances
      disc_state(i) = (d1 < d2)?
                      discrete_angles_[i](idx1):
                      discrete_angles_[i](idx2);
    }
    return disc_state;
  }

  /// MuJoCo
  std::vector<VecDf> ManipulationAction::GetSuccessor(const VecDf &state)
  {
    std::vector<VecDf> successors;
    for (int i=0; i<2*m_->nq; ++i)
    {
      VecDf succ(m_->nq);
      for (int j=0; j<m_->nq; ++j)
      {
        succ(j) = state(j) + mprims_(i,j)*discretization_(j);
        succ(j) = angles::normalize_angle(succ(j));
      }
      succ = contToDisc(succ);

      if (!validateJointLimits(succ))
      {
        continue;
      }

      VecDf free_state(m_->nq), con_state(m_->nq);

      // state coll check
      if (isCollisionFree(succ))
      {
        // edge check only if state check passes
        if (isCollisionFree(state, succ, free_state))
        {
          successors.push_back(succ);
        }
      }
    }

    /// Direct edge to goal
//    VecDf free_state(m_->nq);
//    if (isCollisionFree(state, goal, free_state))
//    {
//      successors.push_back(goal);
//    }
//    else
//    {
//      successors.push_back(free_state);
//    }

    return successors;
  }

  bool ManipulationAction::isCollisionFree(StateVarsType &state_vars) const
  {
    Eigen::Map<const VecDf> state(&state_vars[0], state_vars.size());
    return isCollisionFree(state);
  }

  bool ManipulationAction::isCollisionFree(const VecDf &state) const
  {
    // Set curr configuration
    mju_copy(d_->qpos, state.data(), m_->nq);
    mju_zero(d_->qvel, m_->nv);
    mju_zero(d_->qacc, m_->nv);
    mj_fwdPosition(m_, d_);

    return d_->ncon>0? false: true;
  }

  bool ManipulationAction::isCollisionFree(const VecDf &curr, const VecDf &succ, VecDf &free_state) const
  {
    double ang_dist = angles::calcAngDist(curr, succ);
    int n = static_cast<int>(ceil(ang_dist/(1.5*2e-3)));
    double rho = 1.0/n;

    double coll_free = true;
    free_state = curr;
    for (int k=0; k<=n; ++k)
    {
      VecDf interp = angles::interpolateAngle(curr, succ, rho*k);
      if (!isCollisionFree(interp))
      {
        coll_free = false;
        break;
      }
      free_state = interp;
    }
    return coll_free;
  }

  bool ManipulationAction::validateJointLimits(const VecDf &state)
  {
    return true;
  }

  double ManipulationAction::getCostToSuccessor(const VecDf &current_state, const VecDf &successor_state)
  {
    VecDf angle_dist(m_->nq);
    for (int i=0; i<m_->nq; ++i)
    {
      angle_dist(i) = angles::shortest_angular_distance(current_state(i),
                                                        successor_state(i));
    }
    return angle_dist.norm();
  }


  /// INSAT
  void ManipulationAction::setOpt(OptVecPtrType& opt)
  {
    opt_ = opt;
  }

  bool ManipulationAction::isFeasible(TrajType &traj) const
  {
    bool feas = true;
    for (int i=0; i<traj.cols(); ++i)
    {
      if (!isCollisionFree(traj.col(i)))
      {
        feas = false;
        break;
      }
    }
    return feas;
  }

  TrajType ManipulationAction::optimize(const StateVarsType &s1,
                                      const StateVarsType &s2,
                                      int thread_id) const
  {
    Eigen::Map<const VecDf> p1(&s1[0], s1.size());
    Eigen::Map<const VecDf> p2(&s2[0], s2.size());

    return (*opt_)[thread_id].optimize(this, p1, p2);
  }

  TrajType ManipulationAction::warmOptimize(const TrajType &t1,
                                          const TrajType &t2,
                                          int thread_id) const
  {
    return (*opt_)[thread_id].warmOptimize(this, t1, t2);
  }

  double ManipulationAction::getCost(const TrajType &traj, int thread_id) const
  {
    return (*opt_)[thread_id].calculateCost(traj);
  }

}