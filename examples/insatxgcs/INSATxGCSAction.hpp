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
 * \file   INSATxGCSAction.hpp
 * \author Ramkumar Natarajan (rnataraj@cs.cmu.edu)
 * \date   2/19/23
 */

#ifndef INSATxGCSActionS_HPP
#define INSATxGCSActionS_HPP

#include <iostream>
#include <random>
#include <common/Types.hpp>
#include <common/insat/InsatAction.hpp>
#include <common/robots/Abb.hpp>
#include "planners/insat/opt/GCSOpt.hpp"


namespace ps
{

  class INSATxGCSAction : public InsatAction
  {

  public:

    typedef std::shared_ptr<INSATxGCSAction> Ptr;
    typedef GCSOpt OptType;
    typedef std::vector<OptType> OptVecType;
    typedef std::shared_ptr<OptVecType> OptVecPtrType;


    INSATxGCSAction(const std::string& type,
                    ParamsType params,
                    OptVecPtrType opt,
                    bool is_expensive = true);

    virtual bool CheckPreconditions(const StateVarsType& state);
    ActionSuccessor GetSuccessor(const StateVarsType& state_vars, int thread_id);
    ActionSuccessor GetSuccessorLazy(const StateVarsType& state_vars, int thread_id);
    ActionSuccessor Evaluate(const StateVarsType& parent_state_vars, const StateVarsType& child_state_vars, int thread_id);

    void UpdateStateToSuccs();

    bool IsFeasible(const StateVarsType& state_vars, int thread_id) override {}
    double GetCostToSuccessor(const StateVarsType& current_state, const StateVarsType& successor_state, int thread_id);
    double getCostToSuccessor(const VecDf& current_state, const VecDf& successor_state, int thread_id);

    /// INSAT
    void setOpt(OptVecPtrType& opt);
    bool isFeasible(MatDf& traj, int thread_id) const override {}
    TrajType optimize(const StateVarsType& s1, const StateVarsType& s2, int thread_id) const override {}
    TrajType warmOptimize(const TrajType& t1, const TrajType& t2, int thread_id) const override {}
    TrajType warmOptimize(const TrajType& t, int thread_id) const override {}
    TrajType optimize(const TrajType& incoming_traj,
                      const StateVarsType &s1,
                      const StateVarsType &s2,
                      int thread_id) override {}
    TrajType optimize(const TrajType& incoming_traj,
                      const std::vector<StateVarsType> &ancestors,
                      const StateVarsType& successor,
                      int thread_id) override {}
    TrajType optimize(const std::vector<StateVarsType> &ancestors,
                              const StateVarsType& successor,
                              int thread_id=0);
    double getCost(const TrajType& traj, int thread_id) const;

    MatDf sampleTrajectory(const GCSTraj::TrajInstanceType &traj, double dt) const;
    double calculateCost(const MatDf &disc_traj) const;

  protected:
    LockType lock_;

    VecDf goal_;
    double path_length_weight_;
    double time_weight_;

    /// Optimizer stuff
    OptVecPtrType opt_;
    std::unordered_map<int, std::vector<int>> state_id_to_succ_id_;
    std::vector<GCSVertex*> gcs_vertices_;

  };

}


#endif //INSATxGCSActionS_HPP
