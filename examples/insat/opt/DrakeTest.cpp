// PS
#include <common/Types.hpp>

// Drake
#include <drake/planning/trajectory_optimization/kinematic_trajectory_optimization.h>
#include <drake/solvers/solve.h>

// MuJoCo
#include <mujoco/mujoco.h>

namespace ps_drake
{
    struct InsatParams
    {
        int lowD_dims_ = 6;
        int fullD_dims_ = 12;
        int aux_dims_ = 6;
    };

    struct ABBParams
    {
        ABBParams() : min_q_(6),
                      max_q_(6),
                      min_dq_(6),
                      max_dq_(6)
        {
            min_q_ << -3.14159, -1.0995, -4.1015, -3.4906, -2.0071, -6.9813;
            max_q_ << 3.14159, 1.9198, 0.9599, 3.4906, 2.0071, 6.9813;
            min_dq_ << -2.618, -2.7925, -2.967, -5.585, -6.9813, -7.854;
            max_dq_ << 2.618, 2.7925, 2.967, 5.585, 6.9813, 7.854;

//            min_dq_*=5;
//            max_dq_*=5;
        }

        VecDf min_q_, max_q_, min_dq_, max_dq_;
    };

    /// Highly reusable Drake Kinematic Trajectory Optimizer
    using drake::planning::trajectory_optimization::KinematicTrajectoryOptimization;
    class DrakeOpt : public KinematicTrajectoryOptimization
    {
        typedef drake::solvers::Binding<drake::solvers::Cost> CostType;
        typedef std::vector<CostType> CostVecType;
        typedef drake::solvers::Binding<drake::solvers::Constraint> ConstraintType;
        typedef std::vector<ConstraintType> ConstraintVecType;

        DrakeOpt(int num_positions, int num_control_points,
                 int spline_order = 4, double duration = 1.0) : KinematicTrajectoryOptimization(num_positions,
                                                                                                num_control_points,
                                                                                                spline_order,
                                                                                                duration),
                                                                                                prog_(this->get_mutable_prog())
        {}

        CostType AddDurationCost(double weight=1.0)
        {
            return prog_.AddLinearCost(weight * this->duration());
        }

        ConstraintType AddDurationConstraint(std::optional<double> lb, std::optional<double> ub)
        {
            return prog_.AddBoundingBoxConstraint(
                    lb.value_or(0), ub.value_or(std::numeric_limits<double>::infinity()),
                    this->duration());
        }


//        ConstraintVecType AddPositionBounds(
//                const Eigen::Ref<const VecDf>& lb,
//                const Eigen::Ref<const VecDf>& ub) {
//            DRAKE_DEMAND(lb.size() == num_positions());
//            DRAKE_DEMAND(ub.size() == num_positions());
//            // This leverages the convex hull property of the B-splines: if all of the
//            // control points satisfy these convex constraints and the curve is inside
//            // the convex hull of these constraints, then the curve satisfies the
//            // constraints for all s (and therefore all t).
//            ConstraintVecType constraints;
//            for (int i = 0; i < num_control_points(); ++i) {
//                auto cn = prog_.AddBoundingBoxConstraint(lb, ub, this->control_points().col(i));
//                constraints.emplace_back(cn);
//            }
//            return constraints;
//        }
//
//        ConstraintVecType AddVelocityBounds(
//                const Eigen::Ref<const VecDf>& lb,
//                const Eigen::Ref<const VecDf>& ub) {
//            DRAKE_DEMAND(lb.size() == num_positions());
//            DRAKE_DEMAND(ub.size() == num_positions());
//
//            // We have q̇(t) = drds * dsdt = ṙ(s) / duration, and duration >= 0, so we
//            // use duration * lb <= ṙ(s) <= duration * ub.
//            //
//            // This also leverages the convex hull property of the B-splines: if all of
//            // the control points satisfy these convex constraints and the curve is
//            // inside the convex hull of these constraints, then the curve satisfies the
//            // constraints for all t.
//            for (int i = 0; i < sym_rdot_->num_control_points(); ++i) {
//                prog_.AddLinearConstraint(sym_rdot_->control_points()[i] >=
//                                          max_duration_ * lb &&
//                                          sym_rdot_->control_points()[i] <= max_duration_ * ub);
//            }
//        }


        drake::solvers::MathematicalProgram& prog_;

    };

    class BSplineOpt
    {
    public:
        typedef drake::planning::trajectory_optimization::KinematicTrajectoryOptimization OptType;
        typedef drake::solvers::Binding<drake::solvers::Cost> CostType;
        typedef drake::solvers::Binding<drake::solvers::Constraint> ConstraintType;
        typedef drake::solvers::MathematicalProgramResult OptResult;
        typedef drake::trajectories::BsplineTrajectory<double> TrajInstanceType;

        // Robot
        typedef ABBParams RobotParamsType;

        struct Result
        {

        };

        BSplineOpt(int num_positions, int num_control_points,
                   int spline_order = 4, double duration = 1.0) : opt_(std::make_shared<OptType>(num_positions,
                                                                                                 num_control_points,
                                                                                                 spline_order,
                                                                                                 duration)),
                                                                                                 prog_(opt_->get_mutable_prog())
        {
            opt_->AddDurationCost(1.0);
//            opt_->AddPathLengthCost(1.0);
            opt_->AddPathLengthCost(0.1);

            opt_->AddPositionBounds(robot_params_.min_q_, robot_params_.max_q_);
            opt_->AddVelocityBounds(robot_params_.min_dq_, robot_params_.max_dq_);

            opt_->AddDurationConstraint(duration, duration);

            std::string mj_modelpath = "../third_party/mujoco-2.3.2/model/abb/irb_1600/irb1600_6_12_shield.xml";
//            std::string mj_modelpath = "../third_party/mujoco-2.3.2/model/abb/irb_1600/irb1600_6_12.xml";
            m_ = mj_loadXML(mj_modelpath.c_str(), nullptr, nullptr, 0);
            d_ = mj_makeData(m_);

        }

        OptResult optimizeWithCallback()
        {
            traj_trace_.clear();
            auto convergenceCallback = [&](const Eigen::Ref<const Eigen::VectorXd>& control_vec)
            {
                int r = opt_->control_points().rows();
                int c = opt_->control_points().cols();
//                MatDf control_matix(r, c);
//                control_matrix << control_vec;
                MatDf control_matrix(r,c);
//                for (int i=0; i<r; ++i)
//                {
//                    for (int j=0; j<c; ++j)
//                    {
//                        control_matrix(i,j) = control_vec(i*c+j);
//                    }
//                }

                for (int j=0; j<c; ++j)
                {
                    for (int i=0; i<r; ++i)
                    {
                        control_matrix(i,j) = control_vec(j*r+i);
                    }
                }

                std::vector<MatDf> control_points;
                for (int i=0; i<control_matrix.cols(); ++i)
                {
                    control_points.emplace_back(control_matrix.col(i));
                }

                TrajInstanceType traj = TrajInstanceType(opt_->basis(), control_points);
                traj_trace_.emplace_back(traj);
            };

            drake::solvers::MatrixXDecisionVariable control_points = opt_->control_points();
            Eigen::Map<drake::solvers::VectorXDecisionVariable> control_vec(control_points.data(), control_points.size());

            prog_.AddVisualizationCallback(convergenceCallback, control_vec);
            drake::solvers::Solve(prog_);

            return drake::solvers::Solve(prog_);
        }


        OptResult optimize(const VecDf& s1, const VecDf& s2, bool log_trace=false)
        {
            const VecDf& q0 = s1.topRows(insat_params_.lowD_dims_);
            const VecDf& qF = s2.topRows(insat_params_.lowD_dims_);
            const VecDf& dq0 = s1.bottomRows(insat_params_.aux_dims_);
            const VecDf& dqF = s2.bottomRows(insat_params_.aux_dims_);

            /// Start constraint
            opt_->AddPathPositionConstraint(q0, q0, 0); // Linear constraint
            opt_->AddPathVelocityConstraint(dq0, dq0, 0); // Linear constraint
            /// Goal constraint
            opt_->AddPathPositionConstraint(qF, qF, 1); // Linear constraint
            opt_->AddPathVelocityConstraint(dqF, dqF, 1); // Linear constraint

            /// Cost
            auto c1 = prog_.AddQuadraticErrorCost(MatDf::Identity(insat_params_.lowD_dims_, insat_params_.lowD_dims_),
                                        q0,opt_->control_points().leftCols(1));
            auto c2 = prog_.AddQuadraticErrorCost(MatDf::Identity(insat_params_.lowD_dims_, insat_params_.lowD_dims_),
                                        qF, opt_->control_points().rightCols(1));

            costs_.emplace_back(c1);
            costs_.emplace_back(c2);

            /// Solve
            if (log_trace)
            {
                return optimizeWithCallback();

            }
            return drake::solvers::Solve(prog_);
        }

        void optimize(const VecDf& s1, const VecDf& s2, int N)
        {
        }

        std::vector<TrajInstanceType> getTrace() {return traj_trace_;}

//        void warmOptimize(const InsatAction* act, const TrajType& traj1, const TrajType & traj2)
//        {
//            int N = traj1.cols()+traj2.cols();
//            TrajType init_traj(traj1.rows(), N);
//            TrajType opt_traj(traj1.rows(), N);
//            TrajType soln_traj(traj1.rows(), N);
//
//            init_traj << traj1, traj2;
//            opt_traj = linInterp(traj1.leftCols(1), traj2.rightCols(1), N);
//
//            for (double i=0.0; i<=1.0; i+=1.0/conv_delta_)
//            {
//                soln_traj = (1-i)*init_traj + i*opt_traj;
//                if (!act->isFeasible(soln_traj))
//                {
//                    break;
//                }
//            }
//            return soln_traj;
//        }


        MatDf sampleTrajectory(const drake::trajectories::BsplineTrajectory<double> &traj) {
            int n = static_cast<int>(traj.end_time()/1e-2);
            double dt = n>=2? 1e-2: traj.end_time()/3.0;
            double collision_delta = 0.9;

            MatDf sampled_traj;
            /// save the init state
            VecDf x1 = traj.value(0.0);
            sampled_traj.conservativeResize(insat_params_.lowD_dims_, sampled_traj.cols()+1);
            sampled_traj.rightCols(1) = x1;
            /// loop over t
            for (double t=0; t<traj.end_time(); )
            {
                /// adaptive dt called ddt
                double ddt = dt;

                VecDf x2 = traj.value(std::min(t+ddt, traj.end_time()));

                /// keep decreasing adaptive t until distance is smaller than threshold
                while ((x2-x1).norm() > collision_delta && ddt > traj.end_time()/5.0)
                {
                    ddt /= 2.0;
                    x2 = traj.value(std::min(t+ddt, traj.end_time()));
                }

                /// once distance smaller than threshold collect the sample
                sampled_traj.conservativeResize(insat_params_.lowD_dims_, sampled_traj.cols()+1);
                sampled_traj.rightCols(1) = x2;

                /// update t with adaptive t
                t=t+ddt;
            }
            return sampled_traj;
        }

        /// Mj
        bool isCollisionFree(const VecDf &state) const
        {
            if (!validateJointLimits(state))
            {
                return false;
            }
            // Set curr configuration
            mju_copy(d_->qpos, state.data(), m_->nq);
            mju_zero(d_->qvel, m_->nv);
            mju_zero(d_->qacc, m_->nv);
            mj_fwdPosition(m_, d_);

            return d_->ncon>0? false: true;
        }

        bool validateJointLimits(const VecDf &state) const
        {
            bool valid = true;
            for (int i=0; i<m_->njnt; ++i)
            {
                if (state(i) >= m_->jnt_range[2*i] && state(i) <= m_->jnt_range[2*i+1])
                {
                    continue;
                }
                valid = false;
                break;
            }
            return valid;
        }

        bool isFeasible(MatDf &traj) const
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

        /// Optimizer
        std::shared_ptr<OptType> opt_;
        drake::solvers::MathematicalProgram& prog_;
        std::vector<TrajInstanceType> traj_trace_;

        InsatParams insat_params_;
        ABBParams robot_params_;

        std::vector<CostType> costs_;
        std::vector<ConstraintType> constraints_;

        /// MJ
        mjModel* m_;
        mjData* d_= nullptr;

    };

}

#include <random>
#include <chrono>
#include <numeric>
#include <iostream>
#include <fstream>

std::random_device rd;
//std::mt19937 gen(rd());  //here you could set the seed, but std::random_device already does that
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

void printControlPts(ps_drake::BSplineOpt& opt)
{
    std::cout << "control pts:\n " << opt.opt_->control_points() << std::endl;
    drake::solvers::MatrixXDecisionVariable control_points = opt.opt_->control_points();
    Eigen::Map<drake::solvers::VectorXDecisionVariable> control_vec(control_points.data(), control_points.size());
    std::cout << "control pts:\n " << control_vec << std::endl;

    int r = opt.opt_->control_points().rows();
    int c = opt.opt_->control_points().cols();
    drake::solvers::MatrixXDecisionVariable control_matix(r, c);
    control_matix << control_vec;

    std::cout << "control matrix:\n " << control_matix << std::endl;
}

void printTraj(drake::trajectories::BsplineTrajectory<double> traj, double dt)
{
    for (double t=0.0; t<=traj.end_time(); t+=dt)
    {
        std::cout << "t: " << t << "\ts: " << traj.value(t).transpose() << std::endl;
    }

    std::cout << "num control pts: " << traj.control_points().size() << std::endl;
    std::cout << "control pt: " << traj.control_points()[0].transpose() << std::endl;
    std::cout << "num_basis_functions: " << traj.basis().num_basis_functions() << std::endl;
}

MatDf sampleTrajectory(const drake::trajectories::BsplineTrajectory<double>& traj, double dt)
{
    ps_drake::InsatParams insat_params;

    MatDf sampled_traj;
    int i=0;
    for (double t=0.0; t<=traj.end_time(); t+=dt)
    {
        sampled_traj.conservativeResize(insat_params.lowD_dims_, sampled_traj.cols()+1);
        sampled_traj.col(i) = traj.value(t);
        ++i;
    }
    return sampled_traj;
}

void loadStartsAndGoalsFromFile(const std::string& start_path, const std::string& goal_path, MatDf& start_mat, MatDf& goal_mat)
{
    start_mat = loadEigenFromFile<MatDf>(start_path);
    goal_mat = loadEigenFromFile<MatDf>(goal_path);
}

void runBVPTest(std::vector<double>& time_log, bool save=false, int test_size=1)
{
    typedef std::chrono::high_resolution_clock Clock;
    using namespace std::chrono;

    ps_drake::InsatParams insat_params;
    ps_drake::ABBParams robot_params;

    MatDf all_traj;
    MatDf trace_log;
    MatDf starts;
    MatDf goals;
//    for (int i=0; i<test_size; ++i)
    int i=0;
    while(i < test_size)
    {
        ps_drake::BSplineOpt opt(insat_params.lowD_dims_, 7, 4, 0.7);
        VecDf st(insat_params.fullD_dims_), go(insat_params.fullD_dims_);
        st.setZero();
        go.setZero();
//        VecDf r1 = genRandomVector(robot_params.min_q_, robot_params.max_q_, insat_params.lowD_dims_);
//        VecDf r2 = genRandomVector(robot_params.min_q_, robot_params.max_q_, insat_params.lowD_dims_);
        std::string starts_path = "../examples/manipulation/resources/shield/starts.txt";
        std::string goals_path = "../examples/manipulation/resources/shield/goals.txt";
        MatDf ss, gg;
        loadStartsAndGoalsFromFile(starts_path, goals_path, ss, gg);
        VecDf r1, r2;
        r1 = ss.row(i);
        r2 = gg.row(i);

        r1 << 0.0, 0.0, -1.57, 0.0, 0.0, 0.0;
//        r1 << 0.0, -1.05, 0.7, 0.0, 0.0, 0.0;

        st.topRows(insat_params.lowD_dims_) = r1;
        go.topRows(insat_params.lowD_dims_) = r2;

        if (!opt.isCollisionFree(st) || !opt.isCollisionFree(go))
        { continue;}
        ++i;

        auto start_time = Clock::now();
        auto result = opt.optimize(st, go, true);
        auto end_time = Clock::now();
        if (result.is_success())
        {
//            std::cout << "solver name: " << result.get_solver_id().name() << std::endl;
//            std::cout << "solver success: " << std::endl;
            auto bspline_traj = opt.opt_->ReconstructTrajectory(result);
            auto traj = opt.sampleTrajectory(bspline_traj);
            if (opt.isFeasible(traj))
            {
                std::cout << "---- TRIAL " << i+1 << "------" << "\nst: " << r1.transpose() << "\ngo: " << r2.transpose() << std::endl;
                traj = sampleTrajectory(opt.opt_->ReconstructTrajectory(result), 5e-3);
                double time_elapsed = duration_cast<duration<double> >(end_time - start_time).count();
                time_log.emplace_back(time_elapsed);

//                for (auto& c: bspline_traj.control_points())
//                {
//                    std::cout << c.transpose() << std::endl;
//                }

                starts.conservativeResize(starts.rows()+1, insat_params.lowD_dims_);
                goals.conservativeResize(goals.rows()+1, insat_params.lowD_dims_);
                starts.bottomRows(1) = r1.transpose();
                goals.bottomRows(1) = r2.transpose();

                all_traj.conservativeResize(insat_params.lowD_dims_, all_traj.cols()+traj.cols());
                all_traj.rightCols(traj.cols()) = traj;

                /// -1 for demarcating
                all_traj.conservativeResize(insat_params.lowD_dims_, all_traj.cols()+1);
                all_traj.rightCols(1) = -1*VecDf::Ones(insat_params.lowD_dims_);

                auto bspline = opt.opt_->ReconstructTrajectory(result);
                std::cout << "start error: " << (bspline.InitialValue() - r1).norm() << std::endl;
                std::cout << "goal error: " << (bspline.FinalValue() - r2).norm() << std::endl;
                std::cout << "time length: " << bspline.end_time() << std::endl;
//                std::cout << "Trace stats: " << std::endl;

//                auto traj_trace = opt.getTrace();
//                for (auto& t : traj_trace)
//                {
//                    std::cout << "start error: " << (t.InitialValue() - r1).norm() ;
//                    std::cout << "  goal error: " << (t.FinalValue() - r2).norm() ;
//                    std::cout << "  time length: " << t.end_time() ;
//                    std::cout << " Feasibility: " << opt.isFeasible(traj) << std::endl;
//
//                    std::cout << "value at 0.5: " << t.value(0.5).transpose() << std::endl;
//                }

            }
            else
            {
                /// save trace
//                std::cout << "Trace stats: " << std::endl;
//                auto traj_trace = opt.getTrace();
//                for (auto& t : traj_trace)
//                {
//                    std::cout << "start error: " << (t.InitialValue() - r1).norm() ;
//                    std::cout << "  goal error: " << (t.FinalValue() - r2).norm() ;
//                    std::cout << "  time length: " << t.end_time() ;
//                    std::cout << " Feasibility: " << opt.isFeasible(traj) << std::endl;
//
//                    auto samp_t = opt.sampleTrajectory(t);
//                    trace_log.conservativeResize(insat_params.lowD_dims_, trace_log.cols()+samp_t.cols());
//                    trace_log.rightCols(samp_t.cols()) = samp_t;
//                }

                continue;
            }


//            printTraj(opt.opt_->ReconstructTrajectory(result), 1e-1);
//            printControlPts(opt);
//            opt.opt_->SetInitialGuess(opt.opt_->ReconstructTrajectory())
        }
    }
    if (save)
    {
        std::string traj_path ="../logs/test_abb_traj.txt";
        std::string traj_trace_path ="../logs/test_abb_traj_trace.txt";
        std::string starts_path ="../logs/test_abb_starts.txt";
        std::string goals_path ="../logs/test_abb_goals.txt";

        all_traj.transposeInPlace();
        writeEigenToFile(traj_path, all_traj);
        trace_log.transposeInPlace();
        writeEigenToFile(traj_trace_path, trace_log);
        writeEigenToFile(starts_path, starts);
        writeEigenToFile(goals_path, goals);

//        std::cout << "starts:\n" << starts << std::endl;
//        std::cout << "goals:\n" << goals << std::endl;
//        std::cout << "alltraj:\n" << starts << std::endl;
//        std::cout << "tracelog:\n" << trace_log << std::endl;

    }

}

void runBVPwithWPTest(std::vector<double>& time_log, bool save=false, int test_size=1)
{
    typedef std::chrono::high_resolution_clock Clock;
    using namespace std::chrono;

    ps_drake::InsatParams insat_params;
    ps_drake::ABBParams robot_params;

    for (int i=0; i<test_size; ++i)
    {
        ps_drake::BSplineOpt opt(insat_params.lowD_dims_, 10);
        VecDf st(insat_params.fullD_dims_), go(insat_params.fullD_dims_), wp(insat_params.fullD_dims_);
        st.setZero();
        wp.setZero();
        go.setZero();
        VecDf r1 = genRandomVector(robot_params.min_q_, robot_params.max_q_, insat_params.lowD_dims_);
        VecDf r2 = genRandomVector(robot_params.min_q_, robot_params.max_q_, insat_params.lowD_dims_);
        VecDf r3 = genRandomVector(robot_params.min_q_, robot_params.max_q_, insat_params.lowD_dims_);
        st.topRows(insat_params.lowD_dims_) = r1;
        wp.topRows(insat_params.lowD_dims_) = r2;
        go.topRows(insat_params.lowD_dims_) = r3;

        std::cout << "---- TRIAL " << i+1 << "------"
        << "\nst: " << r1.transpose()
        << "\nwp: " << r2.transpose()
        << "\ngo: " << r3.transpose() << std::endl;

        auto start_time = Clock::now();
        auto result = opt.optimize(st, go);
        auto end_time = Clock::now();
        if (result.is_success())
        {
            std::cout << "SUCCESS" << std::endl;
            double time_elapsed = duration_cast<duration<double> >(end_time - start_time).count();
            time_log.emplace_back(time_elapsed);

            printTraj(opt.opt_->ReconstructTrajectory(result), 1e-1);
//            opt.opt_->SetInitialGuess(opt.opt_->ReconstructTrajectory())
        }
    }
}

int main()
{
//    std::srand(time(NULL));

    typedef std::chrono::high_resolution_clock Clock;
    using namespace std::chrono;
    using drake::planning::trajectory_optimization::KinematicTrajectoryOptimization;

    ps_drake::InsatParams insat_params;
    ps_drake::ABBParams robot_params;

    int test_size = 100;

    std::vector<double> time_log;
    std::vector<double> init_time_log;

    runBVPTest(time_log, true, test_size);

    double mean = accumulate( time_log.begin(), time_log.end(), 0.0)/time_log.size();
//    for (auto i: time_log)
//        std::cout << i << ' ';
//    std::cout << std::endl;
    std::cout << "num success: " << time_log.size() << std::endl;
    std::cout << "Average: " << mean << std::endl;


    return 0;
}

