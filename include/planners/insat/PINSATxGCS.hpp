#ifndef PINSATxGCS_PLANNER_HPP
#define PINSATxGCS_PLANNER_HPP

#include <future>
#include <utility>
#include <planners/GepasePlanner.hpp>
#include <planners/insat/INSATxGCS.hpp>
#include <common/insat/InsatState.hpp>
#include <common/insat/InsatEdge.hpp>

namespace ps
{

  class PINSATxGCS : virtual public GepasePlanner, virtual public INSATxGCS
  {
  public:
    typedef smpl::intrusive_heap<InsatEdge, IsLesserEdge> EdgeQueueMinType;
    typedef smpl::intrusive_heap<InsatState, IsLesserState> BEType;

    PINSATxGCS(ParamsType planner_params);
    ~PINSATxGCS();
    bool Plan();

  protected:
    void initialize();
    void expandEdgeLoop(int thread_id);
    void expand(InsatEdgePtrType edge_ptr, int thread_id);
    void expandEdge(InsatEdgePtrType insat_edge_ptr, int thread_id);
    void exit();

    EdgeQueueMinType edge_open_list_;
    BEType being_expanded_states_;

    std::vector<InsatEdgePtrType> edge_expansion_vec_;
    InsatActionPtrType dummy_action_ptr_;


  };

}

#endif

