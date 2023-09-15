cmake_minimum_required(VERSION 3.2)

project (parallel_search)
add_compile_options(-std=c++17)

list(APPEND CMAKE_PREFIX_PATH /opt/drake)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(drake REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-std=c++17)

set(CMAKE_CXX_FLAGS "-g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
SET(CMAKE_C_FLAGS_RELEASE "-O3")

include_directories(include
                    ${EIGEN3_INCLUDE_DIR}
                    ${drake_INCLUDE_DIRS}
                    ${MUJOCO_INCLUDE_DIR})
include_directories(/opt/drake/include)

#link_directories(/opt/drake/lib)

set(INSATxGCS_SOURCES
        src/common/State.cpp
        src/common/Edge.cpp
        src/common/insat/InsatEdge.cpp
        src/common/insatxgcs/utils.cpp
        src/planners/Planner.cpp
        src/planners/BFSPlanner.cpp
        src/planners/WastarPlanner.cpp
        src/planners/PwastarPlanner.cpp
        src/planners/PasePlanner.cpp
        src/planners/EpasePlanner.cpp
        src/planners/GepasePlanner.cpp
        src/planners/MplpPlanner.cpp
        src/planners/RrtPlanner.cpp
        src/planners/RrtConnectPlanner.cpp
        src/planners/insat/PinsatPlanner.cpp
        src/planners/insat/InsatPlanner.cpp
        src/planners/insat/INSATxGCS.cpp
        src/planners/insat/PINSATxGCS.cpp
        src/planners/insat/opt/GCSOpt.cpp
)


# EXECUTABLES
add_executable(run_insatxgcs
        examples/insatxgcs/run_insatxgcs.cpp
        examples/insatxgcs/INSATxGCSAction.cpp
        ${INSATxGCS_SOURCES})

target_link_libraries(run_insatxgcs
        ${drake_LIBRARIES}
        pthread)


add_executable(gcsopt_test
        examples/insatxgcs/gcsopt_test.cpp
        src/common/insatxgcs/utils.cpp
        src/planners/insat/opt/GCSOpt.cpp)

target_link_libraries(gcsopt_test
        ${drake_LIBRARIES}
        pthread)

