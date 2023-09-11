cmake_minimum_required(VERSION 3.2)

project (parallel_search)
add_compile_options(-std=c++17)

list(APPEND CMAKE_PREFIX_PATH /opt/drake)
# list(APPEND CMAKE_PREFIX_PATH /opt/ros/noetic)  
set(MUJOCO_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/mujoco-2.3.2/include")
set(MUJOCO_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/mujoco-2.3.2/lib")
set(fmt_DIR /home/shield/mambaforge/lib/cmake/fmt)
set(spdlog_DIR /usr/lib/x86_64-linux-gnu/cmake/spdlog)

find_package(OpenCV REQUIRED)
find_package(SBPL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(drake REQUIRED)
find_package(catkin REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
SET(CMAKE_C_FLAGS_RELEASE "-O3")

include_directories(include
                    ${SBPL_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIR}
                    ${drake_INCLUDE_DIRS}
                    ${MUJOCO_INCLUDE_DIR})
include_directories(/opt/drake/include)
include_directories(/home/shield/mambaforge/envs/ros_noetic/include)

link_directories(${SBPL_LIBRARY_DIRS})
link_directories(${MUJOCO_LIB_DIR})
#link_directories(/opt/drake/lib)

catkin_package(
    INCLUDE_DIRS
        include 
        examples
    LIBRARIES
        manip_moveit_interface 
    CATKIN_DEPENDS
    DEPENDS)

set(SOURCES
    src/common/State.cpp 
    src/common/Edge.cpp
    src/common/insat/InsatEdge.cpp
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
    src/planners/insat/opt/DummyOpt.cpp
    src/planners/insat/opt/BSplineOpt.cpp
)

add_library(manip_moveit_interface STATIC
        # examples/manipulation/utils.cpp
        examples/manipulation/MoveitInterface.cpp
        examples/manipulation/ManipulationActions.cpp
        examples/manipulation/bfs3d.cpp
        ${SOURCES})
target_link_libraries(manip_moveit_interface
        ${catkin_LIBRARIES}
        ${SBPL_LIBRARIES}
        ${drake_LIBRARIES}
        mujoco
        pthread)


# EXECUTABLES
add_executable(run_robot_nav_2d
        examples/robot_nav_2d/run_robot_nav_2d.cpp
        examples/robot_nav_2d/RobotNav2dActions.cpp
        ${SOURCES})

target_link_libraries(run_robot_nav_2d
        ${OpenCV_LIBRARIES}
        ${SBPL_LIBRARIES}
        ${drake_LIBRARIES}
        pthread)

add_executable(run_insat_nav_2d
        examples/insat/robot_nav_2d/run_insat_nav_2d.cpp
        examples/insat/robot_nav_2d/InsatNav2dActions.cpp
        ${SOURCES})

target_link_libraries(run_insat_nav_2d
        ${OpenCV_LIBRARIES}
        ${SBPL_LIBRARIES}
        ${drake_LIBRARIES}
        pthread)

add_executable(run_manipulation
        examples/manipulation/run_manipulation.cpp
        examples/manipulation/ManipulationActions.cpp
        examples/manipulation/bfs3d.cpp
        ${SOURCES})

target_link_libraries(run_manipulation
        mujoco
        ${drake_LIBRARIES}
        pthread)

add_executable(drake_test
        examples/insat/opt/DrakeTest.cpp)

target_link_libraries(drake_test
        mujoco
        ${drake_LIBRARIES})

add_executable(domain_template
        examples/domain_template/domain_template.cpp
        examples/domain_template/ActionsTemplate.cpp
        ${SOURCES})

target_link_libraries(domain_template
        ${drake_LIBRARIES}
        pthread)

add_executable(collfilter
        examples/manipulation/collision_filter_finder.cpp)

install(DIRECTORY include/common include/planners
        DESTINATION /usr/local/include/${PROJECT_NAME})
install(DIRECTORY examples/manipulation # source directory
        DESTINATION /usr/local/include/${PROJECT_NAME} # target directory
        FILES_MATCHING # install only matched files
        PATTERN "*.hpp") # select header files
install(DIRECTORY examples/manipulation # source directory
        DESTINATION /usr/local/include/${PROJECT_NAME} # target directory
        FILES_MATCHING # install only matched files
        PATTERN "*.h") # select header files
install(TARGETS manip_moveit_interface
        EXPORT ${PROJECT_NAME}Config
        ARCHIVE DESTINATION /usr/local/lib/${PROJECT_NAME}
        LIBRARY DESTINATION /usr/local/lib/${PROJECT_NAME}
        RUNTIME DESTINATION /usr/local/bin/${PROJECT_NAME})
export(TARGETS manip_moveit_interface
    NAMESPACE ${PROJECT_NAME}::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
install(EXPORT ${PROJECT_NAME}Config
    DESTINATION /usr/local/lib/cmake/${PROJECT_NAME}
    NAMESPACE ${PROJECT_NAME}::)
