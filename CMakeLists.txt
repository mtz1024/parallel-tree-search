cmake_minimum_required(VERSION "3.10.2")
# set location of CUDA compiler if not already set
if(NOT DEFINED ENV{CUDACXX})
    set(ENV{CUDACXX} "/usr/local/cuda/bin/nvcc")
endif()

project("ParallelTrees" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/bin)

add_library(tree
    lib/tree_helper.cpp
    lib/tree.hpp
)

#########
# build #
#########
add_executable(serial src/full2.cpp)
add_executable(parallel src/fullG2.cu)
target_link_libraries(serial PRIVATE tree)
target_link_libraries(parallel PRIVATE tree)

###########
# install #
###########
install(TARGETS serial
    CONFIGURATIONS Release
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
)
install(TARGETS parallel
    CONFIGURATIONS Release
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
)