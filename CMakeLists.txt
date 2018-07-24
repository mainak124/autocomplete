cmake_minimum_required(VERSION 3.6)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
project(hello_tensorflow)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

#set(CXX_FLAGS “-Wall -undefined dynamic_lookup”)
#set(CXX_FLAGS "-Wno-invalid-partial-specialization")
#set(CMAKE_CXX_FLAGS “${CXX_FLAGS}”)

link_directories(/usr/lib /usr/local/lib)
link_directories(/Users/mainakbiswas/anaconda3/lib/python3.6/site-packages/tensorflow/contrib/seq2seq/python/ops)

# # Eigen
# find_package(Eigen REQUIRED)
# include_directories(${Eigen_INCLUDE_DIRS})
# 
# # Protobuf
# FIND_PACKAGE(Protobuf 3.4.0 REQUIRED)
# MESSAGE(STATUS "Using protobuf ${Protobuf_VERSION}")

# # tensorflow
# find_package(TensorFlow REQUIRED)
include_directories(/usr/local/include/google/tensorflow /usr/local/include/google/protobuf /usr/local/include/eigen3)
#include_directories(/Users/mainakbiswas/anaconda3/lib/python3.6/site-packages/tensorflow/contrib/seq2seq/python/ops)

add_executable(autocomplete src/loader.cc)
TARGET_LINK_LIBRARIES(autocomplete "/Users/mainakbiswas/anaconda3/lib/python3.6/site-packages/tensorflow/contrib/seq2seq/python/ops/_beam_search_ops.so")
#TARGET_LINK_LIBRARIES(hello_tensorflow -Wl,--allow-multiple-definition -Wl,--whole-archive "/usr/local/lib/libtensorflow_cc.so" -Wl,--no-whole-archive)
#TARGET_LINK_LIBRARIES(hello_tensorflow -Wl,--allow-multiple-definition -Wl,--whole-archive "/usr/local/lib/libtensorflow_framework.so" -Wl,--no-whole-archive)
target_link_libraries(autocomplete tensorflow_cc tensorflow_framework)
#target_link_libraries(autocomplete beam_search_ops)
#target_link_libraries(hello_tensorflow tensorflow_cc tensorflow_framework ${Protobuf_LIBRARIES})
