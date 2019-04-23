
#may need target_link_options to add -arch flags to nvlink

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


project(dp LANGUAGES CUDA VERSION 0.1.0)




# Find the  CUDA toolkit directory so we can link against CUDA libraries
# https://gitlab.kitware.com/cmake/cmake/issues/17816
set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}")
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
find_library(CUDART_LIB cudart
             HINTS
             "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
             "${CUDA_TOOLKIT_ROOT_DIR}/lib"
             "${CUDA_TOOLKIT_ROOT_DIR}"
)


# create imported library for libcudart
add_library(CUDA::cudart IMPORTED INTERFACE)
set_target_properties(CUDA::cudart PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
  INTERFACE_LINK_LIBRARIES
    "${CUDART_LIB}")
  






# CUDA flags
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_VERBOSE_BUILD OFF)


add_executable(dp main.cu dp.cu)

# use c++11 not g++11
set_target_properties(dp PROPERTIES CMAKE_CXX_EXTENSIONS OFF)

set_target_properties(dp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(dp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
-lineinfo
-Xcompiler=-Wall 
-Xcompiler=-Wextra 
-Xcompiler=-Wcast-qual 
-Xcompiler=-Wcast-align 
-Xcompiler=-Wstrict-aliasing 
-Xcompiler=-Wpointer-arith 
-Xcompiler=-Winit-self 
-Xcompiler=-Wshadow 
-Xcompiler=-Wswitch-enum 
-Xcompiler=-Wredundant-decls 
-Xcompiler=-Wfloat-equal 
-Xcompiler=-Wundef 
-Xcompiler=-Wvla
-Xcompiler=-fmax-errors=1 
-Xcompiler=-Wfatal-errors
>
)

# have to add the arch flags here, as there is no way to pass an arch flag to the device link step in CMAKE
list(APPEND CMAKE_CUDA_FLAGS "-arch=sm_35 -gencode=arch=compute_61,code=sm_61")


target_link_libraries(dp PRIVATE CUDA::cudart)


