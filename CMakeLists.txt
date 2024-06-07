cmake_minimum_required(VERSION 3.14)
project(libthermo)

# # Set the compiler options
# # The --coverage compiler flag is an alternative to -fprofile-arcs -ftest-coverage for recent version of gcc.
# # In versions 3.13 and later of cmake, the target_link_libraries command can be
# # removed and add_link_options("--coverage") added after the add_compile_options command.
# # GoogleTest requires at least C++14
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_REQUIRED_FLAGS -fconcepts)
# set(CMAKE_CXX_FLAGS "-g -O0 -Wall --coverage")
# set(CMAKE_C_FLAGS --coverage)

# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
# )

# # For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)
add_compile_options(
"$<$<COMPILE_LANGUAGE:CXX>:-Ofast;-DNDEBUG;-fconcepts;-std=c++2a;-march=native;-fpic;-ftree-vectorize>"
)
# # -------------------------------------------------------------------------------------------------
# # GTest
# # -------------------------------------------------------------------------------------------------
# enable_testing()
include_directories(src/include/)
add_executable(
  slac_test
  # src/lib/_C.cpp
  src/lib/functional.cpp
  # src/include/_C.hpp
  # src/include/common.hpp
  src/include/functional.hpp
)

# # link everything together
# target_link_libraries(slac_test GTest::gtest_main)

# if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
#   add_test(
#     NAME slac_test
#     COMMAND slac_test --gtest_output=xml:gtest-output.xml
#   )
#   add_test(
#     NAME Coverage
#     COMMAND gcovr --exclude-directories build/CMakeFiles/slac_test.dir/src/lib/tests --cobertura coverage.cobertura.xml
#     WORKING_DIRECTORY ..
#   )
# else()
#   add_test(
#     NAME slac_test
#     COMMAND slac_test --gtest_output=xml:/app/gtest-output.xml
#   )
#   add_test(
#     NAME Coverage
#     COMMAND gcovr --exclude-directories build/CMakeFiles/slac_test.dir/src/lib/tests --cobertura coverage.cobertura.xml
#     WORKING_DIRECTORY ..
#   )
# endif()
