########################################################################
# Utility TEST
########################################################################

add_executable(utility_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/util/eigen_ext_unittest.cpp
    )

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(utility_unittest PRIVATE -Werror -Wextra)
endif()
target_compile_options(utility_unittest PRIVATE -g -Wall -fopenmp)
target_include_directories(utility_unittest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
if (GLMNETPP_ENABLE_COVERAGE)
    target_link_libraries(utility_unittest gcov)
endif()
target_link_libraries(utility_unittest
    ${PROJECT_NAME}
    GTest::gtest_main
    Eigen3::Eigen)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(utility_unittest pthread)
endif()
add_test(utility_unittest utility_unittest)

########################################################################
# Core TEST
########################################################################

add_executable(core_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/core/elastic_net_config_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/lasso_unittest.cpp
    )

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(core_unittest PRIVATE -Werror -Wextra)
endif()
target_compile_options(core_unittest PRIVATE -g -Wall -fopenmp)
target_include_directories(core_unittest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
if (GLMNETPP_ENABLE_COVERAGE)
    target_link_libraries(core_unittest gcov)
endif()
target_link_libraries(core_unittest
    ${PROJECT_NAME}
    GTest::gtest_main
    Eigen3::Eigen)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(core_unittest pthread)
endif()
add_test(core_unittest core_unittest)
