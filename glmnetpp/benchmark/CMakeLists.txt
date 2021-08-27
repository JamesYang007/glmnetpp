# All macro tests
set(
    BENCHMARKS
    lasso_stress_benchmark
)

foreach( benchmark ${BENCHMARKS} )
    add_executable(${benchmark} ${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cpp)
    target_include_directories(${benchmark} PRIVATE
        ${PROJECT_SOURCE_DIR}/test)
    target_link_libraries(${benchmark}
        benchmark::benchmark_main
        ${PROJECT_NAME}
        Eigen3::Eigen)
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
		target_link_libraries(${benchmark} pthread)
    endif()
endforeach()

# All micro tests
set(
    MICRO_BENCHMARKS
    set_vs_vector_loop
)

foreach( benchmark ${MICRO_BENCHMARKS} )
    add_executable(${benchmark} ${CMAKE_CURRENT_SOURCE_DIR}/micro/${benchmark}.cpp)
    target_include_directories(${benchmark} PRIVATE
        ${PROJECT_SOURCE_DIR}/test)
    target_link_libraries(${benchmark}
        benchmark::benchmark_main
        ${PROJECT_NAME}
        Eigen3::Eigen)
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
		target_link_libraries(${benchmark} pthread)
    endif()
endforeach()
