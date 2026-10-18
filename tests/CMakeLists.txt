add_executable(nanochatpp_tests eigen_test.cpp tensor_test.cpp)
target_link_libraries(nanochatpp_tests PRIVATE Eigen3::Eigen GTest::gtest_main)

# Error-contract tests always run with checks enabled, even for an unchecked application.
target_compile_definitions(nanochatpp_tests PRIVATE NANOCHAT_ENABLE_VALIDATION=1)

include(GoogleTest)
gtest_discover_tests(nanochatpp_tests)

foreach(validation_mode IN ITEMS 0 1)
    add_executable(validation_mode_${validation_mode}_tests validation_mode_test.cpp)
    target_link_libraries(validation_mode_${validation_mode}_tests PRIVATE Eigen3::Eigen GTest::gtest_main)
    target_compile_definitions(validation_mode_${validation_mode}_tests PRIVATE NANOCHAT_ENABLE_VALIDATION=${validation_mode})
    gtest_discover_tests(validation_mode_${validation_mode}_tests TEST_PREFIX "validation_${validation_mode}.")
endforeach()
