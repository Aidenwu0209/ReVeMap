cmake_minimum_required(VERSION 3.16)
project(revemap_orb_rgbd LANGUAGES CXX)
set(ORB_SLAM3_ROOT "" CACHE PATH "Built ORB-SLAM3 source checkout")
if(NOT EXISTS "${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so")
  message(FATAL_ERROR "Build ORB-SLAM3 first, then set ORB_SLAM3_ROOT to its checkout")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
add_executable(revemap_orb_rgbd track.cc)
target_compile_features(revemap_orb_rgbd PRIVATE cxx_std_14)
target_include_directories(revemap_orb_rgbd PRIVATE
  "${ORB_SLAM3_ROOT}" "${ORB_SLAM3_ROOT}/include" "${ORB_SLAM3_ROOT}/include/CameraModels"
  "${ORB_SLAM3_ROOT}/Thirdparty/Sophus" ${OpenCV_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS})
target_link_libraries(revemap_orb_rgbd PRIVATE
  "${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so" "${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so"
  "${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib/libg2o.so" ${OpenCV_LIBS} ${Pangolin_LIBRARIES}
  Eigen3::Eigen ZLIB::ZLIB Boost::serialization)
