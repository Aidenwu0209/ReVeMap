// Headless ScanNet RGB-D frontend for the external ORB-SLAM3 library.
#include "System.h"
#include "sensor.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: revemap_orb_rgbd ORBvoc.txt settings.yaml scene.sens\n";
        return 2;
    }
    cv::setNumThreads(2);
    ScanNetInput input(argv[3]);
    ORB_SLAM3::System slam(argv[1], argv[2], ORB_SLAM3::System::RGBD, false);
    if (slam.GetImageScale() != 1.f) throw std::runtime_error("Expected native sensor scale");
    std::ofstream states("FrameTrajectoryRaw.txt");
    states << "# frame timestamp state T_cw_row_major_m\n";
    for (uint64_t i = 0; i < input.frames; ++i) {
        cv::Mat color, depth;
        input.next(color, depth);
        // Match the nanosecond-rounded nominal timestamps in the frozen runner.
        const double timestamp = std::round(i * 1e9 / 30.) * 1e-9;
        const auto begin = std::chrono::steady_clock::now();
        const auto pose = slam.TrackRGBD(color, depth, timestamp).matrix().eval();
        states << i << ' ' << std::fixed << std::setprecision(6) << timestamp << ' ' << slam.GetTrackingState();
        for (int y=0; y<4; ++y) for (int x=0; x<4; ++x)
            states << ' ' << std::scientific << std::setprecision(9) << pose(y,x);
        states << '\n';
        // Local mapping and loop closure run asynchronously at the nominal camera cadence.
        const auto period = std::chrono::duration<double>(1./30.);
        std::this_thread::sleep_until(begin + period);
    }
    slam.Shutdown();
    states.close();
    slam.SaveTrajectoryEuRoC("LargestMapTrajectory.txt");
    slam.SaveTrajectoryTUM("CameraTrajectory.txt");
    return 0;
}
