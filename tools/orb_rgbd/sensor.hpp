// Sequential ScanNet RGB-D decoding. The 64-byte camera-to-world field is skipped.
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <zlib.h>

class ScanNetInput {
    std::ifstream file;
    int color_width, color_height, depth_width, depth_height;
    cv::Mat remap_x, remap_y;

    template<class T> T get() {
        T value{};
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!file) throw std::runtime_error("Incomplete sensor field");
        return value;
    }
public:
    uint64_t frames;
    explicit ScanNetInput(const char *path): file(path,std::ios::binary) {
        if (!file || get<uint32_t>() != 4) throw std::runtime_error("Unexpected sens format");
        auto name_size=get<uint64_t>(); file.seekg(name_size, std::ios::cur);
        std::array<float,16> color_K, color_T, depth_K, depth_T;
        for (auto matrix : {&color_K,&color_T,&depth_K,&depth_T}) {
            for (auto& entry : *matrix) entry=get<float>();
        }
        for (int j=0;j<16;++j) {
            float expected=(j%5==0)?1:0;
            if (std::abs(color_T[j]-expected)>1e-6 || std::abs(depth_T[j]-expected)>1e-6)
                throw std::runtime_error("This registered stream requires identity camera extrinsics");
        }
        if (get<int32_t>()!=2 || get<int32_t>()!=1) throw std::runtime_error("Unsupported RGB-D format");
        color_width=get<uint32_t>(); color_height=get<uint32_t>();
        depth_width=get<uint32_t>(); depth_height=get<uint32_t>();
        if(get<float>()!=1000.f) throw std::runtime_error("Unexpected depth scale");
        frames=get<uint64_t>();
        remap_x.create(depth_height,depth_width,CV_32F);
        remap_y.create(depth_height,depth_width,CV_32F);
        const double sx=double(depth_width)/color_width, sy=double(depth_height)/color_height;
        for(int y=0;y<depth_height;++y) for(int x=0;x<depth_width;++x) {
            remap_x.at<float>(y,x)=((x-depth_K[2])/depth_K[0]*color_K[0]+color_K[2]+.5)*sx-.5;
            remap_y.at<float>(y,x)=((y-depth_K[6])/depth_K[5]*color_K[5]+color_K[6]+.5)*sy-.5;
        }
    }
    void next(cv::Mat& color, cv::Mat& depth) {
        file.seekg(64+16,std::ios::cur);
        auto nc=get<uint64_t>(), nd=get<uint64_t>();
        std::vector<uchar> jpeg(nc), compressed(nd);
        file.read(reinterpret_cast<char*>(jpeg.data()), nc);
        file.read(reinterpret_cast<char*>(compressed.data()), nd);
        if(!file) throw std::runtime_error("Truncated RGB-D observation");
        cv::Mat rgb=cv::imdecode(jpeg,cv::IMREAD_COLOR), resized;
        if(rgb.size()!=cv::Size(color_width,color_height)) throw std::runtime_error("Invalid RGB dimensions");
        cv::resize(rgb,resized,cv::Size(depth_width,depth_height),0,0,cv::INTER_AREA);
        cv::remap(resized,color,remap_x,remap_y,cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        depth.create(depth_height,depth_width,CV_16UC1);
        uLongf size=depth.total()*depth.elemSize();
        if(uncompress(depth.data,&size,compressed.data(),nd)!=Z_OK || size!=depth.total()*depth.elemSize())
            throw std::runtime_error("Invalid compressed depth");
    }
};
