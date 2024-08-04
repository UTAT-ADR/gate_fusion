#ifndef PENCIL_FILTER_H
#define PENCIL_FILTER_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

class PencilFilter {
public:
    PencilFilter(int dilatation_size = 2, int dilation_shape = cv::MORPH_ELLIPSE);
    cv::Mat apply(const cv::Mat& img);

private:
    cv::Mat dilatation(const cv::Mat& img);
    int dilatation_size_;
    int dilation_shape_;
};

#endif // PENCIL_FILTER_H
