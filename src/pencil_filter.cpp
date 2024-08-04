#include "gate_fusion/pencil_filter.hpp"

PencilFilter::PencilFilter(int dilatation_size, int dilation_shape)
    : dilatation_size_(dilatation_size), dilation_shape_(dilation_shape) {}

cv::Mat PencilFilter::apply(const cv::Mat& img) {
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

    cv::Mat dialted = dilatation(gray_img);

    cv::Mat gray_img_32_bit;
    gray_img.convertTo(gray_img_32_bit, CV_32F);

    cv::Mat dialted_my;
    cv::divide(gray_img_32_bit, dialted, dialted_my, 255, CV_8U);

    cv::Mat penciled;
    cv::threshold(dialted_my, penciled, 255, 255, cv::THRESH_TRUNC);

    cv::Mat penciled_rgb;
    cv::cvtColor(penciled, penciled_rgb, cv::COLOR_GRAY2BGR);

    return penciled_rgb;
}

cv::Mat PencilFilter::dilatation(const cv::Mat& img) {
    cv::Mat element = cv::getStructuringElement(
        dilation_shape_,
        cv::Size(2 * dilatation_size_ + 1, 2 * dilatation_size_ + 1),
        cv::Point(dilatation_size_, dilatation_size_)
    );
    cv::Mat dilatation_dst;
    cv::dilate(img, dilatation_dst, element);
    return dilatation_dst;
}
