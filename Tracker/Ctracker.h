#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <array>

#include "defines.h"
#include "track.h"
#include "LocalTracker.h"

// --------------------------------------------------------------------------
class CTracker
{
public:
    enum DistType
    {
        CentersDist = 0,
        RectsDist = 1
    };
    enum KalmanType
    {
        FilterCenter = 0,
        FilterRect = 1
    };
    enum MatchType
    {
        MatchHungrian = 0,
        MatchBipart = 1
    };

    CTracker(bool useLocalTracking,
             DistType distType,
             KalmanType kalmanType,
             MatchType matchType,
             track_t dt_,
             track_t accelNoiseMag_,
             track_t dist_thres_ = 60,
             size_t maximum_allowed_skipped_frames_ = 10,
             size_t max_trace_length_ = 10);
    ~CTracker(void);

    tracks_t tracks;
    void Update(const std::vector<Point_t>& detections,
                const regions_t& regions, cv::Mat image);

    static void calcColorHistogram(const cv::Mat& image, const cv::Mat& blobMask, const int& bins, const int& channels,
                                   const int& colorspace, cv::Mat& histogram);

    static void createNegativeHistograms(const cv::Mat& image, const cv::Mat& mask, const int n, const int bins,
                                         std::vector<cv::Mat>& negative_histograms);


private:
    // Use local tracking for regions between two frames
    bool m_useLocalTracking;

    DistType m_distType;
    KalmanType m_kalmanType;
    MatchType m_matchType;

    // Filter time interval
    track_t dt;

    track_t accelNoiseMag;

    // Distance threshold. If the points are arcs from a friend at a distance,
    // Exceeding this threshold, this pair is not considered in the assignment problem
    track_t dist_thres;
    // The maximum number of frames that a track is saved without receiving measurement data
    size_t maximum_allowed_skipped_frames;
    // Max trace length
    size_t max_trace_length;

    size_t NextTrackID;

    LocalTracker localTracker;
};
