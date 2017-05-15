#include "Ctracker.h"
#include "HungarianAlg.h"

#include <GTL/GTL.h>
#include "mygraph.h"
#include "mwbmatching.h"
#include "tokenise.h"

#include "adaboost/adaboost.h"


void CTracker::calcColorHistogram(const cv::Mat& image, const cv::Mat& blobMask, const int& bins, const int& channels,
                        const int& colorspace, cv::Mat& histogram)
{
    const int histSize[] = {bins, bins, bins};
    const float range[] = {0, 255};
    const float* histRanges[3];
    if (colorspace == 1)
    {
        const float rangeH[] = {0, 179};
        histRanges[0] = rangeH;
        histRanges[1] = range;
        histRanges[2] = range;
    }
    else
    {
        //histRanges = {range, range, range};
        histRanges[0] = range;
        histRanges[1] = range;
        histRanges[2] = range;
    }
    int* ch = new int[channels];
    for(int i = 0; i < channels; i++)
        ch[i] = i;

    cv::Mat imageToHist;											// image used to compute the color histogram
    image.copyTo(imageToHist, blobMask);			// the image is filtered with a mask

    if (colorspace == 1) // HSV
        cv::cvtColor(imageToHist, imageToHist, CV_BGR2HSV);
    else
    {
        if (colorspace == 2) // Lab
            cv::cvtColor(imageToHist, imageToHist, CV_BGR2Lab);
        else
        {
            if (colorspace == 3) // Luv
                cv::cvtColor(imageToHist, imageToHist, CV_BGR2Luv);
        }
    }

    cv::calcHist(&imageToHist, 1, ch, blobMask, histogram, channels, histSize, histRanges, true, false);	// color histogram computation

    int normalize_coeff = cv::countNonZero(blobMask);
    histogram = histogram * (1.0f / normalize_coeff);		// histogram normalization
    delete [] ch;
}

void CTracker::createNegativeHistograms(const cv::Mat& image, const cv::Mat& mask, const int n, const int bins,
                                        std::vector<cv::Mat>& negative_histograms)
{
    // creates histograms of random image patches to be used as negative examples
    int min_height = 10;
    int min_width = 10;

    for(int i = 0; i < n; ++i)
    {
        int x_min;
        int y_min;
        int width;
        int height;

        bool is_contained = true;

        while(is_contained)
        {
            x_min = rand() % (image.cols - min_width - 1);
            y_min = rand() % (image.rows - min_height - 1);
            width = rand() % (image.cols - x_min - 1);
            height = rand() % (image.rows - y_min - 1);
            is_contained = cv::countNonZero(mask(cv::Rect(x_min, y_min, width, height))) == 0;
        }

        cv::Rect neg_rect(x_min, y_min, width, height);

        cv::Mat neg_image = image(neg_rect);
        cv::Mat neg_mask = mask(neg_rect);
        cv::Mat histogram;
        //std::cout << image.channels() << " " << std::flush;
        calcColorHistogram(neg_image, neg_mask, bins, image.channels(), 0, histogram);
        negative_histograms.push_back(histogram);
    }
}

// ---------------------------------------------------------------------------
// Tracker. Manage tracks. Create, remove, update.
// ---------------------------------------------------------------------------
CTracker::CTracker(
        bool useLocalTracking,
        DistType distType,
        KalmanType kalmanType,
        MatchType matchType,
        track_t dt_,
        track_t accelNoiseMag_,
        track_t dist_thres_,
        size_t maximum_allowed_skipped_frames_,
        size_t max_trace_length_
        )
    :
      m_useLocalTracking(useLocalTracking),
      m_distType(distType),
      m_kalmanType(kalmanType),
      m_matchType(matchType),
      dt(dt_),
      accelNoiseMag(accelNoiseMag_),
      dist_thres(dist_thres_),
      maximum_allowed_skipped_frames(maximum_allowed_skipped_frames_),
      max_trace_length(max_trace_length_),
      NextTrackID(0)
{
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CTracker::~CTracker(void)
{
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CTracker::Update(
        const std::vector<Point_t>& detections,
        const regions_t& regions,
        cv::Mat image
        )
{
    assert(detections.size() == regions.size());
    std::vector<cv::Rect> rois;
    rois.resize(regions.size());
    for (size_t i = 0; i < rois.size(); i++)
        rois[i] = regions[i].m_rect;

    if (m_useLocalTracking) // NEVER USE IT in HUMAN TRACKING ! by Yilong
    {
        localTracker.Update(tracks, image);
    }

    std::vector< std::vector<cv::Mat >> histograms;
    histograms.resize(detections.size());
    // histograms[i][0] = positive histogram
    // histograms[i][1-n] = negative histograms


    for (size_t i = 0; i < detections.size(); i++)
    {
        cv::Mat target_image(image, rois[i]); //positive sample

        cv::Mat blobMask(target_image.rows, target_image.cols, CV_8UC1, cv::Scalar(255));
        cv::Mat hist_current_positive;
        calcColorHistogram(target_image, blobMask, BINS, CHANNELS, 0, hist_current_positive);
        histograms[i].push_back(hist_current_positive);

        cv::Mat negativeMask(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
        negativeMask(rois[i]) = cv::Scalar(0);
        createNegativeHistograms(image, negativeMask, 10, BINS, histograms[i]);
    }

    // -----------------------------------
    // If there is no tracks yet, then every cv::Point begins its own track.
    // -----------------------------------
    if (tracks.size() == 0)
    {
        // If no tracks yet
        for (size_t i = 0; i < detections.size(); ++i)
        {
            tracks.push_back(std::make_unique<CTrack>(detections[i], regions[i], dt, accelNoiseMag, NextTrackID++, m_kalmanType == FilterRect));
        }
    }

    size_t N = tracks.size();        // треки
    size_t M = detections.size();    // детекты

    assignments_t assignment(N, -1); // назначения

    if (!tracks.empty())
    {
        // The matrix of distances from the N-th track to the M-th detection
        distMatrix_t Cost(N * M);
        distMatrix_t CostColor(N * M); // distance of color histograms

        // -----------------------------------
        // There are already tracks, we will compose a distance matrix
        // -----------------------------------
        track_t maxCost = 0;
        switch (m_distType)
        {
        case CentersDist:
            for (size_t i = 0; i < tracks.size(); i++)
            {
                for (size_t j = 0; j < detections.size(); j++)
                {
                    auto dist = tracks[i]->CalcDist(detections[j]);
                    auto color_prediction = 1.0f;
                    if (!tracks[i]->color_never_updated)
                        color_prediction = (tracks[i]->classifier).predict(cv::Point(0, 0), histograms[j][0]);
                    else
                        color_prediction = -1.0f;

                    Cost[i + j * N] = dist;
                    CostColor[i + j * N] = color_prediction;
                    if (dist > maxCost)
                    {
                        maxCost = dist;
                    }
                }
            }
            break;

        case RectsDist:
            for (size_t i = 0; i < tracks.size(); i++)
            {
                for (size_t j = 0; j < detections.size(); j++)
                {
                    auto dist = tracks[i]->CalcDist(regions[j].m_rect);
                    auto color_prediction = 1.0f;
                    if (!tracks[i]->color_never_updated)
                        color_prediction = (tracks[i]->classifier).predict(cv::Point(0, 0), histograms[j][0]);
                    else
                        color_prediction = -1.0f;

                    Cost[i + j * N] = dist;
                    CostColor[i + j * N] = color_prediction;
                    if (dist > maxCost)
                    {
                        maxCost = dist;
                    }
                }
            }
            break;
        }

        // Use methods in Hoshikawa, et al. (2009) in order to normalize the Euclidean distance
        /* distMatrix_t Cost_backup(Cost);
        const double DISTANCE_THRESHOLD = 15.0;
        for (size_t i = 0; i < tracks.size(); i++)
        {
            int count_near = 0;
            for (size_t j = 0; j < detections.size(); j++)
            {
                if (Cost[i + j * N] <= DISTANCE_THRESHOLD)
                    count_near += 1;
            }

            double alpha;
            if (count_near == 0)
            {
                alpha = 0;
                //the case when alpha-i = 0
            }
            else if (count_near == 1)
            {
                alpha = 1;
                // the case when alpha-i = 1
            }
            else if (count_near >= 2)
            {
                alpha = -1; //recalculate
                double distance_min = 999999.0;

                assert(m_distType == CentersDist);
                for (size_t j = 0; j < detections.size(); j++)
                    for (size_t k = j + 1; k < detections.size(); k++)
                    {
                        double dist = sqrt((detections[j].x - detections[k].x) * (detections[j].x - detections[k].x)
                                (detections[j].y - detections[k].y) * (detections[j].y - detections[k].y));
                        if (dist < distance_min)
                            distance_min = dist;
                    }
                alpha = distance_min / DISTANCE_THRESHOLD;
                // the case when n >= 2
            }

            for (size_t j = 0; j < detections.size(); j++)
            {
                double cost_dist = Cost[i + j * N];
                double cost_color = 10.0 / (1 + exp(CostColor[i + j * N]));
                Cost[i + j * N] = alpha * cost_dist + (1 - alpha)  * cost_color;
            }

        } */  //seems not working

        // for debug (yilong)
        printf("--------- Distance Matrix ---------\n"
               "   ROW: DETECTION(N), COL: TRACK   \n");

        for (size_t j = 0; j < detections.size(); j++)
        {
            for (size_t i = 0; i < tracks.size(); i++)
            {
                auto  dist = Cost[i + j * N];
                printf("%8.3f", dist);
            }
            printf("\n");
        }
        printf("\n");

        printf("--------- Color Dist Matrix ---------\n"
                       "   ROW: DETECTION(N), COL: TRACK   \n");

        for (size_t j = 0; j < detections.size(); j++)
        {
            for (size_t i = 0; i < tracks.size(); i++)
            {
                auto dist = CostColor[i + j * N];
                printf("%8.3f", dist);
            }
            printf("\n");
        }
        printf("\n");
    
        // -----------------------------------
        // clean assignment from pairs with large distance
        // -----------------------------------
        for (size_t i = 0; i < assignment.size(); i++)
        {
            if (assignment[i] != -1)
            {
                if (Cost[i + assignment[i] * N] > dist_thres)
                {
                    assignment[i] = -1;
                    tracks[i]->skipped_frames++;
                }
            }
            else
            {
                // If track have no assigned detect, then increment skipped frames counter.
                tracks[i]->skipped_frames++;
            }
        }

        // -----------------------------------
        // If track didn't get detects long time, remove it.
        // -----------------------------------
        for (int i = 0; i < static_cast<int>(tracks.size()); i++)
        {
            if (tracks[i]->skipped_frames > maximum_allowed_skipped_frames)
            {
                tracks.erase(tracks.begin() + i);
                assignment.erase(assignment.begin() + i);
                i--;
            }
        }
    }

    // -----------------------------------
    // Search for unassigned detects and start new tracks for them.
    // -----------------------------------
    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (find(assignment.begin(), assignment.end(), i) == assignment.end())
        {
            tracks.push_back(std::make_unique<CTrack>(detections[i], regions[i], dt, accelNoiseMag, NextTrackID++, m_kalmanType == FilterRect));
        }
    }

    // Update Kalman Filters state

    for (size_t i = 0; i < assignment.size(); i++)
    {
        // If track updated less than one time, than filter state is not correct.

        if (assignment[i] != -1) // If we have assigned detect, then update using its coordinates,
        {
            tracks[i]->skipped_frames = 0;
            tracks[i]->Update(detections[assignment[i]], histograms[assignment[i]],
                              regions[assignment[i]], true, max_trace_length);
            tracks[i]->color_never_updated = false;
        }
        else                     // if not continue using predictions
        {
            tracks[i]->Update(Point_t(), histograms[assignment[i]],
                              CRegion(), false, max_trace_length);
        }
    }
}
