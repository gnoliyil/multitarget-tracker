#include "opencv2/opencv.hpp"
#include "BackgroundSubtract.h"
#include "Detector.h"

#include <opencv2/highgui/highgui_c.h>
#include "Ctracker.h"
#include <iostream>
#include <vector>

//------------------------------------------------------------------------
// Mouse callbacks
//------------------------------------------------------------------------
void mv_MouseCallback(int event, int x, int y, int /*flags*/, void* param)
{
    if (event == cv::EVENT_MOUSEMOVE)
    {
        cv::Point2f* p = (cv::Point2f*)param;
        if (p)
        {
            p->x = static_cast<float>(x);
            p->y = static_cast<float>(y);
        }
    }
}

// ----------------------------------------------------------------------
// set to 0 for Bugs tracking example
// set to 1 for mouse tracking example
// ----------------------------------------------------------------------
 #define ExampleNum 0

int main(int argc, char** argv)
{
    std::string inFile("../data/atrium.avi");

    if (argc > 1)
    {
        inFile = argv[1];
    }

    std::string outFile;
    if (argc > 2)
    {
        outFile = argv[2];
    }

    cv::VideoWriter writer;

#if ExampleNum
    cv::Scalar Colors[] = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 127, 255), cv::Scalar(127, 0, 255), cv::Scalar(127, 0, 127) };
    cv::VideoCapture capture(inFile);
    if (!capture.isOpened())
    {
        return 1;
    }
    cv::namedWindow("Video");
    cv::Mat frame;
    cv::Mat gray;

    const int StartFrame = 0;
    capture.set(cv::CAP_PROP_POS_FRAMES, StartFrame);

    const int fps = std::max(1, static_cast<int>(capture.get(cv::CAP_PROP_FPS) + 0.5));

    capture >> frame;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // If true then trajectories will be more smooth and accurate
    // But on high resolution videos with many objects may be to slow
    bool useLocalTracking = true;

    CDetector detector(BackgroundSubtract::ALG_MOG, useLocalTracking, gray);
    detector.SetMinObjectSize(cv::Size(gray.cols / 50, gray.rows / 50));
    //detector.SetMinObjectSize(cv::Size(4, 2));

    CTracker tracker(useLocalTracking,
        CTracker::RectsDist,
        CTracker::FilterRect,
        CTracker::MatchHungrian,
        0.2f,                // Delta time for Kalman filter
        0.1f,                // Accel noise magnitude for Kalman filter
        gray.cols / 20.0f,   // Distance threshold between two frames
        fps,                 // Maximum allowed skipped frames
        5 * fps              // Maximum trace length
        );

    int k = 0;

    double freq = cv::getTickFrequency();

    int64 allTime = 0;

    bool manualMode = false;
    int framesCounter = StartFrame + 1;
    while (k != 27)
    {
        capture >> frame;
        if (frame.empty())
        {
            break;
        }
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (!writer.isOpened())
        {
            writer.open(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), capture.get(cv::CAP_PROP_FPS), frame.size(), true);
        }

        int64 t1 = cv::getTickCount();

        const std::vector<Point_t>& centers = detector.Detect(gray);
        const regions_t& regions = detector.GetDetects();

        tracker.Update(centers, regions, gray);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = static_cast<int>(1000 * (t2 - t1) / freq + 0.5);

        std::cout << "Frame " << framesCounter << ": tracks = " << tracker.tracks.size() << ", time = " << currTime << std::endl;

        for (size_t i = 0; i < tracker.tracks.size(); i++)
        {
            if (tracker.tracks[i]->IsRobust(fps, 0.75f, cv::Size2f(0.2f, 4.0f)))
            {
                cv::rectangle(frame, tracker.tracks[i]->GetLastRect(), cv::Scalar(0, 255, 0), 1, CV_AA);

                for (size_t j = 0; j < tracker.tracks[i]->trace.size() - 1; j++)
                {
                    cv::line(frame, tracker.tracks[i]->trace[j], tracker.tracks[i]->trace[j + 1], Colors[tracker.tracks[i]->track_id % 9], 2, CV_AA);
                }
            }
        }

        detector.CalcMotionMap(frame);

        cv::imshow("Video", frame);

        int waitTime = manualMode ? 0 : std::max<int>(1, 1000 / fps - currTime);
        k = cv::waitKey(waitTime);

        if (k == 'm' || k == 'M')
        {
            manualMode = !manualMode;
        }

        if (writer.isOpened())
        {
            writer << frame;
        }

        ++framesCounter;
        if (framesCounter > 215)
        {
            //break;
        }
    }

    std::cout << "work time = " << (allTime / freq) << std::endl;
    cv::waitKey(0);

#else

    int k = 0;
    cv::Scalar Colors[] = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 255, 255) };
    cv::namedWindow("Video");
    cv::Mat frame = cv::Mat(800, 800, CV_8UC3);

    if (!writer.isOpened())
    {
        writer.open(outFile, cv::VideoWriter::fourcc('P', 'I', 'M', '1'), 20, frame.size(), true);
    }

    // Set mouse callback
    cv::Point2f pointXY;
    cv::setMouseCallback("Video", mv_MouseCallback, (void*)&pointXY);

    bool useLocalTracking = false;

    CTracker tracker(useLocalTracking, CTracker::CentersDist, CTracker::FilterCenter, CTracker::MatchHungrian, 0.3f, 0.5f, 60.0f, 25, 25);
    track_t alpha = 0;
    cv::RNG rng;
    while (k != 27)
    {
        frame = cv::Scalar::all(0);

        // Noise addition (measurements/detections simulation )
        float Xmeasured = pointXY.x + static_cast<float>(rng.gaussian(2.0));
        float Ymeasured = pointXY.y + static_cast<float>(rng.gaussian(2.0));

        // Append circulating around mouse cv::Points (frequently intersecting)
        std::vector<Point_t> pts;
        pts.push_back(Point_t(Xmeasured + 100.0f*sin(-alpha), Ymeasured + 100.0f*cos(-alpha)));
        pts.push_back(Point_t(Xmeasured + 100.0f*sin(alpha), Ymeasured + 100.0f*cos(alpha)));
        pts.push_back(Point_t(Xmeasured + 100.0f*sin(alpha / 2.0f), Ymeasured + 100.0f*cos(alpha / 2.0f)));
        pts.push_back(Point_t(Xmeasured + 100.0f*sin(alpha / 3.0f), Ymeasured + 100.0f*cos(alpha / 1.0f)));
        alpha += 0.05f;

        regions_t regions;
        for (auto p : pts)
        {
            regions.push_back(CRegion(cv::Rect(static_cast<int>(p.x - 1), static_cast<int>(p.y - 1), 3, 3)));
        }


        for (size_t i = 0; i < pts.size(); i++)
        {
            cv::circle(frame, pts[i], 3, cv::Scalar(0, 255, 0), 1, CV_AA);
        }

        tracker.Update(pts, regions, cv::Mat());

        std::cout << tracker.tracks.size() << std::endl;

        for (size_t i = 0; i < tracker.tracks.size(); i++)
        {
            const auto& tracks = tracker.tracks[i];

            if (tracks->trace.size() > 1)
            {
                for (size_t j = 0; j < tracks->trace.size() - 1; j++)
                {
                    cv::line(frame, tracks->trace[j], tracks->trace[j + 1], Colors[i % 6], 2, CV_AA);
                }
            }
        }

        cv::imshow("Video", frame);
        
        if (writer.isOpened())
        {
            writer << frame;
        }

        k = cv::waitKey(10);
    }
#endif

    cv::destroyAllWindows();
    return 0;
}
