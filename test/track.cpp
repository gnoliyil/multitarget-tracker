#include "opencv2/opencv.hpp"

#include <opencv2/highgui/highgui_c.h>
#include "Ctracker.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>

using namespace std; 
using namespace cv; 

struct Detection {
    Detection () {}
    Detection (int frame_, int id_, int x_, int y_, int w_, int h_, float score_):
        frame(frame_), id(id_), rect(x_, y_, w_, h_), score(score_) 
    {}
    int frame; 
    int id; 
    Rect rect;
    float score; 

    friend istream & operator >> (istream & is, Detection & d)
    {
        int frame_, id_, x_, y_, w_, h_; 
        float score_; 
        is >> frame_ >> id_ >> x_ >> y_ >> w_ >> h_ >> score_; 
        if (x_ < 0) 
        {
            w_ -= -x_; x_ = 0; 
        }
        if (x_ + w_ >= 640)
        {
            w_ -= (x_ + w_ - 640);
            w_ -= 1; 
        }
        if (y_ < 0) 
        {
            h_ -= -y_; y_ = 0; 
        }
        if (y_ + h_ >= 480) 
        {
            h_ -= (y_ + h_ - 480); 
            h_ -= 1; 
        }
        Detection d_(frame_, id_, x_, y_, w_, h_, score_); 
        d = d_; 
        return is; 
    }
};

void readDetections(const string& fileName, vector< vector <Detection> > & dets)
{
    ifstream is(fileName); 
    int curr_frame = -1; 
    Detection det; 
    while (is >> det)
    {
        while (curr_frame < det.frame) {
            dets.push_back(vector <Detection> ()); 
            curr_frame++;
        }
        dets[curr_frame].push_back(det);    
        // cout << det.frame << " " << det.id << endl; 
    }
}

int main(int argc, char** argv)
{
    string detName = argv[1]; 
    string imgPath = argv[2]; 

    Scalar Colors[] = { 
        Scalar(255, 0, 0), 
        Scalar(0, 255, 0), 
        Scalar(0, 0, 255), 
        Scalar(255, 255, 0), 
        Scalar(0, 255, 255), 
        Scalar(255, 255, 255) 
    };
    namedWindow("Video");
    Mat frame;
    bool useLocalTracking = false;
    cout << "1" << endl;

    vector < vector <Detection> > dets; 
    CTracker tracker(useLocalTracking, CTracker::CentersDist, CTracker::FilterCenter, CTracker::MatchHungrian, 0.3f, 0.5f, 90.0f, 90, 25);
    track_t alpha = 0;
    RNG rng;
    cout << "1" << endl;

    readDetections(detName, dets); 
    cout << dets.size() << endl;
    int z; cin >> z;

    for (int time = 0 ; time < dets.size(); time ++ )
    {
        char imgFilename[20], xmlFilename[20];
        sprintf(imgFilename, "frame%04d.jpg", time + 1);  
        sprintf(xmlFilename, "frame%04d.xml", time + 1); 
        cout << (imgPath + "/" + imgFilename) << endl ; 
        frame = imread(imgPath + "/" + imgFilename); 

        vector<Point_t> pts; 
        regions_t regions;
        for (auto p : dets[time])
        {
            regions.push_back(CRegion(p.rect));
            pts.push_back(Point_t(p.rect.x + p.rect.width / 2 , p.rect.y + p.rect.height / 2)); 
        }

        for (size_t i = 0; i < dets[time].size(); i++)
        {
            rectangle(frame, dets[time][i].rect, Scalar(255, 255, 0), 2);
        }

        tracker.Update(pts, regions, frame);

        cout << tracker.tracks.size() << endl;

        for (size_t i = 0; i < tracker.tracks.size(); i++)
        {
            const auto& tracks = tracker.tracks[i];
            const auto trackID = tracks->track_id; 

            printf("track id = %d\n", trackID); 
            if (tracks->trace.size() > 1)
            {
                for (size_t j = 0; j < tracks->trace.size() - 1; j++)
                {
                    line(frame, tracks->trace[j], tracks->trace[j + 1], Colors[trackID % 6], 2, CV_AA);
                }
            }
        }

        imshow("Video", frame);
        imwrite(string("out/") + imgFilename, frame); 

        int k = waitKey(1);
        // if (time % 10 == 0) waitKey(0); 

    }

    destroyAllWindows();
    return 0;
}
