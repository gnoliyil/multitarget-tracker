//
// Created by gnoliyil on 17-5-7.
//

#ifndef ADABOOST_H
#define ADABOOST_H

#include <adaboost/StrongClassifier.hpp>
#include <adaboost/KalmanWeakClassifier.hpp>
#include <adaboost/ColorFeature.hpp>
#include <opencv2/imgproc.hpp>

#define BINS 16		  	// number of histogram bins for every dimension
#define CHANNELS 3		// number of histogram dimensions (equal to the number of image channels)

#define NUM_CLASSIFIER 250 // number of weak classifiers considered at each iteration
#define NUM_SELECTOR 50	   // number of weak classifier that compose the strong classifier
#define NUM_DRAW_FEATURES 8 // number of features to draw

typedef adaboost::ColorFeature_<BINS, CHANNELS> _ColorFeature;
typedef adaboost::KalmanWeakClassifier<_ColorFeature> _ColorWeakClassifier;

#endif //ADABOOST_H
