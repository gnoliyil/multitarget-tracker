/*
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2013-, Filippo Basso
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * * Neither the name of the copyright holder(s) nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * StrongClassifier.hpp
 *
 *  Created on: Jul 15, 2011
 *      Author: Filippo Basso
 *
 * If you use part of this code, please cite:
 *    F. Basso, M. Munaro, S. Michieletto and E. Menegatti. Fast and robust multi-people tracking from RGB-D data for a mobile robot.
 *    In Proceedings of the 12th Intelligent Autonomous Systems (IAS) Conference, Jeju Island (Korea), 2012.
 */

#ifndef STRONGCLASSIFIER_H_
#define STRONGCLASSIFIER_H_

#include <adaboost/KalmanWeakClassifier.hpp>
#include <adaboost/WeakClassifier.hpp>
#include <adaboost/Selector.hpp>

namespace adaboost
{

class StrongClassifier
{
protected:
	std::vector<WeakClassifier*> _weakClassifiers;
	std::vector<Selector*> _selectors;
	size_t _worst;

	std::vector<std::pair<std::vector<std::pair<int, int> >, float> > _bestFeatures;

public:
	StrongClassifier();
	StrongClassifier(int size, int selectors);
	virtual ~StrongClassifier();

	void updateTarget(const cv::Size& size);
	void updateTarget(const cv::Rect& rect);

	//void update(const std::vector<cv::Point>& locations, const std::vector<int>& responses, const cv::Mat& integral);
	//void update(const std::vector<cv::Rect>& rects, const std::vector<int>& responses, const cv::Mat& integral);
	void update(const cv::Point& location, int response, const cv::Mat& integral, int numFeaturesToPublish);
	void update(const cv::Rect& target, int response, const cv::Mat& integral);
	float predict(const cv::Rect& target, const cv::Mat& integral);
	float predict(const cv::Point& location, const cv::Mat& integral);

	std::vector<std::pair<std::vector<std::pair<int, int> >, float> > getBestFeatures();

	template <class _W>
	void addWeakClassifier(const _W& classifier);

	template <class _W>
	_W* createWeakClassifier(int add = 1);

	template <class _W>
	_W* replaceWorstWeakClassifier();

};

template <class _W>
inline void StrongClassifier::addWeakClassifier(const _W& classifier)
{
	_weakClassifiers.push_back(new _W(classifier));
}

template <class _W>
inline _W* StrongClassifier::createWeakClassifier(int add)
{
	int minFeatureSize = 0;
	bool featureBigEnough = false;
	_W* w;
	while (!featureBigEnough)
	{
		w = new _W();
		if (w->getSize() > minFeatureSize)
			featureBigEnough = true;
	}
	if (add)
		_weakClassifiers.push_back(w);
	return w;
}

template <class _W>
inline _W* StrongClassifier::replaceWorstWeakClassifier()
{
	_W* w = createWeakClassifier<_W>(0);
	delete _weakClassifiers[_worst];
	_weakClassifiers[_worst] = w;
	return w;
}

} /* namespace adaboost */

#endif /* STRONGCLASSIFIER_H_ */
