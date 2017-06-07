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
 * StrongClassifier.cpp
 *
 *  Created on: Jul 15, 2011
 *      Author: Filippo Basso
 *
 * If you use part of this code, please cite:
 *    F. Basso, M. Munaro, S. Michieletto and E. Menegatti. Fast and robust multi-people tracking from RGB-D data for a mobile robot.
 *    In Proceedings of the 12th Intelligent Autonomous Systems (IAS) Conference, Jeju Island (Korea), 2012.
 */

#include <adaboost/StrongClassifier.hpp>

namespace adaboost
{

StrongClassifier::StrongClassifier()
{
	// TODO Auto-generated constructor stub
}

StrongClassifier::StrongClassifier(int size, int selectors)
{
	_weakClassifiers.reserve(size);
	_selectors.reserve(selectors);
	for(int i = 0; i < selectors; i++)
	{
		_selectors.push_back(new Selector(size));
	}
}

StrongClassifier::~StrongClassifier()
{
	for(size_t i = 0; i < _weakClassifiers.size(); i++)
	{
		delete _weakClassifiers[i];
	}
	for(size_t i = 0; i < _selectors.size(); i++)
	{
		delete _selectors[i];
	}
}

void StrongClassifier::updateTarget(const cv::Rect& rect)
{
	updateTarget(rect.size());
}

void StrongClassifier::updateTarget(const cv::Size& size)
{
	for(size_t i = 0; i < _weakClassifiers.size(); i++)
	{
		adaboost::WeakClassifier* wc = _weakClassifiers[i];
		wc->adjustSize(size);
	}
}

void StrongClassifier::update(const cv::Rect& target, int response, const cv::Mat& integral)
{
	for(size_t i = 0; i < _weakClassifiers.size(); i++)
	{
		adaboost::WeakClassifier* wc = _weakClassifiers[i];
		wc->train(target, response, integral);
	}

	float lambda = 1.0f;

	for(size_t j = 0; j < _selectors.size(); j++)
	{
		Selector* s = _selectors[j];
		s->selectBest(_weakClassifiers, lambda);
	}

	int max = 0;
	_worst = 0;

	for(size_t i = 0; i < _weakClassifiers.size(); i++)
	{
		adaboost::WeakClassifier* wc = _weakClassifiers[i];
		if(wc->getMarks() > max && !wc->isSelected())
		{
			_worst = i;
			max = wc->getMarks();
		}
		wc->reset();
	}

}

void StrongClassifier::update(const cv::Point& location, int response, const cv::Mat& integral, int numFeaturesToPublish)
{
	for(size_t i = 0; i < _weakClassifiers.size(); i++)
	{
		adaboost::WeakClassifier* wc = _weakClassifiers[i];
		wc->train(location, response, integral);
	}

	float lambda = 1.0f;

	for(size_t j = 0; j < _selectors.size(); j++)
	{
		Selector* s = _selectors[j];
		adaboost::WeakClassifier* b = s->selectBest(_weakClassifiers, lambda);
		if(j < numFeaturesToPublish)
		{
			if(j == 0)
			{
				_bestFeatures.clear();
			}
			std::vector<std::pair<int, int> > v;
			b->getLimits(v);
			std::pair<std::vector<std::pair<int, int> >, float> vPlusWeight;
			vPlusWeight.first = v;
			vPlusWeight.second = s->getWeight();
			_bestFeatures.push_back(vPlusWeight);
		}
	}

	int max = 0;
	_worst = 0;

	for(size_t i = 0; i < _weakClassifiers.size(); i++)
	{
		adaboost::WeakClassifier* wc = _weakClassifiers[i];
		if(wc->getMarks() > max && !wc->isSelected())
		{
			_worst = i;
			max = wc->getMarks();
		}
		wc->reset();
	}
}

std::vector<std::pair<std::vector<std::pair<int, int> >, float> > StrongClassifier::getBestFeatures()
{
	return _bestFeatures;
}

float StrongClassifier::predict(const cv::Rect& target, const cv::Mat& integral)
{
	float sum = 0.0f;
	for(size_t i = 0; i < _selectors.size(); i++)
	{
		Selector* s = _selectors[i];
		sum += s->predict(target, integral); //TODO this method resizes the feature
	}
	return sum/_selectors.size();
}

float StrongClassifier::predict(const cv::Point& location, const cv::Mat& integral)
{
	float sum = 0.0f;
	for(size_t i = 0; i < _selectors.size(); i++)
	{
		Selector* s = _selectors[i];
		sum += s->predict(location, integral);
	}
	return sum/_selectors.size();
}

} /* namespace adaboost */
