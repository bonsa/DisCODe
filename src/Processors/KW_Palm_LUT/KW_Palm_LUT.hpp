/*!
 * \file KW_Palm_LUT.hpp
 * \brief
 * \author kwasak
 * \date 2010-11-05
 */

#ifndef KW_PALM_LUT_HPP_
#define KW_PALM_LUT_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "Panel_Empty.hpp"
#include "DataStream.hpp"
#include "Props.hpp"

#include <Types/stream_OpenCV.hpp>

#include <cv.h>
#include <highgui.h>

#include <sstream>

namespace Processors {
namespace KW_Palm {

using namespace cv;

/*!
 * \brief KW_PALM_LUT properties
 */
struct Props: public Base::Props
{
	int lambda;
	int features;
	int value;
	cv::Mat mean;
	cv::Mat cov;


	/*!
	 * \copydoc Base::Props::load
	 */
	void load(const ptree & pt)
	{
		lambda = pt.get("lambda", 1);
		value = pt.get("value", 50);
		features = pt.get("features", 2);


		cov = str2mat(cv::Size(features, features), pt.get("cov", ""), 1.0);
		mean = str2mat(cv::Size(features, 1), pt.get("mean", ""), 1.0);

		for (int i = 0; i <mean.size().width; ++i) {
				LOG(LERROR) << "LOAD";
				LOG(LERROR) << "Mean[" << i << "] = " << mean.at<float>(0, i);
			}

			for (int i = 0; i < cov.size().height; ++i)
				for (int j = 0; j < cov.size().width; ++j) {
					LOG(LERROR) << "LOAD";
					LOG(LERROR) << "Covar[" << i << "," << j << "] = " << cov.at<float>(i, j);
			}
	}

	/*!
	 * \copydoc Base::Props::save
	 */
	void save(ptree & pt)
	{
	}

protected:
	cv::Mat str2mat(cv::Size size, std::string s, double norm) {
		std::stringstream ss;
		cv::Mat mat = cv::Mat::eye(size, CV_32F);
		double val;

		ss << s;

		for (int i = 0; i < size.height; ++i) {
			for (int j = 0; j < size.width; ++j) {
				ss >> val;
				val /= norm;
				mat.at<float>(i,j) = val;
			}
		}

		return mat;
	}
};

/*!
 * \class KW_Palm_LUT
 * \brief Example processor class.
 */
class KW_Palm_LUT: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	KW_Palm_LUT(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~KW_Palm_LUT();

	/*!
	 * Return window properties
	 */
	Base::Props * getProperties()
	{
		return &props;
	}

protected:

	/*!
	 * Connects source to given device.
	 */
	bool onInit();

	/*!
	 * Disconnect source from device, closes streams, etc.
	 */
	bool onFinish();

	/*!
	 * Retrieves data from device.
	 */
	bool onStep();

	/*!
	 * Start component
	 */
	bool onStart();

	/*!
	 * Stop component
	 */
	bool onStop();


	/*!
	 * Event handler function.
	 */
	void onNewImage();

	/// Event handler.
	Base::EventHandler <KW_Palm_LUT> h_onNewImage;

	/// Input image
	Base::DataStreamIn <Mat> in_img;


	/// Event raised, when image is processed
	Base::Event * newImage;

	/// Output data stream - skin part
	Base::DataStreamOut <Mat> out_img;


	/// Properties
	Props props;

private:
	cv::Mat skin_img;



};

}//: namespace KW_Palm
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("KW_Palm_LUT", Processors::KW_Palm::KW_Palm_LUT, Common::Panel_Empty)

#endif /* KW_PALM_LUT_HPP_ */

