/*!
 * \file KW_Skin_DetectionYCbCr.hpp
 * \brief
 * \author kwasak
 * \date 2010-11-23
 */

#ifndef KW_SKIN_DETECTION_YCBCR_HPP_
#define KW_SKIN_DETECTION_YCBCR_HPP_

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
namespace KW_SkinYCbCr {

using namespace cv;

/*!	
 * \brief KW_Skin_DetectionCbCrY properties
 */
struct Props: public Base::Props
{

	/*!
	 * \copydoc Base::Props::load
	 */
	void load(const ptree & pt)
	{

	}

	/*!
	 * \copydoc Base::Props::save
	 */
	void save(ptree & pt)
	{
	}

protected:

};

/*!
 * \class KW_Skin_DetectionCbCrY
 * \brief Example processor class.
 */
class KW_Skin_DetectionYCbCr: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	KW_Skin_DetectionYCbCr(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~KW_Skin_DetectionYCbCr();

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
	Base::EventHandler <KW_Skin_DetectionYCbCr> h_onNewImage;

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

}//: namespace KW_SkinYCbCr
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("KW_Skin_DetectionYCbCr", Processors::KW_SkinYCbCr::KW_Skin_DetectionYCbCr, Common::Panel_Empty)

#endif /* KW_SKIN_DETECTION_CBCRY_HPP_ */

