/*!
 * \file KW_Increase_Contrast.hpp
 * \brief
 * \author kwasak
 * \date 2010-12-17
 */

#ifndef KW_INCREASE_CONTRAST_HPP_
#define KW_INCREASE_CONTRAST_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "Panel_Empty.hpp"
#include "DataStream.hpp"
#include "Props.hpp"

#include <cv.h>
#include <highgui.h>

namespace Processors {
namespace KW_Contrast {

using namespace cv;

/*!
 * \brief KW_mean_skin properties
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
};


struct min_max {
        float maxT;
        float minS;
        float maxS;
};

/*!
 * \class KW_Increase_Contrast
 * \brief Example processor class.
 */
class KW_Increase_Contrast: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	KW_Increase_Contrast(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~KW_Increase_Contrast();

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
	Base::EventHandler <KW_Increase_Contrast> h_onNewImage;

	/// Input image
	Base::DataStreamIn <Mat> in_img;
	Base::DataStreamIn <min_max> in_img2;

	/// Event raised, when image is processed
	Base::Event * newImage;

	/// Output data stream - hue part with continous red
	Base::DataStreamOut <Mat> out_img;


	/// Properties
	Props props;

private:
	cv::Mat TSL_img;



};

}//: namespace KW_Filter
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("KW_Increase_Contrast", Processors::KW_Contrast::KW_Increase_Contrast, Common::Panel_Empty)

#endif /* KW_INCREASE_CONTRAST_HPP_ */

