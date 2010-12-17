/*!
 * \file KW_Initial_Filter.hpp
 * \brief
 * \author kwasak
 * \date 2010-12-17
 */

#ifndef KW_INILIAL_FILTER_HPP_
#define KW_INITIAL_FILTER_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "Panel_Empty.hpp"
#include "DataStream.hpp"
#include "Props.hpp"

#include <cv.h>
#include <highgui.h>

namespace Processors {
namespace KW_Filter {

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

/*!
 * \class KW_Initial_Filter
 * \brief Example processor class.
 */
class KW_Initial_Filter: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	KW_Initial_Filter(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~KW_Initial_Filter();

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
	Base::EventHandler <KW_Initial_Filter> h_onNewImage;

	/// Input image
	Base::DataStreamIn <Mat> in_img;

	/// Event raised, when image is processed
	Base::Event * newImage;

	/// Output data stream - hue part with continous red
	Base::DataStreamOut <Mat> out_img;


	/// Properties
	Props props;

private:
	cv::Mat RGB_img;
	cv::Mat Filtered_img;

	int k;
};

}//: namespace KW_Filter
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("KW_Initial_Filter", Processors::KW_Filter::KW_Initial_Filter, Common::Panel_Empty)

#endif /* KW_INITIAL_FILTER_HPP_ */

