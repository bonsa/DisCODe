/*!
 * \file KW_SkinDetectionAkademik.hpp
 * \brief
 * \author kwasak
 * \date 2011-03-05
 */

#ifndef KW_SKIN_DETECTION_AKADEMIK_HPP_
#define KW_SKIN_DETECTION_AKADEMIK_HPP_

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
namespace KW_SkinAkademik {

using namespace cv;

/*!
 * \brief KW_Skin_Detection_TSLn properties
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
 * \class KW_SkinDetectionAkademik
 * \brief Example processor class.
 */
class KW_SkinDetectionAkademik: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	KW_SkinDetectionAkademik(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~KW_SkinDetectionAkademik();

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
	Base::EventHandler <KW_SkinDetectionAkademik> h_onNewImage;

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

}//: namespace KW_SkinAkademik
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("KW_SkinDetectionAkademik", Processors::KW_SkinAkademik::KW_SkinDetectionAkademik, Common::Panel_Empty)

#endif /* KW_SKIN_DETECTION_AKADEMIK_HPP_ */

