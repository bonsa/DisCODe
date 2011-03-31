/*!
 * \file KW_PalmDetection.hpp
 * \brief Rozpoznanie, który blob opisuje dłoń
 * \author kwasak
 * \date 2011-03-31
 */

#ifndef KW_PALM_DETECTION_HPP_
#define KW__HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "Panel_Empty.hpp"
#include "DataStream.hpp"
#include "Props.hpp"

#include <iostream>
#include <fstream>
#include <cv.h>
#include <highgui.h>

#include "Types/Blobs/BlobResult.hpp"
#include "Types/DrawableContainer.hpp"

namespace Processors {
namespace KW_Palm {

using namespace cv;
using namespace std;

/*!
 * \brief KW_PalmDetection properties
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
 * \class KW_PalmDetection
 * \brief Example processor class.
 */



class KW_PalmDetection: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	KW_PalmDetection(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~KW_PalmDetection();

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

	/// New image is waiting
	Base::EventHandler <KW_PalmDetection> h_onNewImage;


	/*!
	 * Event handler function.
	 */
	void onNewBlobs();

	/// New set of blobs is waiting
	Base::EventHandler <KW_PalmDetection> h_onNewBlobs;


	/// Input blobs
	Base::DataStreamIn <Types::Blobs::BlobResult> in_blobs;

	/// Input tsl image
	Base::DataStreamIn <cv::Mat> in_tsl;

	/// Event raised, when data is processed
	Base::Event * newImage;


	Base::DataStreamOut < Types::DrawableContainer > out_signs;

	/// Properties
	Props props;

private:

	cv::Mat tsl_img;
	cv::Mat segments;

	bool blobs_ready;
	bool tsl_ready;

	Types::Blobs::BlobResult blobs;

};

}//: namespace KW_Palm
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("KW_PalmDetection", Processors::KW_Palm::KW_PalmDetection, Common::Panel_Empty)

#endif /* KW_PALM_DETECTION_HPP_ */

