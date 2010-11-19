/*!
 * \file KW_Palm_Description.hpp
 * \brief
 * \author kwasak
 * \date 2010-11-19
 */

#ifndef KW_PALM_DESCRIPTION_HPP_
#define KW_PALM_DESCRIPTION_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "Panel_Empty.hpp"
#include "DataStream.hpp"
#include "Props.hpp"

#include <cv.h>
#include <highgui.h>

#include <vector>

#include "Types/Blobs/BlobResult.hpp"
#include "Types/DrawableContainer.hpp"

namespace Processors {
namespace KW_Palm {

using namespace cv;

/*!
 * \brief KW_Palm_Description properties
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
 * \class KW_Palm_Description
 * \brief Example processor class.
 */
class KW_Palm_Description: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	KW_Palm_Description(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~KW_Palm_Description();

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
	Base::EventHandler <KW_Palm_Description> h_onNewImage;


	/*!
	 * Event handler function.
	 */
	void onNewBlobs();

	/// New set of blobs is waiting
	Base::EventHandler <KW_Palm_Description> h_onNewBlobs;


	/// Input blobs
	Base::DataStreamIn <Types::Blobs::BlobResult> in_blobs;

	/// Input binary image
	Base::DataStreamIn <cv::Mat> in_binary;

	/// Event raised, when data is processed
	Base::Event * newImage;

	/*******************************************************************/
	/// Output data stream - list of ellipses around found signs
	Base::DataStreamOut < Types::DrawableContainer > out_dpalm;

	/// Properties
	Props props;

private:
	cv::Mat binary_img;
	cv::Mat segments;

	bool blobs_ready;
	bool binary_ready;

	Types::Blobs::BlobResult blobs;
};

}//: namespace KW_Palm
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("KW_Palm_Description", Processors::KW_Palm::KW_Palm_Description, Common::Panel_Empty)

#endif /* KW_PALM_DESCRIPTION_HPP_ */

