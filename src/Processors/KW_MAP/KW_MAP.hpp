/*!
 * \file KW_MAP.hpp
 * \brief Estymacja MAP, bez uśredniania wykresy odległości miedzy punktami konturu a przesuniętym śr. ciężkości
 * \author kwasak
 * \date 2011-04-03
 */

#ifndef KW_MAP_HPP_
#define KW_MAP_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "Panel_Empty.hpp"
#include "DataStream.hpp"
#include "Props.hpp"

#include <iostream>
#include <fstream>
#include <cv.h>
#include <highgui.h>

#include <vector>
#include "Types/Blobs/BlobResult.hpp"
#include "Types/DrawableContainer.hpp"

namespace Processors {
namespace KW_MAP {

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
 * \class KW_MAP
 * \brief Example processor class.
 */



class KW_MAP: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	KW_MAP(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~KW_MAP();

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
	Base::EventHandler <KW_MAP> h_onNewImage;


	/*!
	 * Event handler function.
	 */
	void onNewBlobs();

	/// New set of blobs is waiting
	Base::EventHandler <KW_MAP> h_onNewBlobs;


	/// Input blobs
	Base::DataStreamIn <Types::Blobs::BlobResult> in_blobs;

	/// Input tsl image
	Base::DataStreamIn <cv::Mat> in_img;

	/// Event raised, when data is processed
	Base::Event * newImage;

	/// Output data stream - list of ellipses around found signs
	Base::DataStreamOut < Types::Blobs::BlobResult > out_signs;

	Base::DataStreamOut < Types::DrawableContainer > out_draw;

	/// Properties
	Props props;


	/*!
	 * Otrzymanie punktów na konturze dłoni
	 */
	void getCharPoints();

	/*!
	 * Funkcja obracająca punkt p o kąt angle względem układu współrzędnych znajdującego się w punkcie pO
	 */
	CvPoint rot(CvPoint p, double angle, CvPoint p0);


	void charPointsToState();

	void fingerToStateLeft(CvPoint p2, CvPoint p1);
	void fingerToStateRigt();


private:

	cv::Mat tsl_img;
	cv::Mat segments;

	bool blobs_ready;
	bool img_ready;


	Types::Blobs::BlobResult blobs;

	double last_x, last_y;

	Types::DrawableContainer drawcont;

	// wspołrzędne punktów charakterystycznych konturu
	vector<CvPoint> charPoint;

	// wektor obserwacji dłoni
	vector<int> z;
	// wektor stanu dłoni
	vector<int> state;
};

}//: namespace KW_MAP
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("KW_MAP", Processors::KW_MAP::KW_MAP, Common::Panel_Empty)

#endif /* KW_MAP_HPP_ */

