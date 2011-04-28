/*!
 * \file KW_MAP2.hpp
 * \brief Estymacja MAP2, uproszczona dłoń, zbieranie próbek
 * \author kwasak
 * \date 2011-04-28
 */

#ifndef KW_MAP2_SAMPLES_HPP_
#define KW_MAP2_SAMPLES_HPP_

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
namespace KW_MAP2_Samples {

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
 * \class KW_MAP2_Samples
 * \brief Example processor class.
 */



class KW_MAP2_Samples: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	KW_MAP2_Samples(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~KW_MAP2_Samples();

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
	Base::EventHandler <KW_MAP2_Samples> h_onNewImage;


	/*!
	 * Event handler function.
	 */
	void onNewBlobs();

	/// New set of blobs is waiting
	Base::EventHandler <KW_MAP2_Samples> h_onNewBlobs;

	/*!
	 * Event handler function.
	 */
	void calculate();

	/// Event handler.
	/// New image is waiting
	Base::EventHandler <KW_MAP2_Samples> h_calculate;

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
	 * Otrzymanie obserwacji dłoni
	 */
	void getObservation();

	/*!
	 * Funkcja obracająca punkt p o kąt angle względem układu współrzędnych znajdującego się w punkcie pO
	 */
	cv::Point rot(cv::Point p, double angle, cv::Point p0);

	// Funkcja wyliczajaca wartosci parametrów stanu na podstawie wartosci obserwacji
	void observationToState();


private:

	cv::Mat tsl_img;
	cv::Mat segments;

	bool blobs_ready;
	bool img_ready;

	// czy funkcja jest pierwszy raz uruchomiana
	bool first;

	// największy blob czyli dłoń
	Types::Blobs::BlobResult blobs;

	// kontener przechowujący elementy do rysowania
	Types::DrawableContainer drawcont;

	// wektor obserwacji dłoni
	vector<double> z;

	//czubek srodkowego palca
	cv::Point topPoint;

	// wektor stanu dłoni
	vector<double> s;

	// wektor obserwacji dłoni
	vector<double> h_z;

	//macierz H
	double H[5][5];

	//macierz przechowująca parametry stanu dla kilku obrazków
	double nStates[5][18];

	//macierz przechowująca punkty char dla kilku obrazków
	double nObservation[5][18];

	// ile obrazków w sekwencji juz task podbrał
	int ileObrazkow;


};

}//: namespace KW_MAP2_Samples
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("KW_MAP2_Samples", Processors::KW_MAP2_Samples::KW_MAP2_Samples, Common::Panel_Empty)

#endif /* KW_MAP2_SAMPLES_HPP_ */

