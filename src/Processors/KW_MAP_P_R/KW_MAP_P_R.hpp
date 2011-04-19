/*!
 * \file KW_MAP.hpp
 * \brief  Estymacja MAP, obliczanie macierzy kowariancji P i R oraz macierze odwrotne  P i R
 * \author kwasak
 * \date 2011-04-14
 */

#ifndef KW_MAP_P_R_HPP_
#define KW_MAP_P_R_HPP_

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
namespace KW_MAP_P_R {

using namespace cv;
using namespace std;

/*!
 * \brief KW_MAP_P_R properties
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
 * \class KW_MAP_P_R
 * \brief Example processor class.
 */



class KW_MAP_P_R: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	KW_MAP_P_R(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~KW_MAP_P_R();

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
	Base::EventHandler <KW_MAP_P_R> h_onNewImage;

	/*!
	 * Event handler function.
	 */
	void onNewBlobs();

	/// New set of blobs is waiting
	Base::EventHandler <KW_MAP_P_R> h_onNewBlobs;

	/*!
	 * Event handler function. wywołuje akcję obliczania P, invP, R, invR
	 */
	void calculate();

	/// Event handler.
	/// New image is waiting
	Base::EventHandler <KW_MAP_P_R> h_calculate;


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
	cv::Point rot(cv::Point p, double angle, cv::Point p0);

	// Funkcja wyznaczająca wektor stanu na podstawie punktów charakterystycznych
	void charPointsToState();

	// Funkcja obliczająca parametry stanu dotyczące palców
	void fingerToState(cv::Point p2, cv::Point p1, int sig);

	// Funkcja wyznaczająca punkty charakterystyczne na podstawie parametrów stanu dotyczących plców
	void stateToFinger(double s1, double s2, double s3, double s4, double angle, int sig);


private:

	cv::Mat tsl_img;
	cv::Mat segments;

	bool blobs_ready;
	bool img_ready;

	// informacja czy jest to pierwsze wywołanie funkcji step
	bool first;

	// blob o największej powierzchni czyli dłoń
	Types::Blobs::BlobResult blobs;

	// kontener elementów do narysowania
	Types::DrawableContainer drawcont;

	// wspołrzędne punktów charakterystycznych konturu
	vector<cv::Point> charPoint;

	// wektor obserwacji dłoni
	vector<cv::Point> z;

	// różnica stanów
	vector<double> diff;

	// wektor stanu dłoni
	vector<double> state;

	// średni wektor stanu
	double pMean[29];

	//macierz przechowująca parametry stanu dla kilku obrazków
	double nStates[29][18];

	//średni wektor parametrów stanu
	vector<double> meanStates;

	// średni wektor obserwacji
	int rMean[20];

	//macierz przechowująca punkty char dla kilku obrazków
	int nChar[20][18];

	//średni wektor punktów charakterystycznych
	vector<int> meanChar;

	// ile obrazków w sekwencji juz task podbrał
	int ileObrazkow;

	//macierz H
	double H[29][20];

	//macierz kowariancji P
	cv:: Mat P;

	//macierz kowariancji R
	double R[20][20];

	//macierz odwrotna R
	cv::Mat invR;

	//macierz odwrotna P
	cv::Mat invP;

	// liczba elementów wetora punktów charakterystycznych
	unsigned int nrChar;

	// liczba elementów wetora stanu
	unsigned int nrStates;
};

}//: namespace KW_MAP_P_R
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("KW_MAP_P_R", Processors::KW_MAP_P_R::KW_MAP_P_R, Common::Panel_Empty)

#endif /* KW_MAP_HPP_ */

