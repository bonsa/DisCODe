/*!
 * \file KW_MAP2.hpp
 * \brief Estymacja MAP, uproszczona dłoń
 * \author kwasak
 * \date 2011-04-27
 */

#ifndef KW_MAP2_HPP_
#define KW_MAP2_HPP_

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
namespace KW_MAP2 {

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
 * \class KW_MAP2
 * \brief Example processor class.
 */



class KW_MAP2: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	KW_MAP2(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~KW_MAP2();

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
	Base::EventHandler <KW_MAP2> h_onNewImage;


	/*!
	 * Event handler function.
	 */
	void onNewBlobs();

	/// New set of blobs is waiting
	Base::EventHandler <KW_MAP2> h_onNewBlobs;


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

	void projectionFingertips();

	/*!
	 * Funkcja obracająca punkt p o kąt angle względem układu współrzędnych znajdującego się w punkcie pO
	 */
	cv::Point rot(cv::Point p, double angle, cv::Point p0);

	// Funckja odpowiedzialna za projekcie obserwacji
	void projectionObservation(vector<double> z, int R, int G, int B);

	// Funkcja wyliczajaca wartosci parametrów stanu na podstawie wartosci obserwacji
	void observationToState();

	// Funckja odpowiedzialna za projekcie stanu
	void projectionState(vector<double> s,int R, int G, int B);

	// Funkcja wyliczajaca wartosci parametrów obserwacji na podstawie wartosci obserwacji
	void stateToObservation();

	// Funkcja obliczająca o jaki wektor nalezy zaktualizowac wektor stan
	void calculateDiff();

	// Funckja aktualizująca wektor stanu i macierz kowariancji P
	void updateState();

	// Funkcja sprawdzająca czy warunek końcowy jest sprawdzony
	void stopCondition();


	//*****************************************************************//
	//*FUNKCJE wspolne dla PALCÓW****************************************//
	//*****************************************************************//

	// Otrzymanie obserwacji środkowego palca
	vector <double> getFingerObservation(int i);

	// Funckja odpowiedzialna za projekcie obserwacji palca
	void projectionFingerObservation(vector<double> z, int R, int G, int B);

	vector <double> observationFingerToState(vector <double> z_Finger,float a, float b);

	void projectionFingerState(vector<double> s, int R, int G, int B);

	// Funkcja wyliczajaca wartosci parametrów obserwacji na podstawie wartosci obserwacji
	vector <double> stateFingerToObservation(vector <double> s_Finger, float a);

	void calculateFingerH(vector<double> s_Finger, double H_Finger[5][6], float a);

	vector <double> calculateFingerDiff(vector <double> h_z_Finger, vector <double> z_Finger, double invR_Finger[6][6], double H_Finger[5][6], double P_Finger[5][5]);

	vector <double> updateFingerState(vector <double> diff_Finger, vector <double> s_Finger, double P_Finger[5][5]);
	//*****************************************************************//
	//*FUNKCJE SRODKOWEGO PALCA****************************************//
	//*****************************************************************//

	// Funkcja obliczająca o jaki wektor nalezy zaktualizowac wektor stan
	void calculateMiddleFingerDiff();

	// Funckja aktualizująca wektor stanu i macierz kowariancji P
	void updateMiddleFingerState();



private:

	cv::Mat tsl_img;
	cv::Mat segments;

	bool blobs_ready;
	bool img_ready;

	// wektor stanu dłoni
	vector<int> idFingertips;

	vector<cv::Point> fingertips;

	// czy funkcja jest pierwszy raz uruchomiana
	bool first;

	// największy blob czyli dłoń
	Types::Blobs::BlobResult blobs;

	// kontener przechowujący elementy do rysowania
	Types::DrawableContainer drawcont;

	// wektor obserwacji dłoni
	vector<double> z;

	double MaxY, MinY,  MaxX, MinX;

	//czubek srodkowego palca
	cv::Point topPoint;

	// wektor stanu dłoni
	vector<double> s;

	// poczatkowa hipoteza
	vector<double> s0;

	// wektor stanu dłoni obliczona na podstawie aktualnej obserwacji
	vector<double> sTest;

	// wektor obserwacji dłoni
	vector<double> h_z;

	//macierz H
	double H[5][5];

	//macierz P
	double P[5][5];

	//macierz R
	double R[5][5];

	//macierz odwrotna kowariancji P
	double invP[5][5];

	//macierz odwrotna kowariancji R
	double invR[5][5];

	// różnica punktów charakterystycznych estymacji i punktów aktualnego obrazka
	vector<double> diff;

	//wspołczynnik zapominania
	double factor;

	// funkcja warunek stopu, jesli STOP = true estymacja MAP jest zakończona
	bool STOP;

	//*****************************************************************//
	//*ZMIENNE SRODKOWEGO PALCA****************************************//
	//*****************************************************************//

	// wektor obserwacji dłoni
	vector<double> z_MFinger;

	// wektor stanu dłoni
	vector<double> s_MFinger;

	// poczatkowa hipoteza środkowego palca
	vector<double> s0_MFinger;

	//macierz H
	double H_MFinger[5][6];

	//macierz P
	double P_MFinger[5][5];

	//macierz R
	double R_MFinger[6][6];

	//macierz odwrotna kowariancji P
	double invP_MFinger[5][5];

	//macierz odwrotna kowariancji R
	double invR_MFinger[6][6];

	// wektor obserwacji dłoni
	vector<double> h_z_MFinger;

	// różnica punktów charakterystycznych estymacji i punktów aktualnego obrazka
	vector<double> diff_MFinger;

	// wektor stanu dłoni obliczona na podstawie aktualnej obserwacji
	vector<double> sTest2;

	//*****************************************************************//
	//*ZMIENNE PALCA WSKAZUJĄCEG0**************************************//
	//*****************************************************************//
	//forefinger
	// wektor obserwacji dłoni
	vector<double> z_FFinger;

	// wektor stanu dłoni
	vector<double> s_FFinger;

	// poczatkowa hipoteza środkowego palca
	vector<double> s0_FFinger;

	// wektor stanu dłoni obliczona na podstawie aktualnej obserwacji
	vector<double> sTest3;

	// wektor obserwacji palca
	vector<double> h_z_FFinger;

	//macierz H
	double H_FFinger[5][6];


};

}//: namespace KW_MAP2
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("KW_MAP2", Processors::KW_MAP2::KW_MAP2, Common::Panel_Empty)

#endif /* KW_MAP2_HPP_ */

