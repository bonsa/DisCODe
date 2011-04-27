/*!
 * \file KW_MAP.cpp
 * \brief Estymacja MAP dla uproszczonej dłoni 
 * \author kwasak
 * \date 2011-04-27
 */

#include <memory>
#include <string>
#include <math.h> 

#include "KW_MAP2.hpp"
#include "Logger.hpp"
#include "Types/Ellipse.hpp"
#include "Types/Line.hpp"
#include "Types/Rectangle.hpp"
#include <vector>
#include <iomanip>

namespace Processors {
namespace KW_MAP2 {

using namespace cv;

KW_MAP2::~KW_MAP2() {
	LOG(LTRACE) << "Good bye KW_MAP2\n";
}

bool KW_MAP2::onInit() {
	LOG(LTRACE) << "KW_MAP2::initialize\n";

	h_onNewImage.setup(this, &KW_MAP2::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	h_onNewBlobs.setup(this, &KW_MAP2::onNewBlobs);
	registerHandler("onNewBlobs", &h_onNewBlobs);

	registerStream("in_blobs", &in_blobs);
	registerStream("in_img", &in_img);

	newImage = registerEvent("newImage");

	registerStream("out_signs", &out_signs);
	registerStream("out_draw", &out_draw);

	factor = 0.01;
	nrChar = 20;
	nrStates = 29;

	// czy warunek stopu jest spełniony
	STOP = false;

	//pierwsze uruchomienie komponentu
	first = true;

	return true;
}

bool KW_MAP2::onFinish() {
	LOG(LTRACE) << "KW_MAP2::finish\n";

	return true;
}

bool KW_MAP2::onStep() {
	LOG(LTRACE) << "KW_MAP2::step\n";

	blobs_ready = img_ready = false;

	try {
		ileObrazkow = ileObrazkow + 1;
		drawcont.clear();

		if(STOP == false)
		{
			z.clear();
			charPoint.clear();
			diff.clear();
			state.clear();
			T.clear();
			diffStates.clear();

			getObservation();
		//	projectionMeasurePoints();
/*
			if (first == true)
			{
				// char --> s, z pomiarów oblicza stan
				charPointsToState();
				first = false;
			}
			else
			{
				// s --> z
				stateToCharPoint();
				projectionEstimatedPoints();
				calculateDiff();
				updateState();
			}

			projectionStates();
			stopCondition();
*/
		}
		else
		{
	//		projectionStates();
		}

		out_draw.write(drawcont);
		newImage->raise();

		return true;
	} catch (...) {
		LOG(LERROR) << "KW_MAP::getCharPoints failed\n";
		return false;
	}
}

bool KW_MAP2::onStop() {
	return true;
}

bool KW_MAP2::onStart() {
	return true;
}

void KW_MAP2::onNewImage() {
	LOG(LTRACE) << "KW_MAP::onNewImage\n";

	img_ready = true;
	tsl_img = in_img.read();
	//co robi tak linijka?
	tsl_img = tsl_img.clone();
	if (blobs_ready && img_ready)
		onStep();
}

void KW_MAP2::onNewBlobs() {
	LOG(LTRACE) << "KW_MAP::onNewBlobs\n";

	blobs_ready = true;
	blobs = in_blobs.read();
	if (blobs_ready && img_ready)
		onStep();
}

void KW_MAP2::getObservation(){

	LOG(LTRACE) << "KW_MAP::getCharPoints\n";

	try {

		// id największego bloba, czyli dłoni
		int id = 0;
		//numerElements - liczba punktów wchodzących w skład konturu
		unsigned int numerElements;
		// aktualnie pobrany blob
		Types::Blobs::Blob *currentBlob;
		// wynikowy blob
		Types::Blobs::BlobResult result;
		// punkty znajdujace sie na konturze
		CvSeq * contour;
		// czyta elementy na konturu
		CvSeqReader reader;
		// punkt, na którym aktualnie jest wykonywana operacja
		cv::Point actualPoint;
		// wektor zawierający punkty konturu
		vector<cv::Point> contourPoints;
		// wektor odległości między punktami konturu a przesuniętym środkiem ciężkości
		vector<float> dist;
		//usredniony (wygładzony) wektor odległości między punktami konturu a przesuniętym środkiem ciężkości
		vector<float> meanDist;
		//wektor czastowych pochodnych wektor odległości między punktami konturu a przesuniętym środkiem ciężkości
		vector<float> derivative;
		//zmienna pomocnicza
		float TempDist;
		//zapamietuje poprzedni znak różnicy miedzy punktami,
		//1- funkcja jest rosnoca, -1 - funkcja malejąca
		int lastSign;
		int lastMinDist;
		//idenksy punktów charakterystycznych;
		vector<int> indexPoint;
		//powyzej tej odległości od środa cieżkosci moga znajdować sie ekstrema
		int MINDIST;
		//id ostatenio wyznaczonego ekstremum
		int idLastExtreme;
		//kontener przechowujący elementy, które mozna narysować
		Types::DrawableContainer signs;

		//momenty goemetryczne potrzebne do obliczenia środka ciężkości
		double m00, m10, m01, m11, m20, m02;
		//powierzchnia bloba, powiedzchnia największego bloba, współrzędne środka ciężkości, maksymalna wartośc współrzędnej Y
		double Area, MaxArea, CenterOfGravity_x, CenterOfGravity_y, MaxY, MinY,  MaxX, MinX;

		double length, width;
		MaxArea = 0;
		MaxY = 0;
		MaxX = 0;
		MinY = 100000000000000;
		MinX = 100000000000000;

		//największy blob to dłoń
		for (int i = 0; i < blobs.GetNumBlobs(); i++) {
			currentBlob = blobs.GetBlob(i);

			Area = currentBlob->Area();
			//szukanie bloba o największej powierzchni
			if (Area > MaxArea) {
				MaxArea = Area;
				// id największego bloba, czyli dłoni
				id = i;
			}
		}


		// calculate moments
		m00 = currentBlob->Moment(0, 0);
		m01 = currentBlob->Moment(0, 1);
		m10 = currentBlob->Moment(1, 0);

		//obliczenie środka cięzkości
		CenterOfGravity_x = m10 / m00;
		CenterOfGravity_y = m01 / m00;

		z.push_back(CenterOfGravity_x);
		z.push_back(CenterOfGravity_x);

		m11 = currentBlob->Moment(1, 1);
		m02 = currentBlob->Moment(0, 2);
		m20 = currentBlob->Moment(2, 0);

		double alfa = atan2(2*m11,(m20 - m02));
		alfa /= 2;

		z.push_back(alfa);

		//current Blob przychowuje największego bloba, czyli dłoni
		currentBlob = blobs.GetBlob(id);
		//kontur największego bloba
		contour = currentBlob->GetExternalContour()->GetContourPoints();
		cvStartReadSeq(contour, &reader);

		for (int j = 0; j < contour->total; j = j + 1)
		{
			CV_READ_SEQ_ELEM( actualPoint, reader);

			if (j % 10 == 1)
			{
				//wpisanie punktów z konturu do wektora
				contourPoints.push_back(cv::Point(actualPoint.x, actualPoint.y));
				//szukanie max i min wartości y
				if (actualPoint.y > MaxY)
				{
					MaxY = actualPoint.y;
				}
				else if (actualPoint.y < MaxY)
				{
					MinY = actualPoint.y;
				}
				//szukanie max i min wartości x
				if (actualPoint.x > MaxX)
				{
					MaxX = actualPoint.x;
				}
				else if (actualPoint.x < MaxX)
				{
					MinX = actualPoint.x;
				}
			}
		}

		//najpierw nalezy wyprostowac, Potem odejmujemy
		length = MaxY - MinY;
		width = MaxX - MinX;

		z.push_back(length);
		z.push_back(width);

		/******


		MINDIST = (MaxY - CenterOfGravity_y) * (MaxY - CenterOfGravity_y) * 4
				/ 9;
		//przesuniety punkt środka ciężkości
		charPoint.push_back(cv::Point(CenterOfGravity_x, CenterOfGravity_y
				+ (MaxY - CenterOfGravity_y) * 4 / 5));

		//środek cieżkości przesuwał troche w dół ekranu, aby ułatwić wyznaczanie punktóe charakterystycznych
		CenterOfGravity_y += (MaxY - CenterOfGravity_y) * 2 / 3;

		//liczba punktów wchodząca w skład konturu
		numerElements = contourPoints.size();

		//******************************************************************
		//obliczenie roznicy miedzy punktami konturu a przesuniętym środkiem ciężkosci
		for (unsigned int i = 0; i < numerElements; i++) {
			TempDist = (contourPoints[i].x - CenterOfGravity_x)
					* (contourPoints[i].x - CenterOfGravity_x)
					+ (contourPoints[i].y - CenterOfGravity_y)
							* (contourPoints[i].y - CenterOfGravity_y);
			if (TempDist > MINDIST)
				dist.push_back(TempDist);
			else
				//jeśli odległość jest mniejsza niż MINDIST oznacza to, że jest to dolna cześć dłoni i nie znajdują się tam żadnego punkty charakterystyczne poza przesuniętym środkiem ciężkości, dlatego te punkty można ominąć
				dist.push_back(MINDIST);
		}

		//******************************************************************
		//obliczenie pochodnej, szukanie ekstremów
		derivative.push_back(dist[1] - dist[0]);
		if (derivative[0] > 0)
			lastSign = 1;
		else
			lastSign = -1;

		//1 -oznacza, że ostatni element z konturu należał do dolnej czesci dłoni
		lastMinDist = 0;
		idLastExtreme = 0;
		//pierwszy punkt kontury to wierzchołek punktu środkowego.
		indexPoint.push_back(0);

		for (unsigned int i = 1; i < numerElements - 2; i++) {

			//różnica miedzy sąsiedznimi punktami
			derivative.push_back(dist[i + 1] - dist[i]);

			if (dist[i + 1] > MINDIST && dist[i] > MINDIST) {
				//jeżeli ostatnio był wykryta dolna cześci dłoni, następnym charakterystycznych punktem powinien być czubek palca, dlatego lastSign = 1;
				if (lastMinDist == 1) {
					lastSign = 1;
					lastMinDist = 0;
				}
				//maksiumum - czubek palca, funkcja rosła i zaczeła maleć
				if (derivative[i] < 0 && lastSign == 1) {
					if (((contourPoints[i].x - contourPoints[idLastExtreme].x)
							* (contourPoints[i].x
									- contourPoints[idLastExtreme].x)
							+ (contourPoints[i].y
									- contourPoints[idLastExtreme].y)
									* (contourPoints[i].y
											- contourPoints[idLastExtreme].y))
							> 900) {
						indexPoint.push_back(i);
						lastSign = -1;
						idLastExtreme = i;
					}
				}
				//minimum - punkt między palcami
				else if (derivative[i] > 0 && lastSign == -1) {
					if (((contourPoints[i].x - contourPoints[idLastExtreme].x)
							* (contourPoints[i].x
									- contourPoints[idLastExtreme].x)
							+ (contourPoints[i].y
									- contourPoints[idLastExtreme].y)
									* (contourPoints[i].y
											- contourPoints[idLastExtreme].y))
							> 900) {
						indexPoint.push_back(i);
						lastSign = 1;
						idLastExtreme = i;
					}
				}
			} else {
				// element należący do dołu dłoni
				lastMinDist = 1;
			}
		}

		int idLeftPoint = 0;
		int xLeftPoint = 1000000;
		for (unsigned int i = 0; i < indexPoint.size(); i++) {
			//znajdujemy punkt najbardziej wysynięty na lewo, czyli wierzchołek małego palca
			if (xLeftPoint > contourPoints[indexPoint[i]].x) {
				xLeftPoint = contourPoints[indexPoint[i]].x;
				idLeftPoint = i;
			}
		}

		for (int i = idLeftPoint; i >= 0; i--) {
			//wpisanie do tablicy punktów charakterystycznych punktów opisujących trzy lewe palce
			charPoint.push_back(cv::Point(contourPoints[indexPoint[i]].x,
					contourPoints[indexPoint[i]].y));
		}

		for (int i = indexPoint.size() - 1; i > idLeftPoint; i--) {
			//wpisanie do tablicy punktów charakterystycznych punktów opisujących dwa prawe palce
			charPoint.push_back(cv::Point(contourPoints[indexPoint[i]].x,
					contourPoints[indexPoint[i]].y));
		}
		*/

		result.AddBlob(blobs.GetBlob(id));
		out_signs.write(result);

	} catch (...) {
		LOG(LERROR) << "KW_MAP::getCharPoints failed\n";

	}
}


//konstruktor
KW_MAP2::KW_MAP2(const std::string & name) :
	Base::Component(name) {
	LOG(LTRACE) << "Hello KW_MAP\n";
	ileObrazkow = 0;
	
	
}

}//: namespace KW_MAP2
}//: namespace Processors
