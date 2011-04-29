/*!
 * \file KW_MAP2_Samples.cpp
 * \brief Estymacja MAP2 - zbieranie probek  
 * \date 2011-04-26
 */

#include <memory>
#include <string>
#include <math.h> 

#include "KW_MAP2_Samples.hpp"
#include "Logger.hpp"
#include "Types/Ellipse.hpp"
#include "Types/Line.hpp"
#include "Types/Rectangle.hpp"
#include <vector>
#include <iomanip>

namespace Processors {
namespace KW_MAP2_Samples {

using namespace cv;

KW_MAP2_Samples::~KW_MAP2_Samples() {
	LOG(LTRACE) << "Good bye KW_MAP2\n";

	ileObrazkow = 0;
}

bool KW_MAP2_Samples::onInit() {
	LOG(LTRACE) << "KW_MAP2::initialize\n";

	h_onNewImage.setup(this, &KW_MAP2_Samples::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	h_onNewBlobs.setup(this, &KW_MAP2_Samples::onNewBlobs);
	registerHandler("onNewBlobs", &h_onNewBlobs);

	h_calculate.setup(this, &KW_MAP2_Samples::calculate);
	registerHandler("calculate", &h_calculate);


	registerStream("in_blobs", &in_blobs);
	registerStream("in_img", &in_img);

	newImage = registerEvent("newImage");

	registerStream("out_signs", &out_signs);
	registerStream("out_draw", &out_draw);


	//pierwsze uruchomienie komponentu
	first = true;

	return true;
}

bool KW_MAP2_Samples::onFinish() {
	LOG(LTRACE) << "KW_MAP2_Samples::finish\n";

	return true;
}

bool KW_MAP2_Samples::onStep() {
	LOG(LTRACE) << "KW_MAP2_Samples::step\n";

	blobs_ready = img_ready = false;

	try {

		ileObrazkow = ileObrazkow + 1;
		cout<<"ilosc obrazkow"<<ileObrazkow<<"\n" ;

		drawcont.clear();
		z.clear();
		s.clear();
		h_z.clear();

		getObservation();
		observationToState();

		out_draw.write(drawcont);
		newImage->raise();

		return true;
	} catch (...) {
		LOG(LERROR) << "KW_MAP2_Samples::getCharPoints failed\n";
		return false;
	}
}


bool KW_MAP2_Samples::onStop() {
	return true;
}

bool KW_MAP2_Samples::onStart() {
	return true;
}

void KW_MAP2_Samples::onNewImage() {
	LOG(LTRACE) << "KW_MAP2_Samples::onNewImage\n";

	img_ready = true;
	tsl_img = in_img.read();
	//co robi tak linijka?
	tsl_img = tsl_img.clone();
	if (blobs_ready && img_ready)
		onStep();
}

void KW_MAP2_Samples::onNewBlobs() {
	LOG(LTRACE) << "KW_MAP2_Samples::onNewBlobs\n";

	blobs_ready = true;
	blobs = in_blobs.read();
	if (blobs_ready && img_ready)
		onStep();
}

void KW_MAP2_Samples::getObservation(){

	LOG(LTRACE) << "KW_MAP2_Samples::getCharPoints\n";

	try {

		// id największego bloba, czyli dłoni
		int id = 0;
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
		// obrocony actualPoint o kat nachylania dłoni wzgledem układu wspolrzędnych w punkcie środa masy
		cv::Point tempPoint;
		// wektor zawierający punkty konturu
		vector<cv::Point> contourPoints;

		Types::DrawableContainer signs;

		//momenty goemetryczne potrzebne do obliczenia środka ciężkości
		double m00, m10, m01;
		//powierzchnia bloba, powiedzchnia największego bloba, współrzędne środka ciężkości, maksymalna wartośc współrzędnej Y
		double Area, MaxArea, CenterOfGravity_x, CenterOfGravity_y, MaxY, MinY,  MaxX, MinX;

		double height, width;
		MaxArea = 0;
		MaxY = 0;
		MaxX = 0;
		MinY = 1000000000.0;
		MinX = 1000000000.0;

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
		//current Blob przychowuje największego bloba, czyli dłoni
		currentBlob = blobs.GetBlob(id);


		// calculate moments
		m00 = currentBlob->Moment(0, 0);
		m01 = currentBlob->Moment(0, 1);
		m10 = currentBlob->Moment(1, 0);

		//obliczenie środka cięzkości
		CenterOfGravity_x = m10 / m00;
		CenterOfGravity_y = m01 / m00;

		z.push_back(CenterOfGravity_x);
		z.push_back(CenterOfGravity_y);

		Types::Ellipse * elE;
		elE = new Types::Ellipse(Point2f(CenterOfGravity_x, CenterOfGravity_y), Size2f(10, 10));
		elE->setCol(CV_RGB(0,255,0));
		drawcont.add(elE);


		//kontur największego bloba
		contour = currentBlob->GetExternalContour()->GetContourPoints();
		cvStartReadSeq(contour, &reader);
		CV_READ_SEQ_ELEM( actualPoint, reader);
		topPoint = actualPoint;

		double dx = - z[0] + topPoint.x;
		double dy = - z[1] + topPoint.y;

		Types::Ellipse * el;
		el = new Types::Ellipse(Point2f(topPoint.x, topPoint.y), Size2f(10, 10));
		el->setCol(CV_RGB(0,0,0));
		drawcont.add(el);
		//argument kąta nachylenia
		double angle = abs(atan2(dy, dx));

		z.push_back(angle);

		MinX = currentBlob->MinX();
		MaxX = currentBlob->MaxX();
		MinY = currentBlob->MinY();
		MaxY = currentBlob->MaxY();

		height = MaxY - MinY;
		width = MaxX - MinX;

		z.push_back(height);
		z.push_back(width);

		nObservation[0][ileObrazkow -1] = CenterOfGravity_x;
		nObservation[1][ileObrazkow -1] = CenterOfGravity_y;
		nObservation[2][ileObrazkow -1] = angle * 180 / M_PI;
		nObservation[3][ileObrazkow -1] = height;
		nObservation[4][ileObrazkow -1] = width;

		result.AddBlob(blobs.GetBlob(id));
		out_signs.write(result);

	} catch (...) {
		LOG(LERROR) << "KW_MAP::getCharPoints failed\n";

	}
}

//punkcja obracająca punkt p o kąt angle według układu współrzędnych znajdującym się w punkcie p0
cv::Point KW_MAP2_Samples::rot(cv::Point p, double angle, cv::Point p0) {
	cv::Point t;
	t.x = p0.x + (int) ((double) (p.x - p0.x) * cos(angle) - (double) (p.y - p0.y) * sin(angle));
	t.y = p0.y + (int) ((double) (p.x - p0.x) * sin(angle) + (double) (p.y - p0.y) * cos(angle));
	return t;
}


// Funkcja wyliczajaca wartosci parametru stanu na podstawie wartosci obserwacji
void KW_MAP2_Samples::observationToState()
{
	float s_mx, s_my, s_angle, s_heigth, s_width;

	s_mx = z[0] - 0.025 * z[4];
	s_my = z[1] + 1.0/7.0 * z[3];

	Types::Ellipse * el;

	el = new Types::Ellipse(cv::Point(s_mx, s_my), Size2f(10, 10));
	el->setCol(CV_RGB(255,0,0));
	drawcont.add(el);

	s_angle = z[2];
	s_heigth = 0.4 * z[3];
	s_width = 0.5 * z[4];

	s.push_back(s_mx);
	s.push_back(s_my);
	s.push_back(s_angle);
	s.push_back(s_heigth);
	s.push_back(s_width);

	nStates[0][ileObrazkow -1] = s_mx;
	nStates[1][ileObrazkow -1] = s_my;
	nStates[2][ileObrazkow -1] = s_angle * 180 / M_PI;
	nStates[3][ileObrazkow -1] = s_heigth;
	nStates[4][ileObrazkow -1] = s_width;


}

void KW_MAP2_Samples::calculate()
{
	//zapisywanie macierzy P, invP, R, invR do pliku, tworzenie pliku
	std::ofstream plik("/home/kasia/Test.txt");

	plik<<"\n ";
	plik<<"RSamples\n ";

	plik<<"R = [\n ";
	for (int i = 0; i< 5; i++)
	{
		for(int j = 0; j< ileObrazkow; j++)
		{
			plik<<setprecision(5)<<nObservation[i][j]<<" \t";
		}
		plik<<";";
	}
	plik<<"]";

	plik<<"\n ";
	plik<<"PSamples\n ";

	plik<<"P = [\n ";
	for (int i = 0; i< 5; i++)
	{
		for(int j = 0; j< ileObrazkow; j++)
		{
			plik<<setprecision(5)<<nStates[i][j]<<" \t";
		}
		plik<<";";
	}
	plik<<"]\n\n";
	
	for (unsigned int i = 0 ; i < 5; i++)
	{
		for (int j = 0; j < ileObrazkow; j++)
		{
			meanStates[i] += nStates[i][j];
		}
		meanStates[i] /= ileObrazkow;
		plik<<"meanStates["<<i<<"] = "<<meanStates[i]<<";\n";
	}

	plik.close();
}
//konstruktor
KW_MAP2_Samples::KW_MAP2_Samples(const std::string & name) :
	Base::Component(name) {
	LOG(LTRACE) << "Hello KW_MAP2_Samples\n";


}

}//: namespace KW_MAP2_Samples
}//: namespace Processors
