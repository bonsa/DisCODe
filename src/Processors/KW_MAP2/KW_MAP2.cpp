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

		drawcont.clear();

		z.clear();
		s.clear();
		h_z.clear();

		getObservation();
		projectionObservation(z, 255, 255, 255);
		observationToState();
		projectionState();
		stateToObservation();
		projectionObservation(h_z, 255, 0, 255);




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

		result.AddBlob(blobs.GetBlob(id));
		out_signs.write(result);

	} catch (...) {
		LOG(LERROR) << "KW_MAP::getCharPoints failed\n";

	}
}

//punkcja obracająca punkt p o kąt angle według układu współrzędnych znajdującym się w punkcie p0
cv::Point KW_MAP2::rot(cv::Point p, double angle, cv::Point p0) {
	cv::Point t;
	t.x = p0.x + (int) ((double) (p.x - p0.x) * cos(angle) - (double) (p.y - p0.y) * sin(angle));
	t.y = p0.y + (int) ((double) (p.x - p0.x) * sin(angle) + (double) (p.y - p0.y) * cos(angle));
	return t;
}

void KW_MAP2::projectionObservation(vector<double> z, int R, int G, int B)
{
	cv::Point obsPointA;
	cv::Point obsPointB;
	cv::Point obsPointC;
	cv::Point obsPointD;

	double rotAngle = 0;

	if(z[2]> M_PI_2)
	{
		rotAngle = (z[2] - M_PI_2);
	}
	else if (z[2]< M_PI_2)
	{
		rotAngle = - (M_PI_2 - z[2]);
	}

	obsPointA.x = z[0] - 0.5 * z[4];
	obsPointA.y = z[1] - 4/7.0 *z[3];

	obsPointB.x = z[0] + 0.5 * z[4];
	obsPointB.y = z[1] - 4/7.0 *z[3];

	obsPointC.x = z[0] + 0.5 * z[4];
	obsPointC.y = z[1] + 3/7.0 *z[3];

	obsPointD.x = z[0] - 0.5 * z[4];
	obsPointD.y = z[1] + 3/7.0 *z[3];


	Types::Ellipse * el;
	Types::Line * elL;



	/*
	el = new Types::Ellipse(cv::Point(obsPointA.x, obsPointA.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,0,0));
	drawcont.add(el);

	el = new Types::Ellipse(cv::Point(obsPointB.x, obsPointB.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,0,0));
	drawcont.add(el);

	el = new Types::Ellipse(cv::Point(obsPointC.x, obsPointC.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,0,0));
	drawcont.add(el);

	el = new Types::Ellipse(cv::Point(obsPointD.x, obsPointD.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,0,0));
	drawcont.add(el);

	elL = new Types::Line(cv::Point(obsPointA.x, obsPointA.y), cv::Point(obsPointB.x, obsPointB.y));
	elL->setCol(CV_RGB(0,0,0));
	drawcont.add(elL);

	elL = new Types::Line(cv::Point(obsPointB.x, obsPointB.y), cv::Point(obsPointC.x, obsPointC.y));
	elL->setCol(CV_RGB(0,0,0));
	drawcont.add(elL);

	elL = new Types::Line(cv::Point(obsPointC.x, obsPointC.y), cv::Point(obsPointD.x, obsPointD.y));
	elL->setCol(CV_RGB(0,0,0));

	drawcont.add(elL);
	elL = new Types::Line(cv::Point(obsPointD.x, obsPointD.y), cv::Point(obsPointA.x, obsPointA.y));
	elL->setCol(CV_RGB(0,0,0));
	drawcont.add(elL);

	cv::Point pt1 = rot(topPoint, rotAngle, cv::Point(z[0], z[1]));

	el = new Types::Ellipse(cv::Point(pt1.x, pt1.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,0,0));
	drawcont.add(el);
*/
	obsPointA = rot(obsPointA, - rotAngle, cv::Point(z[0], z[1]));
	obsPointB = rot(obsPointB, - rotAngle, cv::Point(z[0], z[1]));
	obsPointC = rot(obsPointC, - rotAngle, cv::Point(z[0], z[1]));
	obsPointD = rot(obsPointD, - rotAngle, cv::Point(z[0], z[1]));


	el = new Types::Ellipse(cv::Point(obsPointA.x, obsPointA.y), Size2f(10, 10));
	el->setCol(CV_RGB(R,G,B));
	drawcont.add(el);

	el = new Types::Ellipse(cv::Point(obsPointB.x, obsPointB.y), Size2f(10, 10));
	el->setCol(CV_RGB(R,G,B));
	drawcont.add(el);

	el = new Types::Ellipse(cv::Point(obsPointC.x, obsPointC.y), Size2f(10, 10));
	el->setCol(CV_RGB(R,G,B));
	drawcont.add(el);

	el = new Types::Ellipse(cv::Point(obsPointD.x, obsPointD.y), Size2f(10, 10));
	el->setCol(CV_RGB(R,G,B));
	drawcont.add(el);

	elL = new Types::Line(cv::Point(obsPointA.x, obsPointA.y), cv::Point(obsPointB.x, obsPointB.y));
	elL->setCol(CV_RGB(R,G,B));
	drawcont.add(elL);

	elL = new Types::Line(cv::Point(obsPointB.x, obsPointB.y), cv::Point(obsPointC.x, obsPointC.y));
	elL->setCol(CV_RGB(R,G,B));
	drawcont.add(elL);

	elL = new Types::Line(cv::Point(obsPointC.x, obsPointC.y), cv::Point(obsPointD.x, obsPointD.y));
	elL->setCol(CV_RGB(R,G,B));

	drawcont.add(elL);
	elL = new Types::Line(cv::Point(obsPointD.x, obsPointD.y), cv::Point(obsPointA.x, obsPointA.y));
	elL->setCol(CV_RGB(R,G,B));
	drawcont.add(elL);


}


// Funkcja wyliczajaca wartosci parametru stanu na podstawie wartosci obserwacji
void KW_MAP2::observationToState()
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


}

void KW_MAP2::projectionState()
{
	cv::Point obsPointA;
	cv::Point obsPointB;
	cv::Point obsPointC;
	cv::Point obsPointD;

	double rotAngle = 0;

	if(z[2]> M_PI_2)
	{
		rotAngle = (z[2] - M_PI_2);
	}
	else if (z[2]< M_PI_2)
	{
		rotAngle = - (M_PI_2 - z[2]);
	}

	obsPointA.x = s[0] - 0.5 * s[4];
	obsPointA.y = s[1] - 0.5 * s[3];

	obsPointB.x = s[0] + 0.5 * s[4];
	obsPointB.y = s[1] - 0.5 *s[3];

	obsPointC.x = s[0] + 0.5 * s[4];
	obsPointC.y = s[1] + 0.5 *s[3];

	obsPointD.x = s[0] - 0.5 * s[4];
	obsPointD.y = s[1] + 0.5 *s[3];

	Types::Ellipse * el;

	el = new Types::Ellipse(cv::Point(obsPointA.x, obsPointA.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,0,255));
	drawcont.add(el);

	el = new Types::Ellipse(cv::Point(obsPointB.x, obsPointB.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,0,255));
	drawcont.add(el);

	el = new Types::Ellipse(cv::Point(obsPointC.x, obsPointC.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,0,255));
	drawcont.add(el);

	el = new Types::Ellipse(cv::Point(obsPointD.x, obsPointD.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,0,255));
	drawcont.add(el);

	Types::Line * elL;
	elL = new Types::Line(cv::Point(obsPointA.x, obsPointA.y), cv::Point(obsPointB.x, obsPointB.y));
	elL->setCol(CV_RGB(0,0,255));
	drawcont.add(elL);

	elL = new Types::Line(cv::Point(obsPointB.x, obsPointB.y), cv::Point(obsPointC.x, obsPointC.y));
	elL->setCol(CV_RGB(0,0,255));
	drawcont.add(elL);

	elL = new Types::Line(cv::Point(obsPointC.x, obsPointC.y), cv::Point(obsPointD.x, obsPointD.y));
	elL->setCol(CV_RGB(0,0,255));

	drawcont.add(elL);
	elL = new Types::Line(cv::Point(obsPointD.x, obsPointD.y), cv::Point(obsPointA.x, obsPointA.y));
	elL->setCol(CV_RGB(0,0,255));
	drawcont.add(elL);

	obsPointA = rot(obsPointA, - rotAngle, cv::Point(z[0], z[1]));
	obsPointB = rot(obsPointB, - rotAngle, cv::Point(z[0], z[1]));
	obsPointC = rot(obsPointC, - rotAngle, cv::Point(z[0], z[1]));
	obsPointD = rot(obsPointD, - rotAngle, cv::Point(z[0], z[1]));

	el = new Types::Ellipse(cv::Point(obsPointA.x, obsPointA.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,255,255));
	drawcont.add(el);

	el = new Types::Ellipse(cv::Point(obsPointB.x, obsPointB.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,255,255));
	drawcont.add(el);

	el = new Types::Ellipse(cv::Point(obsPointC.x, obsPointC.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,255,255));
	drawcont.add(el);

	el = new Types::Ellipse(cv::Point(obsPointD.x, obsPointD.y), Size2f(10, 10));
	el->setCol(CV_RGB(0,255,255));
	drawcont.add(el);

	elL = new Types::Line(cv::Point(obsPointA.x, obsPointA.y), cv::Point(obsPointB.x, obsPointB.y));
	elL->setCol(CV_RGB(0,255,255));
	drawcont.add(elL);

	elL = new Types::Line(cv::Point(obsPointB.x, obsPointB.y), cv::Point(obsPointC.x, obsPointC.y));
	elL->setCol(CV_RGB(0,255,255));
	drawcont.add(elL);

	elL = new Types::Line(cv::Point(obsPointC.x, obsPointC.y), cv::Point(obsPointD.x, obsPointD.y));
	elL->setCol(CV_RGB(0,255,255));

	drawcont.add(elL);
	elL = new Types::Line(cv::Point(obsPointD.x, obsPointD.y), cv::Point(obsPointA.x, obsPointA.y));
	elL->setCol(CV_RGB(0,255,255));
	drawcont.add(elL);


}

void KW_MAP2:: stateToObservation()
{
	float hz_mx, hz_my, hz_angle, hz_heigth, hz_width;

	hz_mx = s[0] +  0.05 * s[4];
	hz_my = s[1] - 5.0/14.0 * s[3];
	hz_angle = s[2];
	hz_heigth = 5.0/2.0 * s[3];
	hz_width = 2 * s[4];

	h_z.push_back(hz_mx);
	h_z.push_back(hz_my);
	h_z.push_back(hz_angle);
	h_z.push_back(hz_heigth);
	h_z.push_back(hz_width);

	cout<<"h_z\n";
	cout<<h_z[0]<<"\n";
	cout<<h_z[1]<<"\n";
	cout<<h_z[2]<<"\n";
	cout<<h_z[3]<<"\n";
	cout<<h_z[4]<<"\n";
	cout<<"koniec h_z\n";

}

//konstruktor
KW_MAP2::KW_MAP2(const std::string & name) :
	Base::Component(name) {
	LOG(LTRACE) << "Hello KW_MAP\n";
	
	
}

}//: namespace KW_MAP2
}//: namespace Processors
