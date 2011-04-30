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
		h_z.clear();
		diff.clear();
		sTest.clear();

		getObservation();
		projectionObservation(z, 255, 255, 255);
		observationToState();
		projectionState(sTest, 255, 0, 0);
		projectionState(s, 0, 255, 255);
		stateToObservation();
		projectionObservation(h_z, 255, 0, 255);
		calculateDiff();
		updateState();

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

		z.push_back(angle * 180/ M_PI);

		cout<<"angle"<<angle * 180/ M_PI<<"\n";
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
	if((z[2] * M_PI/180 )> M_PI_2)
	{
		rotAngle = ((z[2] * M_PI/180) - M_PI_2);
	}
	else if ((z[2] * M_PI / 180)< M_PI_2)
	{
		rotAngle = - (M_PI_2 - (z[2] * M_PI / 180));
	}
	/*
	if(z[2]> M_PI_2)
	{
		rotAngle = (z[2] - M_PI_2);
	}
	else if (z[2]< M_PI_2)
	{
		rotAngle = - (M_PI_2 - z[2]);
	}
*/
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

	s_mx = z[0] - 0.05 * z[4];
//	s_mx = z[0] - 0.07 * z[4];
//	s_my = z[1] + 1.5/7.0 * z[3];
	s_my = z[1] + 1.0/7.0 * z[3];

	Types::Ellipse * el;

	el = new Types::Ellipse(cv::Point(s_mx, s_my), Size2f(10, 10));
	el->setCol(CV_RGB(255,0,0));
	drawcont.add(el);

	s_angle = z[2];
	s_heigth = 0.4 * z[3];
	s_width = 0.5 * z[4];
	//s_width = 0.55 * z[4];

	sTest.push_back(s_mx);
	sTest.push_back(s_my);
	sTest.push_back(s_angle);
	sTest.push_back(s_heigth);
	sTest.push_back(s_width);


}

void KW_MAP2::projectionState(vector<double> s, int R, int G, int B)
{
	cv::Point obsPointA;
	cv::Point obsPointB;
	cv::Point obsPointC;
	cv::Point obsPointD;

	double rotAngle = 0;

	if((s[2] * M_PI / 180)> M_PI_2)
	{
		rotAngle = ((s[2] * M_PI / 180) - M_PI_2);
	}
	else if ((s[2] * M_PI / 180)< M_PI_2)
	{
		rotAngle = - (M_PI_2 - (s[2] * M_PI / 180));
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
	Types::Line * elL;

/*
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

void KW_MAP2:: stateToObservation()
{
	float hz_mx, hz_my, hz_angle, hz_heigth, hz_width;

	hz_mx = s[0] +  0.1 * s[4];
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



// Funkcja obliczająca o jaki wektor nalezy zaktualizowac wektor stan
void KW_MAP2::calculateDiff()
{
	//różnicaiedzy wektorami h(s) i z
	double D[5];
	float error2 = 0;

	for (unsigned int i = 0; i < 5; i ++)
	{
		//różnica miedzy punktami charakterystycznymi aktualnego obraz
		D[i] =  h_z[i] - z[i];
		cout<<"\nD"<<D[i];
	}

	double t1[5];
	for (unsigned int i = 0; i < 5; i++) {
		t1[i] = 0;
		for (unsigned int j = 0; j < 5; j++) {
			//t = iloraz odwrotnej macierzy R * roznica D
			t1[i] += invR[i][j] * D[j];
		}
	}

	double t2[5];
	for (unsigned int i = 0; i < 5; i++) {
		t2[i] = 0;
		for (unsigned int j = 0; j < 5; j++) {
			//t1 = iloraz macierzy H * t1
			t2[i] += H[i][j] * t1[j];
		}
	//	cout<<t1[i]<<"\n";
	}

	double t3[5];
	for (unsigned int i = 0; i < 5; i++) {
		t3[i] = 0;
		for (unsigned int j = 0; j < 5; j++) {
			//mnożenie macierzy P * t2
			t3[i] +=   P[i][j] * t2[j];

		}
		t3[i] *= 0.1;//*factor;
		diff.push_back(t3[i]);
		error2 += abs(t3[i]);

	}
	cout <<"ERROR2"<<error2<<"\n";
}

void KW_MAP2::updateState()
{
	for (unsigned int i = 0; i < 5; i++) {
		cout << i << " diff\t" << diff[i] << "\n";
	}

	for (unsigned int i = 0; i < 5; i++) {
		s[i] = s[i] - diff[i];
		cout << i << " states\t" << s[i] << "\n";
	}

	for (unsigned int i = 0; i < 5; i++) {
		for (unsigned int j = 0; j < 5; j++) {
			P[i][j] *= (1 - factor);
		}
	}
}


//konstruktor
KW_MAP2::KW_MAP2(const std::string & name) :
	Base::Component(name) {
	LOG(LTRACE) << "Hello KW_MAP\n";


	/*
	 * dane do MAP2
	
	factor = 0.001;

	s.push_back(197.68);
	s.push_back(398.36);
	s.push_back(96.86);
	s.push_back(136.24);
	s.push_back(176.81);

	for(unsigned int i = 0; i<5; i++)
	{
		cout << i << " states\t" << s[i] << "\n";

	}

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			H[i][j] = 0;
		}
	}

	H[0][0] = 1;
	H[4][0] = 0.05;
	H[1][1] = 1;
	H[3][1] = -5.0/14.0;
	H[2][2] = 1;
	H[3][3] = 5.0/3.0;
	H[4][4] = 2;

	P[0][0] = 46.990883;
	P[0][1] = 1.528119;
	P[0][2] = -5.323601;
	P[0][3] = 3.192531;
	P[0][4] = -3.358225;

	P[1][0] = 1.528119;
	P[1][1] = 0.514513;
	P[1][2] = -0.282289;
	P[1][3] = 0.419617;
	P[1][4] = 0.028410;

	P[2][0] = -5.323601;
	P[2][1] = -0.282289;
	P[2][2] = 0.662295;
	P[2][3] = -0.413586;
	P[2][4] = 0.308156;

	P[3][0] = 3.192531;
	P[3][1] = 0.419617;
	P[3][2] = -0.413586;
	P[3][3] = 0.482469;
	P[3][4] = -0.141358;

	P[4][0] = -3.358225;
	P[4][1] = 0.028410;
	P[4][2] = 0.308156;
	P[4][3] = -0.141358;
	P[4][4] = 0.976080;


	invP[0][0] = 0.842304;
	invP[0][1] = 3.529979;
	invP[0][2] = 5.934017;
	invP[0][3] = -3.432465;
	invP[0][4] = 0.424710;

	invP[1][0] = 3.529979;
	invP[1][1] = 23.353079;
	invP[1][2] = 23.376121;
	invP[1][3] = -23.427403;
	invP[1][4] = 0.692416;

	invP[2][0] = 5.934017;
	invP[2][1] = 23.376121;
	invP[2][2] = 45.764166;
	invP[2][3] = -19.650916;
	invP[2][4] = 2.441733;

	invP[3][0] = -3.432465;
	invP[3][1] = -23.427403;
	invP[3][2] = -19.650916;
	invP[3][3] = 28.063895;
	invP[3][4] = -0.859359;

	invP[4][0] = 0.424710;
	invP[4][1] = 0.692416;
	invP[4][2] = 2.441733;
	invP[4][3] = -0.859359;
	invP[4][4] = 1.570248;


	R[0][0] = 49.392222;
	R[0][1] = 0.405527;
	R[0][2] = -5.620102;
	R[0][3] = 8.432222;
	R[0][4] = -7.008954;

	R[1][0] = 0.405527;
	R[1][1] = 0.293047;
	R[1][2] = -0.141200;
	R[1][3] = 0.655098;
	R[1][4] = 0.166275;

	R[2][0] = -5.620102;
	R[2][1] = -0.141200;
	R[2][2] = 0.701253;
	R[2][3] = -1.094788;
	R[2][4] = 0.652565;

	R[3][0] = 8.432222;
	R[3][1] = 0.655098;
	R[3][2] = -1.094788;
	R[3][3] = 3.192810;
	R[3][4] = -0.748366;

	R[4][0] = -7.008954;
	R[4][1] = 0.166275;
	R[4][2] = 0.652565;
	R[4][3] = -0.748366;
	R[4][4] = 4.133987;


	invR[0][0] = 0.798964;
	invR[0][1] = 3.350462;
	invR[0][2] = 5.609231;
	invR[0][3] = -0.831034;
	invR[0][4] = 0.183962;

	invR[1][0] = 3.350462;
	invR[1][1] = 22.143922;
	invR[1][2] = 22.068010;
	invR[1][3] = -5.763473;
	invR[1][4] = 0.263006;

	invR[2][0] = 5.609231;
	invR[2][1] = 22.068010;
	invR[2][2] = 43.135090;
	invR[2][3] = -4.309000;
	invR[2][4] = 1.033462;

	invR[3][0] = -0.831034;
	invR[3][1] = -5.763473;
	invR[3][2] = -4.309000;
	invR[3][3] = 2.189405;
	invR[3][4] = -0.100624;

	invR[4][0] = 0.183962;
	invR[4][1] = 0.263006;
	invR[4][2] = 1.033462;
	invR[4][3] = -0.100624;

*/

	//MAP3

	factor = 0.0001;

	s.push_back(352.18);
	s.push_back(331.46);
	s.push_back(96.532);
	s.push_back(152.78);
	s.push_back(178.57);

	for(unsigned int i = 0; i<5; i++)
	{
		cout << i << " states\t" << s[i] << "\n";

	}

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			H[i][j] = 0;
		}
	}

	H[0][0] = 1;
	H[4][0] = 0.05;
	H[1][1] = 1;
	H[3][1] = -5.0/14.0;
	H[2][2] = 1;
	H[3][3] = 5.0/3.0;
	H[4][4] = 2;


	P[0][0] = 7898.123517;
	P[0][1] = 37.669557;
	P[0][2] = 16.654725;
	P[0][3] = -354.473747;
	P[0][4] = -470.330553;

	P[1][0] = 37.669557;
	P[1][1] = 262.201038;
	P[1][2] = 27.000436;
	P[1][3] = 3.523747;
	P[1][4] = 71.038316;

	P[2][0] = 16.654725;
	P[2][1] = 27.000436;
	P[2][2] = 12.361747;
	P[2][3] = 0.812833;
	P[2][4] = 16.639193;

	P[3][0] = -354.473747;
	P[3][1] = 3.523747;
	P[3][2] = 0.812833;
	P[3][3] = 176.328000;
	P[3][4] = 274.791053;

	P[4][0] = -470.330553;
	P[4][1] = 71.038316;
	P[4][2] = 16.639193;
	P[4][3] = 274.791053;
	P[4][4] = 508.217763;


	invP[0][0] = 0.000141;
	invP[0][1] = 0.000020;
	invP[0][2] = -0.000068;
	invP[0][3] = 0.000514;
	invP[0][4] = -0.000148;

	invP[1][0] = 0.000020;
	invP[1][1] = 0.005368;
	invP[1][2] = -0.008217;
	invP[1][3] = 0.004397;
	invP[1][4] = -0.002840;

	invP[2][0] = -0.000068;
	invP[2][1] = -0.008217;
	invP[2][2] = 0.118934;
	invP[2][3] = 0.024500;
	invP[2][4] = -0.016055;

	invP[3][0] = 0.000514;
	invP[3][1] = 0.004397;
	invP[3][2] = 0.024500;
	invP[3][3] = 0.050643;
	invP[3][4] = -0.028324;

	invP[4][0] = -0.000148;
	invP[4][1] = -0.002840;
	invP[4][2] = -0.016055;
	invP[4][3] = -0.028324;
	invP[4][4] = 0.018068;

	R[0][0] = 7852.336300;
	R[0][1] = 162.862329;
	R[0][2] = 17.485060;
	R[0][3] = -851.867079;
	R[0][4] = -889.894553;

	R[1][0] = 162.862329;
	R[1][1] = 282.185173;
	R[1][2] = 26.707663;
	R[1][3] = -148.634526;
	R[1][4] = -54.218000;

	R[2][0] = 17.485060;
	R[2][1] = 26.707663;
	R[2][2] = 12.361747;
	R[2][3] = 2.032082;
	R[2][4] = 33.278387;

	R[3][0] = -851.867079;
	R[3][1] = -148.634526;
	R[3][2] = 2.032082;
	R[3][3] = 1102.050000;
	R[3][4] = 1373.955263;

	R[4][0] = -889.894553;
	R[4][1] = -54.218000;
	R[4][2] = 33.278387;
	R[4][3] = 1373.955263;
	R[4][4] = 2032.871053;


	invR[0][0] = 0.000141;
	invR[0][1] = 0.000020;
	invR[0][2] = -0.000068;
	invR[0][3] = 0.000208;
	invR[0][4] = -0.000077;

	invR[1][0] = 0.000020;
	invR[1][1] = 0.005367;
	invR[1][2] = -0.008215;
	invR[1][3] = 0.002525;
	invR[1][4] = -0.001420;

	invR[2][0] = -0.000068;
	invR[2][1] = -0.008215;
	invR[2][2] = 0.118929;
	invR[2][3] = 0.008627;
	invR[2][4] = -0.008026;

	invR[3][0] = 0.000208;
	invR[3][1] = 0.002525;
	invR[3][2] = 0.008627;
	invR[3][3] = 0.008715;
	invR[3][4] = -0.005873;

	invR[4][0] = -0.000077;
	invR[4][1] = -0.001420;
	invR[4][2] = -0.008026;
	invR[4][3] = -0.005873;
	invR[4][4] = 0.004521;

}

}//: namespace KW_MAP2
}//: namespace Processors
