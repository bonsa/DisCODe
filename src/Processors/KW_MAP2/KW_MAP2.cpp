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


		//czubki palców
		fingertips.clear();
		idFingertips.clear();

		//srodkowy palec
		z_MFinger.clear();
		h_z_MFinger.clear();
		diff_MFinger.clear();
		sTest2.clear();
	//	s_MFinger.clear();

		//palec wskazujacy
		z_FFinger.clear();
		sTest3.clear();

		if(STOP == false)
		{
			z.clear();
			h_z.clear();
			diff.clear();
			sTest.clear();

			getObservation();
			projectionFingertips();
		//	projectionObservation(z, 255, 255, 255);
			observationToState();
			projectionState(sTest, 0, 255, 255);
		//	projectionState(s, 255, 255, 255);
			stateToObservation();
		//	projectionObservation(h_z, 255, 0, 255);
			calculateDiff();
			updateState();
		//	stopCondition();
		}
		else
		{
			//projectionState(s, 0, 255, 255);
		}

		//palec środkowy
		sTest.clear();
		getMiddleFingerObservation();
		projectionFingerObservation(z_MFinger, 200, 200, 200);
		observationMiddleFingerToState();
		projectionFingerState(sTest2, 0, 255, 255);
	//	projectionFingerState(s_MFinger, 255, 255, 255);
	//	stateMiddleFingerToObservation(sTest2);
		stateMiddleFingerToObservation(s_MFinger);
	//	projectionFingerObservation(h_z_MFinger, 255, 0, 0);
		calculateMiddleFingerH();
		calculateMiddleFingerDiff();
		updateMiddleFingerState();

		//palec wskazujacy
		getForeFingerObservation();
		projectionFingerObservation(z_FFinger, 200, 200, 200);
		observationForeFingerToState();
		projectionFingerState(sTest3, 0, 255, 255);



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
		//zmienna pomocnicza
		float TempDist;
		//zapamietuje poprzedni znak różnicy miedzy punktami,
		//1- funkcja jest rosnoca, -1 - funkcja malejąca
		int lastSign;
		int lastMinDist;
		//idenksy punktów charakterystycznych;
		//powyzej tej odległości od środa cieżkosci moga znajdować sie ekstrema
		int MINDIST;
		//id ostatenio wyznaczonego ekstremum
		int idLastExtreme;

		Types::DrawableContainer signs;

		//momenty goemetryczne potrzebne do obliczenia środka ciężkości
		double m00, m10, m01;
		//powierzchnia bloba, powiedzchnia największego bloba, współrzędne środka ciężkości, maksymalna wartośc współrzędnej Y
		double Area, MaxArea, CenterOfGravity_x, CenterOfGravity_y;

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

		MinX = currentBlob->MinX();
		MaxX = currentBlob->MaxX();
		MinY = currentBlob->MinY();
		MaxY = currentBlob->MaxY();

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
		idFingertips.push_back(0);

		for (int j = 0; j < contour->total; j = j + 1) {

			CV_READ_SEQ_ELEM( actualPoint, reader);

			if (j % 20 == 1)
			{
				//wpisanie punktów z konturu do wektora
				contourPoints.push_back(cv::Point(actualPoint.x, actualPoint.y));
			}
		}

		MINDIST = (MaxY - CenterOfGravity_y) * (MaxY - CenterOfGravity_y) * 4/ 9;

		//środek cieżkości przesuwał troche w dół ekranu, aby ułatwić wyznaczanie punktów charakterystycznych
		CenterOfGravity_y += (MaxY - CenterOfGravity_y) * 2 / 3;


		//******************************************************************
		//obliczenie roznicy miedzy punktami konturu a przesuniętym środkiem ciężkosci
		vector<double> dist;
		//1 -oznacza, że ostatni element z konturu należał do dolnej czesci dłoni
		lastMinDist = 0;
		idLastExtreme = 0;
		lastSign = 0; //ta linijka nic nie znaczy, nie chce miec warninga
		for (unsigned int i = 0; i < contourPoints.size(); i++)
		{
			TempDist = (contourPoints[i].x - CenterOfGravity_x)	* (contourPoints[i].x - CenterOfGravity_x) + (contourPoints[i].y - CenterOfGravity_y) * (contourPoints[i].y - CenterOfGravity_y);

			if (TempDist > MINDIST)
				dist.push_back(TempDist);
			else
				//jeśli odległość jest mniejsza niż MINDIST oznacza to, że jest to dolna cześć dłoni i nie znajdują się tam żadnego punkty charakterystyczne poza przesuniętym środkiem ciężkości, dlatego te punkty można ominąć
				dist.push_back(MINDIST);

			if (i == 1)
			{
				if ((dist[1] - dist[0]) > 0)
					lastSign = 1;
				else
					lastSign = -1;
			}
			else if (i > 1)
			{
				if (dist[i] > MINDIST && dist[i - 1] > MINDIST)
				{

					//jeżeli ostatnio była wykryta dolna cześci dłoni, następnym charakterystycznych punktem powinien być czubek palca, dlatego lastSign = 1;
					if (lastMinDist == 1)
					{
						lastSign = 1;
						lastMinDist = 0;
					}
					//maksiumum - czubek palca, funkcja rosła i zaczeła maleć
					if ((dist[i] - dist[i-1]) < 0 && lastSign == 1)
					{
						if (((contourPoints[i-1].x - contourPoints[idLastExtreme].x) * (contourPoints[i-1].x - contourPoints[idLastExtreme].x) + (contourPoints[i-1].y - contourPoints[idLastExtreme].y) * (contourPoints[i-1].y - contourPoints[idLastExtreme].y)) > 5000)
						{
							cout<<"!!!!!!!!!!!!"<<dist[i]<<"\n";
							if((dist[i-1]>20000) && (idFingertips.size()<5))
							{
								idFingertips.push_back(i-1);
								lastSign = -1;
								idLastExtreme = i-1;
							}
						}
					}
					//minimum - punkt między palcami
					else if ((dist[i] - dist[i-1]) > 0 && lastSign == -1)
					{
						if (((contourPoints[i-1].x - contourPoints[idLastExtreme].x) * (contourPoints[i-1].x - contourPoints[idLastExtreme].x) + (contourPoints[i-1].y - contourPoints[idLastExtreme].y) * (contourPoints[i-1].y - contourPoints[idLastExtreme].y)) > 5000)
						{
							lastSign = 1;
							idLastExtreme = i-1;
						}
					}
				}
				else
				{
					// element należący do dołu dłoni
					lastMinDist = 1;
				}
			}

		}

		int idLeftPoint = 0;
		int xLeftPoint = 1000000;
		for (unsigned int i = 0; i < idFingertips.size(); i++) {
			//znajdujemy punkt najbardziej wysynięty na lewo, czyli wierzchołek małego palca
			if (xLeftPoint > contourPoints[idFingertips[i]].x)
			{
				xLeftPoint = contourPoints[idFingertips[i]].x;
				idLeftPoint = i;
			}
		}

		for (int i = idLeftPoint; i >= 0; i--) {
			//wpisanie do tablicy punktów charakterystycznych punktów opisujących trzy lewe palce
			fingertips.push_back(cv::Point( contourPoints[idFingertips[i]].x, contourPoints[idFingertips[i]].y));
		}

		for (int i = idFingertips.size() - 1; i > idLeftPoint; i--) {
			//wpisanie do tablicy punktów charakterystycznych punktów opisujących dwa prawe palce
			fingertips.push_back(cv::Point( contourPoints[idFingertips[i]].x,  contourPoints[idFingertips[i]].y));
		}


		double dx = - z[0] + topPoint.x;
		double dy = - z[1] + topPoint.y;

		Types::Ellipse * el;
		el = new Types::Ellipse(Point2f(topPoint.x, topPoint.y), Size2f(10, 10));
		el->setCol(CV_RGB(0,0,0));
		drawcont.add(el);
		//argument kąta nachylenia
		double angle = abs(atan2(dy, dx));

		//kat w radianach
		z.push_back(angle);

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

//projekcja na obraz punktów charakterystycznych z aktualnego obrazka
void KW_MAP2::projectionFingertips() {
//	LOG(LTRACE) << "KW_MAP::projectionMeasurePoints\n";

	Types::Ellipse * el;
	for (unsigned int i = 0; i < fingertips.size(); i++) {
		cout<<"SIZE"<<fingertips.size()<<"\n";
		el = new Types::Ellipse(cv::Point(fingertips[i].x, fingertips[i].y),Size2f(10, 10));
		el->setCol(CV_RGB(0,0,255));
		drawcont.add(el);
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
/*	if((z[2] * M_PI/180 )> M_PI_2)
	{
		rotAngle = ((z[2] * M_PI/180) - M_PI_2);
	}
	else if ((z[2] * M_PI / 180)< M_PI_2)
	{
		rotAngle = - (M_PI_2 - (z[2] * M_PI / 180));
	}
	*/

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



	Types::Line * elL;

	/*
	 *
	Types::Ellipse * el;
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

/*
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
*/

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
	s_my = z[1] + 1.0/7.0 * z[3];

	Types::Ellipse * el;

	el = new Types::Ellipse(cv::Point(s_mx, s_my), Size2f(10, 10));
	el->setCol(CV_RGB(255,0,0));
	drawcont.add(el);

	s_angle = z[2];
	s_heigth = 0.4 * z[3];
	s_width = 0.5 * z[4];

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
	if(s[2]> M_PI_2)
	{
		rotAngle = (s[2] - M_PI_2);
	}
	else if (s[2]< M_PI_2)
	{
		rotAngle = - (M_PI_2 - s[2]);
	}


/*	if((s[2] * M_PI / 180)> M_PI_2)
	{
		rotAngle = ((s[2] * M_PI / 180) - M_PI_2);
	}
	else if ((s[2] * M_PI / 180)< M_PI_2)
	{
		rotAngle = - (M_PI_2 - (s[2] * M_PI / 180));
	}
*/
	obsPointA.x = s[0] - 0.5 * s[4];
	obsPointA.y = s[1] - 0.5 * s[3];

	obsPointB.x = s[0] + 0.5 * s[4];
	obsPointB.y = s[1] - 0.5 *s[3];

	obsPointC.x = s[0] + 0.5 * s[4];
	obsPointC.y = s[1] + 0.5 *s[3];

	obsPointD.x = s[0] - 0.5 * s[4];
	obsPointD.y = s[1] + 0.5 *s[3];

	Types::Line * elL;

	obsPointA = rot(obsPointA, - rotAngle, cv::Point(s[0], s[1]));
	obsPointB = rot(obsPointB, - rotAngle, cv::Point(s[0], s[1]));
	obsPointC = rot(obsPointC, - rotAngle, cv::Point(s[0], s[1]));
	obsPointD = rot(obsPointD, - rotAngle, cv::Point(s[0], s[1]));

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
	//	cout<<"\nD"<<D[i];
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
	//cout <<"ERROR2"<<error2<<"\n";
}

void KW_MAP2::updateState()
{
	for (unsigned int i = 0; i < 5; i++) {
	//	cout << i << " diff\t" << diff[i] << "\n";
	}

	for (unsigned int i = 0; i < 5; i++) {
		s[i] = s[i] - diff[i];
//		cout << i << " states\t" << s[i] << "\n";
	}

	for (unsigned int i = 0; i < 5; i++) {
		for (unsigned int j = 0; j < 5; j++) {
			P[i][j] *= (1 - factor);
		}
	}
}

void KW_MAP2::stopCondition()
{
	// różnica stanów estymacji i hipoteza poczatkowa (średniego wektora cech stanu)
	vector<double> diffS;

	//wektor błędu, jeśli suma jego elementów jest mniejsza niż określony warunek stopu nastepuje koniec estymacji
	vector<double> tempError;

	for (unsigned int i = 0; i < 5; i++)
	{
		diffS.push_back(s[i] - s0[i]);
	}

	double t1[5];
	for (unsigned int i = 0; i < 5; i++)
	{
		t1[i] = 0;
		for (unsigned int j = 0; j < 5; j++) {
			//mnożenie macierzy odwrotnej P * roznica S
			t1[i] += invP[i][j] * diffS[j];
		}
	}

	double error = 0;
	for (unsigned int i = 0; i < 5; i++)
	{
		tempError.push_back(diff[i]+t1[i]);
		error += abs(tempError[i]);
	}

	//warunek końca estymacji MAP
//	cout<<"\nERROR!!!!!: "<<error<<"\n";
	if(error < 1.3)
	{
		cout<< "STOP is true\n";
		STOP = true;
	}

}

//*****************************************************************//
//*SRODKOWY PALEC**************************************************//
//*****************************************************************//

// Otrzymanie obserwacji środkowego palca
void KW_MAP2::getMiddleFingerObservation()
{

	double downX, downY, topX, topY, alfa, w;

	downX = z[0] - 3.0/7.0 * (topPoint.x - z[0]);
	downY = z[1] - 3.0/7.0 * (topPoint.y - z[1]);

	Types::Ellipse * el;

	el = new Types::Ellipse(cv::Point(z[0], z[1]), Size2f(10, 10));
	el->setCol(CV_RGB(255,255,255));
	drawcont.add(el);

	topX = topPoint.x;
	topY = topPoint.y;

	//z jest w stopniach, a alfa ma byc w radianach
	alfa = z[2];  //kat w radianach
	w = z[4];

	z_MFinger.push_back(downX);
	z_MFinger.push_back(downY);
	z_MFinger.push_back(topX);
	z_MFinger.push_back(topY);
	z_MFinger.push_back(alfa);
	z_MFinger.push_back(w);

}

void KW_MAP2::projectionFingerObservation(vector<double> z, int R, int G, int B)
{
	cv::Point obsPointDown = cv::Point(z[0], z[1]);
	cv::Point obsPointTop  = cv::Point(z[2], z[3]);

		Types::Ellipse * el;
		Types::Line * elL;

		el = new Types::Ellipse(cv::Point(obsPointDown.x, obsPointDown.y), Size2f(10, 10));
		el->setCol(CV_RGB(R,G,B));
		drawcont.add(el);

		el = new Types::Ellipse(cv::Point(obsPointTop.x, obsPointTop.y), Size2f(10, 10));
		el->setCol(CV_RGB(R,G,B));
		drawcont.add(el);

		elL = new Types::Line(cv::Point(obsPointDown.x, obsPointDown.y), cv::Point(obsPointTop.x, obsPointTop.y));
		elL->setCol(CV_RGB(R,G,B));
		drawcont.add(elL);

		drawcont.add(elL);
}

void KW_MAP2:: observationMiddleFingerToState()
{

	float s_mx, s_my, s_angle, s_heigth, s_width;

	s_mx = z_MFinger[0] + 0.7 * (z_MFinger[2] - z_MFinger[0]);
	s_my = z_MFinger[1] + 0.7 * (z_MFinger[3] - z_MFinger[1]);

	Types::Ellipse * el;

	el = new Types::Ellipse(cv::Point(s_mx, s_my ), Size2f(10, 10));
	el->setCol(CV_RGB(0,255,255));
	drawcont.add(el);

	s_angle = z_MFinger[4];
	s_heigth = 0.6 * (sqrt((z_MFinger[0]-z_MFinger[2])*(z_MFinger[0]-z_MFinger[2])+(z_MFinger[1]-z_MFinger[3])*(z_MFinger[1]-z_MFinger[3])));
	s_width = 0.12 * z_MFinger[5];

	sTest2.push_back(s_mx);
	sTest2.push_back(s_my);
	sTest2.push_back(s_angle);
	sTest2.push_back(s_heigth);
	sTest2.push_back(s_width);

}



void KW_MAP2::projectionFingerState(vector<double> s, int R, int G, int B)
{
	cv::Point obsPointA;
	cv::Point obsPointB;
	cv::Point obsPointC;
	cv::Point obsPointD;

	double rotAngle = 0;

	//katy sa w radianach

	if((s[2])> M_PI_2)
	{
		rotAngle = ((s[2]) - M_PI_2);
	}
	else if ((s[2])< M_PI_2)
	{
		rotAngle = - (M_PI_2 - (s[2]));
	}


	/*
	if((s[2] * M_PI / 180)> M_PI_2)
	{
		rotAngle = ((s[2] * M_PI / 180) - M_PI_2);
	}
	else if ((s[2] * M_PI / 180)< M_PI_2)
	{
		rotAngle = - (M_PI_2 - (s[2] * M_PI / 180));
	}
	*/

	obsPointA.x = s[0] - 0.5 * s[4];
	obsPointA.y = s[1] - 0.5 * s[3];

	obsPointB.x = s[0] + 0.5 * s[4];
	obsPointB.y = s[1] - 0.5 *s[3];

	obsPointC.x = s[0] + 0.5 * s[4];
	obsPointC.y = s[1] + 0.5 *s[3];

	obsPointD.x = s[0] - 0.5 * s[4];
	obsPointD.y = s[1] + 0.5 *s[3];


	Types::Line * elL;

	obsPointA = rot(obsPointA, - rotAngle, cv::Point(s[0], s[1]));
	obsPointB = rot(obsPointB, - rotAngle, cv::Point(s[0], s[1]));
	obsPointC = rot(obsPointC, - rotAngle, cv::Point(s[0], s[1]));
	obsPointD = rot(obsPointD, - rotAngle, cv::Point(s[0], s[1]));

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
// Funkcja wyliczajaca wartosci parametrów obserwacji na podstawie wartosci obserwacji
void KW_MAP2::stateMiddleFingerToObservation(vector <double> s_MFinger)
{
	float hz_downX, hz_downY, hz_topX, hz_topY, hz_angle, hz_width;

	hz_downX = s_MFinger[0] - 7.0/6.0 * s_MFinger[3]*cos(s_MFinger[2]);
	hz_downY = s_MFinger[1] + 7.0/6.0 * s_MFinger[3]*sin(s_MFinger[2]);

	hz_topX = s_MFinger[0] + 1.0/2.0 * s_MFinger[3]*cos(s_MFinger[2]);
	hz_topY = s_MFinger[1] - 1.0/2.0 * s_MFinger[3]*sin(s_MFinger[2]);

	hz_angle = s_MFinger[2];
	hz_width = 25/3.0 * s_MFinger[4];

	h_z_MFinger.push_back(hz_downX);
	h_z_MFinger.push_back(hz_downY);
	h_z_MFinger.push_back(hz_topX);
	h_z_MFinger.push_back(hz_topY);
	h_z_MFinger.push_back(hz_angle);
	h_z_MFinger.push_back(hz_width);

}

void KW_MAP2::calculateMiddleFingerH()
{

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			H_MFinger[i][j] = 0;
		}
	}

	H_MFinger[0][0] = 1;
	H_MFinger[2][0] = 7.0/6.0 * s_MFinger[3] * sin(s_MFinger[2]);
	H_MFinger[3][0] = - 7.0/6.0 * cos(s_MFinger[2]);

	H_MFinger[1][1] = 1;
	H_MFinger[2][1] = 7.0/6.0 * s_MFinger[3] * cos(s_MFinger[2]);
	H_MFinger[3][1] = 7.0/6.0 * sin(s_MFinger[2]);

	H_MFinger[0][2] = 1;
	H_MFinger[2][2] = - 1.0/2.0 * s_MFinger[3] * sin(s_MFinger[2]);
	H_MFinger[3][2] = 1.0/2.0 * cos(s_MFinger[2]);

	H_MFinger[1][3] = 1;
	H_MFinger[2][3] = - 1.0/2.0 * s_MFinger[3] * cos(s_MFinger[2]);
	H_MFinger[3][3] = - 1.0/2.0 * sin(s_MFinger[2]);

	H_MFinger[2][4] = 1.0;
	H_MFinger[4][5] = 25/3.0;
}

void  KW_MAP2::calculateMiddleFingerDiff()
{
	//różnicaiedzy wektorami h(s) i z
	double D[6];
	float error3 = 0;

	for (unsigned int i = 0; i < 6; i ++)
	{
		//różnica miedzy punktami charakterystycznymi aktualnego obraz
		D[i] =  h_z_MFinger[i] - z_MFinger[i];
		cout<<"D"<<i<<" = "<<D[i]<<"\n";
	}

	double t1[6];
	for (unsigned int i = 0; i < 6; i++) {
		t1[i] = 0;
		for (unsigned int j = 0; j < 6; j++) {
			//t = iloraz odwrotnej macierzy R * roznica D
			t1[i] += invR_MFinger[i][j] * D[j];
		}
	}

	double t2[5];
	for (unsigned int i = 0; i < 5; i++) {
		t2[i] = 0;
		for (unsigned int j = 0; j < 6; j++) {
			//t1 = iloraz macierzy H * t1
			t2[i] += H_MFinger[i][j] * t1[j];
		}
	//	cout<<t1[i]<<"\n";
	}

	double t3[5];
	for (unsigned int i = 0; i < 5; i++) {
		t3[i] = 0;
		for (unsigned int j = 0; j < 5; j++) {
			//mnożenie macierzy P * t2
			t3[i] +=  P_MFinger[i][j] * t2[j];

		}
		t3[i] *= 0.1;//*factor;
		diff_MFinger.push_back(t3[i]);
		error3 += abs(t3[i]);

	}
	cout <<"\nERROR M Finger"<<error3<<"\n";
}

// Funckja aktualizująca wektor stanu i macierz kowariancji P
void KW_MAP2::updateMiddleFingerState()
{

	for (unsigned int i = 0; i < 5; i++) {
		cout << i << " diff_MFinger\t" << diff_MFinger[i] << "\n";
	}

	for (unsigned int i = 0; i < 5; i++) {
		s_MFinger[i] = s_MFinger[i] - diff_MFinger[i];
		cout << i << " states\t" << s_MFinger[i] << "\n";
	}

	for (unsigned int i = 0; i < 5; i++) {
		for (unsigned int j = 0; j < 5; j++) {
			P_MFinger[i][j] *= (1 - factor);
		}
	}
}

//*****************************************************************//
//*FUNKCJE PALCA WSKAZUJĄCEGO**************************************//
//*****************************************************************//

// Otrzymanie obserwacji środkowego palca
void KW_MAP2::getForeFingerObservation()
{

	double downX, downY, topX, topY, alfa, w;

	downX = z[0] - 3.0/7.0 * (topPoint.x - z[0]);
	downY = z[1] - 3.0/7.0 * (topPoint.y - z[1]);

	Types::Ellipse * el;

	el = new Types::Ellipse(cv::Point(downX, downY), Size2f(10, 10));
	el->setCol(CV_RGB(255,0,255));
	drawcont.add(el);

	topX = fingertips[3].x;
	topY = fingertips[3].y;

	el = new Types::Ellipse(cv::Point(topX, topY), Size2f(10, 10));
	el->setCol(CV_RGB(255,0,255));
	drawcont.add(el);

	//z jest w stopniach, a alfa ma byc w radianach

	double dx, dy;
	dx = topX - downX;
	dy = topY - downY;

	double angle = abs(atan2(dy, dx));
	alfa = angle;  //kat w radianach
	w = z[4];

	z_FFinger.push_back(downX);
	z_FFinger.push_back(downY);
	z_FFinger.push_back(topX);
	z_FFinger.push_back(topY);
	z_FFinger.push_back(alfa);
	z_FFinger.push_back(w);
}

// Funkcja wyliczajaca wartosci parametrów stanu palca wskazujacego na podstawie wartosci obserwacji
void KW_MAP2::observationForeFingerToState()
{
	float s_mx, s_my, s_angle, s_heigth, s_width;

	s_mx = z_FFinger[0] + 0.72 * (z_FFinger[2] - z_FFinger[0]);
	s_my = z_FFinger[1] + 0.72 * (z_FFinger[3] - z_FFinger[1]);

	Types::Ellipse * el;

	el = new Types::Ellipse(cv::Point(s_mx, s_my ), Size2f(10, 10));
	el->setCol(CV_RGB(0,255,255));
	drawcont.add(el);

	s_angle = z_FFinger[4];
	s_heigth = 0.56 * (sqrt((z_FFinger[0]-z_FFinger[2])*(z_FFinger[0]-z_FFinger[2])+(z_FFinger[1]-z_FFinger[3])*(z_FFinger[1]-z_FFinger[3])));
	s_width = 0.12 * z_FFinger[5];

	sTest3.push_back(s_mx);
	sTest3.push_back(s_my);
	sTest3.push_back(s_angle);
	sTest3.push_back(s_heigth);
	sTest3.push_back(s_width);

}


/*************************************************************************/

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
	s.push_back(1.6848);
	s.push_back(152.78);
	s.push_back(178.57);

	s0.push_back(352.18);
	s0.push_back(331.46);
	s0.push_back(1.6848);
	s0.push_back(152.78);
	s0.push_back(178.57);

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
	P[0][2] = 0.290219;
	P[0][3] = -354.473747;
	P[0][4] = -470.330553;

	P[1][0] = 37.669557;
	P[1][1] = 262.201038;
	P[1][2] = 0.471210;
	P[1][3] = 3.523747;
	P[1][4] = 71.038316;

	P[2][0] = 0.290219;
	P[2][1] = 0.471210;
	P[2][2] = 0.003764;
	P[2][3] = 0.014130;
	P[2][4] = 0.290246;

	P[3][0] = -354.473747;
	P[3][1] = 3.523747;
	P[3][2] = 0.014130;
	P[3][3] = 176.328000;
	P[3][4] = 274.791053;

	P[4][0] = -470.330553;
	P[4][1] = 71.038316;
	P[4][2] = 0.290246;
	P[4][3] = 274.791053;
	P[4][4] = 508.217763;


	invP[0][0] = 0.000141;
	invP[0][1] = 0.000020;
	invP[0][2] = -0.003878;
	invP[0][3] = 0.000514;
	invP[0][4] = -0.000148;

	invP[1][0] = 0.000020;
	invP[1][1] = 0.005368;
	invP[1][2] = -0.471001;
	invP[1][3] = 0.004396;
	invP[1][4] = -0.002840;

	invP[2][0] = -0.003878;
	invP[2][1] = -0.471001;
	invP[2][2] = 390.568673;
	invP[2][3] = 1.403795;
	invP[2][4] = -0.919834;

	invP[3][0] = 0.000514;
	invP[3][1] = 0.004396;
	invP[3][2] = 1.403795;
	invP[3][3] = 0.050642;
	invP[3][4] = -0.028323;

	invP[4][0] = -0.000148;
	invP[4][1] = -0.002840;
	invP[4][2] = -0.919834;
	invP[4][3] = -0.028323;
	invP[4][4] = 0.018067;


	R[0][0] = 7852.336300;
	R[0][1] = 162.862329;
	R[0][2] = 0.304703;
	R[0][3] = -851.867079;
	R[0][4] = -889.894553;

	R[1][0] = 162.862329;
	R[1][1] = 282.185173;
	R[1][2] = 0.466120;
	R[1][3] = -148.634526;
	R[1][4] = -54.218000;

	R[2][0] = 0.304703;
	R[2][1] = 0.466120;
	R[2][2] = 0.003764;
	R[2][3] = 0.035326;
	R[2][4] = 0.580491;

	R[3][0] = -851.867079;
	R[3][1] = -148.634526;
	R[3][2] = 0.035326;
	R[3][3] = 1102.050000;
	R[3][4] = 1373.955263;

	R[4][0] = -889.894553;
	R[4][1] = -54.218000;
	R[4][2] = 0.580491;
	R[4][3] = 1373.955263;
	R[4][4] = 2032.871053;


	invR[0][0] = 0.000141;
	invR[0][1] = 0.000020;
	invR[0][2] = -0.003881;
	invR[0][3] = 0.000208;
	invR[0][4] = -0.000077;

	invR[1][0] = 0.000020;
	invR[1][1] = 0.005367;
	invR[1][2] = -0.470892;
	invR[1][3] = 0.002525;
	invR[1][4] = -0.001420;

	invR[2][0] = -0.003881;
	invR[2][1] = -0.470892;
	invR[2][2] = 390.553502;
	invR[2][3] = 0.494273;
	invR[2][4] = -0.459845;

	invR[3][0] = 0.000208;
	invR[3][1] = 0.002525;
	invR[3][2] = 0.494273;
	invR[3][3] = 0.008715;
	invR[3][4] = -0.005873;

	invR[4][0] = -0.000077;
	invR[4][1] = -0.001420;
	invR[4][2] = -0.459845;
	invR[4][3] = -0.005873;
	invR[4][4] = 0.004521;


	//srodkowy palec


	s_MFinger.push_back(347.62);
	s_MFinger.push_back(159.3);
	s_MFinger.push_back(1.6848);
	s_MFinger.push_back(177.87);
	s_MFinger.push_back(42.858);

	s0_MFinger.push_back(347.62);
	s0_MFinger.push_back(159.3);
	s0_MFinger.push_back(1.6848);
	s0_MFinger.push_back(177.87);
	s0_MFinger.push_back(42.858);


	 P_MFinger[0][0] = 7888.528068;
	 P_MFinger[0][1] = 406.435705;
	 P_MFinger[0][2] = -0.135018;
	 P_MFinger[0][3] = -435.621592;
	 P_MFinger[0][4] = -122.957084;

	 P_MFinger[1][0] = 406.435705;
	 P_MFinger[1][1] = 521.170785;
	 P_MFinger[1][2] = 0.491322;
	 P_MFinger[1][3] = -285.146180;
	 P_MFinger[1][4] = -70.328615;

	 P_MFinger[2][0] = -0.135018;
	 P_MFinger[2][1] = 0.491322;
	 P_MFinger[2][2] = 0.003764;
	 P_MFinger[2][3] = 0.034951;
	 P_MFinger[2][4] = 0.069659;

	 P_MFinger[3][0] = -435.621592;
	 P_MFinger[3][1] = -285.146180;
	 P_MFinger[3][2] = 0.034951;
	 P_MFinger[3][3] = 348.553026;
	 P_MFinger[3][4] = 97.536609;

	 P_MFinger[4][0] = -122.957084;
	 P_MFinger[4][1] = -70.328615;
	 P_MFinger[4][2] = 0.069659;
	 P_MFinger[4][3] = 97.536609;
	 P_MFinger[4][4] = 29.273343;


	invP_MFinger[0][0] = 0.000137;
	invP_MFinger[0][1] = -0.000040;
	invP_MFinger[0][2] = 0.008904;
	invP_MFinger[0][3] = 0.000139;
	invP_MFinger[0][4] = -0.000007;

	invP_MFinger[1][0] = -0.000040;
	invP_MFinger[1][1] = 0.004670;
	invP_MFinger[1][2] = -0.560854;
	invP_MFinger[1][3] = 0.005335;
	invP_MFinger[1][4] = -0.005386;

	invP_MFinger[2][0] = 0.008904;
	invP_MFinger[2][1] = -0.560854;
	invP_MFinger[2][2] = 580.463754;
	invP_MFinger[2][3] = 3.656078;
	invP_MFinger[2][4] = -14.873096;

	invP_MFinger[3][0] = 0.000139;
	invP_MFinger[3][1] = 0.005335;
	invP_MFinger[3][2] = 3.656078;
	invP_MFinger[3][3] = 0.084669;
	invP_MFinger[3][4] = -0.277407;

	invP_MFinger[4][0] = -0.000007;
	invP_MFinger[4][1] = -0.005386;
	invP_MFinger[4][2] = -14.873096;
	invP_MFinger[4][3] = -0.277407;
	invP_MFinger[4][4] = 0.980883;


	R_MFinger[0][0] = 7895.858947;
	R_MFinger[0][1] = 0.497900;
	R_MFinger[0][2] = 7772.402632;
	R_MFinger[0][3] = 668.163684;
	R_MFinger[0][4] = 0.634515;
	R_MFinger[0][5] = -788.814211;

	R_MFinger[1][0] = 0.497900;
	R_MFinger[1][1] = 303.854636;
	R_MFinger[1][2] = -154.575263;
	R_MFinger[1][3] = 124.265368;
	R_MFinger[1][4] = 0.447130;
	R_MFinger[1][5] = 344.682000;

	R_MFinger[2][0] = 7772.402632;
	R_MFinger[2][1] = -154.575263;
	R_MFinger[2][2] = 7986.368421;
	R_MFinger[2][3] = 609.263158;
	R_MFinger[2][4] = -0.464874;
	R_MFinger[2][5] = -1125.710526;

	R_MFinger[3][0] = 668.163684;
	R_MFinger[3][1] = 124.265368;
	R_MFinger[3][2] = 609.263158;
	R_MFinger[3][3] = 901.252632;
	R_MFinger[3][4] = 0.510406;
	R_MFinger[3][5] = -984.963158;

	R_MFinger[4][0] = 0.634515;
	R_MFinger[4][1] = 0.447130;
	R_MFinger[4][2] = -0.464874;
	R_MFinger[4][3] = 0.510406;
	R_MFinger[4][4] = 0.003764;
	R_MFinger[4][5] = 0.580491;

	R_MFinger[5][0] = -788.814211;
	R_MFinger[5][1] = 344.682000;
	R_MFinger[5][2] = -1125.710526;
	R_MFinger[5][3] = -984.963158;
	R_MFinger[5][4] = 0.580491;
	R_MFinger[5][5] = 2032.871053;


	invR_MFinger[0][0] = 0.529629;
	invR_MFinger[0][1] = 0.008552;
	invR_MFinger[0][2] = -0.530462;
	invR_MFinger[0][3] = -0.010065;
	invR_MFinger[0][4] = -146.296595;
	invR_MFinger[0][5] = -0.052785;

	invR_MFinger[1][0] = 0.008552;
	invR_MFinger[1][1] = 0.036847;
	invR_MFinger[1][2] = -0.008406;
	invR_MFinger[1][3] = -0.031746;
	invR_MFinger[1][4] = 1.035149;
	invR_MFinger[1][5] = -0.023261;

	invR_MFinger[2][0] = -0.530462;
	invR_MFinger[2][1] = -0.008406;
	invR_MFinger[2][2] = 0.531433;
	invR_MFinger[2][3] = 0.009892;
	invR_MFinger[2][4] = 146.558360;
	invR_MFinger[2][5] = 0.052817;

	invR_MFinger[3][0] = -0.010065;
	invR_MFinger[3][1] = -0.031746;
	invR_MFinger[3][2] = 0.009892;
	invR_MFinger[3][3] = 0.031387;
	invR_MFinger[3][4] = -1.029756;
	invR_MFinger[3][5] = 0.022456;

	invR_MFinger[4][0] = -146.296595;
	invR_MFinger[4][1] = 1.035149;
	invR_MFinger[4][2] = 146.558360;
	invR_MFinger[4][3] = -1.029756;
	invR_MFinger[4][4] = 41198.890066;
	invR_MFinger[4][5] = 11.950979;

	invR_MFinger[5][0] = -0.052785;
	invR_MFinger[5][1] = -0.023261;
	invR_MFinger[5][2] = 0.052817;
	invR_MFinger[5][3] = 0.022456;
	invR_MFinger[5][4] = 11.950979;
	invR_MFinger[5][5] = 0.020669;


}


}//: namespace KW_MAP2
}//: namespace Processors
