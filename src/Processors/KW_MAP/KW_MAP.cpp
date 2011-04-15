/*!
 * \file KW_MAP.cpp
 * \brief Estymacja MAP, bez uśredniania punktów
 * \author kwasak
 * \date 2011-04-03
 */

#include <memory>
#include <string>
#include <math.h> 

#include "KW_MAP.hpp"
#include "Logger.hpp"
#include "Types/Ellipse.hpp"
#include "Types/Line.hpp"
#include "Types/Rectangle.hpp"
#include <vector>
#include <iomanip>

namespace Processors {
namespace KW_MAP {

using namespace cv;



KW_MAP::~KW_MAP() {
	LOG(LTRACE) << "Good bye KW_MAP\n";
}

bool KW_MAP::onInit() {
	LOG(LTRACE) << "KW_MAP::initialize\n";

	h_onNewImage.setup(this, &KW_MAP::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	h_onNewBlobs.setup(this, &KW_MAP::onNewBlobs);
	registerHandler("onNewBlobs", &h_onNewBlobs);

	registerStream("in_blobs", &in_blobs);
	registerStream("in_img", &in_img);

	newImage = registerEvent("newImage");

	registerStream("out_signs", &out_signs);
	registerStream("out_draw", &out_draw);

	learnRate = 0.01;
/*
	for( int i = 0; i < 29; i++)
	{
		pMean[i] = 0;
		cout<<pMean[i]<<"\n";
	}
*/
	first = true;

	return true;
}

bool KW_MAP::onFinish() {
	LOG(LTRACE) << "KW_MAP::finish\n";

/*	for (unsigned int i = 0; i < state.size(); i++)
    {
		pMean[i] = pMean[i]/ileObrazkow;
        cout<<pMean[i]<<"\n";
    }
*/
	return true;
}

bool KW_MAP::onStep() {
	LOG(LTRACE) << "KW_MAP::step\n";

	blobs_ready = img_ready = false;


	try {
		//ileObrazkow = ileObrazkow + 1;
		//cout<<"ilosc obrazkow"<<ileObrazkow<<"\n" ;

		drawcont.clear();
		z.clear();
		charPoint.clear();
		diff.clear();
		//state.clear();

		getCharPoints();

		if(first == true)
		{
			cout<<first<<"!!!!!!!!!!!!!!!!!!!!\n";
			 // z --> s, z pomiarów oblicza stan
			charPointsToState();
			//stateToCharPoint();
			first = false;
		}
		else
		{
			cout<<first<<"lalalallalalallalal!!!!!!!!!!!!!!!!!!\n";
			cout<<"jestem tulalalallalalallalal!!!!!!!!!!!!!!!!!!\n";

			// s --> z
			stateToCharPoint();
			calculateDiff();
			updateState();
		}

		out_draw.write(drawcont);
		newImage->raise();

		return true;
	} catch (...) {
		LOG(LERROR) << "KW_MAP::getCharPoints failed\n";
		return false;
	}
}

bool KW_MAP::onStop() {
	return true;
}

bool KW_MAP::onStart() {
	return true;
}

void KW_MAP::onNewImage() {
	LOG(LTRACE) << "KW_MAP::onNewImage\n";

	img_ready = true;
	tsl_img = in_img.read();
	//co robi tak linijka?
	tsl_img = tsl_img.clone();
	if (blobs_ready && img_ready)
		onStep();
}

void KW_MAP::onNewBlobs() {
	LOG(LTRACE) << "KW_MAP::onNewBlobs\n";

	blobs_ready = true;
	blobs = in_blobs.read();
	if (blobs_ready && img_ready)
		onStep();
}

void KW_MAP::getCharPoints() {

	LOG(LTRACE) << "KW_MAP::getCharPoints\n";

	try {

		int id = 0;
		//numerElements - liczba punktów wchodzących w skład konturu
		// i, ii - indeksy
		unsigned int numerElements;
		std::ofstream plik("/home/kasia/Test.txt");
		Types::Blobs::Blob *currentBlob;
		Types::Blobs::BlobResult result;
		CvSeq * contour;
		CvSeqReader reader;
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
		int lastSign, lastMinDist;
		//idenksy punktów charakterystycznych;
		vector<int> indexPoint;
		//powyzej tej odległości od środa cieżkosci moga znajdować sie ekstrema
		int MINDIST;

		Types::DrawableContainer signs; //kontener przechowujący elementy, które mozna narysować

		// iterate through all found blobs

		double m00, m10, m01;
		double Area, MaxArea, CenterOfGravity_x, CenterOfGravity_y, MaxY;

		MaxArea = 0;
		MaxY = 0;

		//największy blob to dłoń
		for (int i = 0; i < blobs.GetNumBlobs(); i++) {
			currentBlob = blobs.GetBlob(i);

			Area = currentBlob->Area();
			if (Area > MaxArea) {
				MaxArea = Area;
				// id największego bloba, czyli dłoni
				id = i;
			}
		}

		//obliczenia tylko dla najwiekszego blobu, czyli dloni
		currentBlob = blobs.GetBlob(id);
		contour = currentBlob->GetExternalContour()->GetContourPoints();
		cvStartReadSeq(contour, &reader);

		int cnt = 0;
		for (int j = 0; j < contour->total; j = j + 1) {
			CV_READ_SEQ_ELEM( actualPoint, reader);

			if (j % 10 == 1) {
				//plik << actualPoint.x << " " << actualPoint.y << std::endl;
				contourPoints.push_back(cv::Point(actualPoint.x, actualPoint.y));
				if (actualPoint.y > MaxY) {
					MaxY = actualPoint.y;
				}
				cnt++;
			}
		}

		//środek cięzkości
		// calculate moments
		m00 = currentBlob->Moment(0, 0);
		m01 = currentBlob->Moment(0, 1);
		m10 = currentBlob->Moment(1, 0);

		CenterOfGravity_x = m10 / m00;
		CenterOfGravity_y = m01 / m00;

		MINDIST = (MaxY - CenterOfGravity_y) * (MaxY - CenterOfGravity_y) * 4/ 9;
		//przesuniety punkt środka ciężkości
		charPoint.push_back(cv::Point(CenterOfGravity_x, CenterOfGravity_y + (MaxY - CenterOfGravity_y) * 4 / 5));

		CenterOfGravity_y += (MaxY - CenterOfGravity_y) * 2 / 3;

		numerElements = contourPoints.size();

		//******************************************************************
		//obliczenie roznicy miedzy punktami z odwodu a środkiem ciężkosci
		for (unsigned int i = 0; i < numerElements; i++) {
			TempDist = (contourPoints[i].x - CenterOfGravity_x)	* (contourPoints[i].x - CenterOfGravity_x)	+ (contourPoints[i].y - CenterOfGravity_y) * (contourPoints[i].y - CenterOfGravity_y);
			if (TempDist > MINDIST)
				dist.push_back(TempDist);
			else
				dist.push_back(MINDIST);
		}

		//******************************************************************
		//obliczenie pochodnej, szukanie ekstremów
		derivative.push_back(dist[1] - dist[0]);
		if (derivative[0] > 0)
			lastSign = 1;
		else
			lastSign = -1;

		lastMinDist = 0;
		//pierwszy punkt kontury to wierzchołek punktu środkowego.
		indexPoint.push_back(0);

		for (unsigned int i = 1; i < numerElements - 2; i++) {
			plik << dist[i] << "\n";
			derivative.push_back(dist[i + 1] - dist[i]);

			if (dist[i + 1] > MINDIST && dist[i] > MINDIST) {
				if (lastMinDist == 1) {
					lastSign = 1;
					lastMinDist = 0;
				}
				//maksiumum, funkcja rosła i zaczeła maleć
				if (derivative[i] < 0 && lastSign == 1) {
					indexPoint.push_back(i);
					lastSign = -1;
				}
				//minimum
				else if (derivative[i] > 0 && lastSign == -1) {
					indexPoint.push_back(i);
					lastSign = 1;
				}
			} else {
				lastMinDist = 1;
			}
		}

		//	Types::Ellipse * el = new Types::Ellipse(Point2f(CenterOfGravity_x, CenterOfGravity_y), Size2f(20,20));
		//	drawcont.add(el);
		//	Types::Ellipse * el2 = new Types::Ellipse(Point2f(last_x, last_y), Size2f(7,7));
		//	drawcont.add(el2);

		last_x = CenterOfGravity_x;
		last_y = CenterOfGravity_y;

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
			charPoint.push_back(cv::Point(contourPoints[indexPoint[i]].x,
					contourPoints[indexPoint[i]].y));
			//	drawcont.add(new Types::Ellipse(Point2f(contourPoints[indexPoint[i]].x, contourPoints[indexPoint[i]].y), Size2f(10,10)));
		}

		for (int i = indexPoint.size() - 1; i > idLeftPoint; i--) {
			charPoint.push_back(cv::Point(contourPoints[indexPoint[i]].x, contourPoints[indexPoint[i]].y));
			//	drawcont.add(new Types::Ellipse(Point2f(contourPoints[indexPoint[i]].x, contourPoints[indexPoint[i]].y), Size2f(10,10)));
		}

		//plik <<"Punkt środka cieżkosci: "<< CenterOfGravity_x <<" "<< CenterOfGravity_y;

		Types::Ellipse * el;


		el = new Types::Ellipse(Point2f(charPoint[0].x, charPoint[0].y), Size2f(10,10));
		el->setCol(CV_RGB(255,0,0));
		drawcont.add(el);

		el = new Types::Ellipse(Point2f(charPoint[1].x, charPoint[1].y), Size2f(10,10));
		el->setCol(CV_RGB(0,0,255));
		drawcont.add(el);

		drawcont.add(new Types::Ellipse(Point2f(charPoint[2].x, charPoint[2].y), Size2f(10,10)));
		drawcont.add(new Types::Ellipse(Point2f(charPoint[3].x, charPoint[3].y), Size2f(10,10)));
		drawcont.add(new Types::Ellipse(Point2f(charPoint[4].x, charPoint[4].y), Size2f(10,10)));
		drawcont.add(new Types::Ellipse(Point2f(charPoint[5].x, charPoint[5].y), Size2f(10,10)));
		drawcont.add(new Types::Ellipse(Point2f(charPoint[6].x, charPoint[6].y), Size2f(10,10)));
		drawcont.add(new Types::Ellipse(Point2f(charPoint[7].x, charPoint[7].y), Size2f(10,10)));
		drawcont.add(new Types::Ellipse(Point2f(charPoint[8].x, charPoint[8].y), Size2f(10,10)));
		drawcont.add(new Types::Ellipse(Point2f(charPoint[9].x, charPoint[9].y), Size2f(10,10)));

		result.AddBlob(blobs.GetBlob(id));
		out_signs.write(result);

	} catch (...) {
		LOG(LERROR) << "KW_MAP::getCharPoints failed\n";

	}
}

void KW_MAP::charPointsToState() {
	LOG(LTRACE) << "KW_MAP::charPointsToState\n";

	//obliczanie parametrów prostokątaq opisującego wewnętrzą część dłoni
	//wspolrzedna x lewego gornego punktu
	state.push_back(charPoint[0].x - (charPoint[8].x - charPoint[0].x ));
	//wspolrzedna y lewego gornego punktu
	state.push_back(charPoint[6].y);
	//szerokosc
	state.push_back(abs(2 * (charPoint[8].x - charPoint[0].x )));
	//wysokosc
	state.push_back(abs(charPoint[0].y - charPoint[6].y));


//	drawcont.add(new Types::Rectangle(state[0], state[1], state[2], state[3]));
//	drawcont.add(new Types::Line(cv::Point(charPoint[0].x, charPoint[0].y),cv::Point(charPoint[1].x, charPoint[1].y)));
//	drawcont.add(new Types::Line(cv::Point(charPoint[0].x, charPoint[0].y),cv::Point(charPoint[3].x, charPoint[3].y)));
//	drawcont.add(new Types::Line(cv::Point(charPoint[0].x, charPoint[0].y),cv::Point(charPoint[5].x, charPoint[5].y)));
//	drawcont.add(new Types::Line(cv::Point(charPoint[0].x, charPoint[0].y),cv::Point(charPoint[7].x, charPoint[7].y)));
//	drawcont.add(new Types::Line(cv::Point(charPoint[0].x, charPoint[0].y),cv::Point(charPoint[9].x, charPoint[9].y)));

	fingerToState(charPoint[1], charPoint[2], 1);
	fingerToState(charPoint[3], charPoint[4], 1);
	fingerToState(charPoint[5], charPoint[6], 1);
	fingerToState(charPoint[7], charPoint[6], -1);
	fingerToState(charPoint[9], charPoint[8], -1);
/*
	for(unsigned int i = 0; i < state.size(); i++)
	{
		pMean[i] += state[i];
		//cout<<pMean[i]<<"\n";
	}

*/
}

cv::Point KW_MAP::rot(cv::Point p, double angle, cv::Point p0) {
	cv::Point t;
	t.x = p0.x + (int) ((double) (p.x - p0.x) * cos(angle) - (double) (p.y - p0.y) * sin(angle));
	t.y = p0.y + (int) ((double) (p.x - p0.x) * sin(angle) + (double) (p.y - p0.y) * cos(angle));
	return t;
}

//p2 - czubek palca, p1 - punkt miedzy palcami
void KW_MAP::fingerToState(cv::Point p2, cv::Point p1, int sig) {

	LOG(LTRACE) << "KW_MAP::fingerToState\n";

	double uj = (double) (-p2.x + charPoint[0].x) / (-p2.y + charPoint[0].y);
	double angle = atan(uj);
	cv::Point pt1 = rot(p1, angle, charPoint[0]);
	cv::Point pt2 = rot(p2, angle, charPoint[0]);

	cv::Point statePoint;
	cv::Point statePoint2;
	cv::Point statePoint3;
	cv::Point statePoint4;

	if(sig == 1)
		statePoint.x = pt2.x - (pt1.x - pt2.x);
	else if (sig == -1)
		statePoint.x = pt1.x;

	statePoint.y = pt2.y;
	int width = abs(2 * (pt1.x - pt2.x));
	int height = abs(pt1.y - pt2.y);

	statePoint2.x = statePoint.x;
	statePoint2.y = statePoint.y + height;

	statePoint3.x = statePoint.x + width;
	statePoint3.y = statePoint.y + height;

	statePoint4.x = statePoint.x + width;
	statePoint4.y = statePoint.y;

	angle = -angle;
	statePoint = rot(statePoint, angle, charPoint[0]);
	statePoint2 = rot(statePoint2, angle, charPoint[0]);
	statePoint3 = rot(statePoint3, angle, charPoint[0]);
	statePoint4 = rot(statePoint4, angle, charPoint[0]);

	//górny lewy wierzchołek
	state.push_back(statePoint.x);
	state.push_back(statePoint.y);
	//szerokosc
	state.push_back(width);
	//wysokosc
	state.push_back(height);
	state.push_back(-angle);



//	drawcont.add(new Types::Line(cv::Point(statePoint.x, statePoint.y), cv::Point(statePoint2.x, statePoint2.y)));
//	drawcont.add(new Types::Line(cv::Point(statePoint2.x, statePoint2.y), cv::Point(statePoint3.x, statePoint3.y)));
//	drawcont.add(new Types::Line(cv::Point(statePoint3.x, statePoint3.y), cv::Point(statePoint4.x, statePoint4.y)));
//	drawcont.add(new Types::Line(cv::Point(statePoint4.x, statePoint4.y), cv::Point(statePoint.x, statePoint.y)));
}

//******************************************SPRAWDŹ CZY DZIALA***********************************************

//funkcja obliczajaca punkty charakterystyczne trzech lewych palców
void KW_MAP::stateToFinger(double s1, double s2, double s3, double s4, double angle, int sig)
{
	LOG(LTRACE) << "KW_MAP::stateToFinger\n";

	cv::Point rotPoint;
	cv::Point tempPoint;

	if (sig == 1)
	{
		tempPoint.x = s1 + 0.5 * s3 * cos(angle);
		tempPoint.y = s2 - 0.5 * s3 * sin(angle);
		z.push_back(tempPoint);

		tempPoint.x = s1 + s3 * cos(angle) + s4 * sin(angle);
		tempPoint.y = s2 - s3 * sin(angle) + s4 * cos(angle);
		z.push_back(tempPoint);
	}
	if (sig == 2)
	{
		tempPoint.x = s1 +  0.5 * s3 * cos(angle);
		tempPoint.y = s2 -  0.5 * s3 * sin(angle);
		z.push_back(tempPoint);
	}
	if (sig == 3)
	{
		tempPoint.x = s1 + s4 * sin(angle);
		tempPoint.y = s2 + s4 * cos(angle);
		z.push_back(tempPoint);

		tempPoint.x = s1 +  0.5 * s3 * cos(angle);
		tempPoint.y = s2 -  0.5 * s3 * sin(angle);
		z.push_back(tempPoint);
	}
}

void KW_MAP::stateToCharPoint()
{
	LOG(LTRACE) << "KW_MAP::stateToCharPoint\n";

	cv::Point rotPoint;
	cv::Point tempPoint;
	// punkt dołu dłoni
	z.push_back(cv::Point((state[0] + 0.5*state[2]), (state[1] + state[3])));

	//punkty pierwszego palca od lewej
	stateToFinger(state[4], state[5], state[6], state[7], state[8],1);
	//punkty drugiego palca od lewej
	stateToFinger(state[9], state[10], state[11], state[12], state[13],1);
	//punkty środkowego palca
	stateToFinger(state[14], state[15], state[16], state[17], state[18],1);
	//punkty czwartego palca od lewej
	stateToFinger(state[19], state[20], state[21], state[22], state[23],2);
	//punkty kciuka
	stateToFinger(state[24], state[25], state[26], state[27], state[28],3);

	drawcont.add(new Types::Ellipse(Point2f(z[0].x, z[0].y), Size2f(20,20)));
	drawcont.add(new Types::Ellipse(Point2f(z[1].x, z[1].y), Size2f(20,20)));
	drawcont.add(new Types::Ellipse(Point2f(z[2].x, z[2].y), Size2f(20,20)));
	drawcont.add(new Types::Ellipse(Point2f(z[3].x, z[3].y), Size2f(20,20)));
	drawcont.add(new Types::Ellipse(Point2f(z[4].x, z[4].y), Size2f(20,20)));
	drawcont.add(new Types::Ellipse(Point2f(z[5].x, z[5].y), Size2f(20,20)));
	drawcont.add(new Types::Ellipse(Point2f(z[6].x, z[6].y), Size2f(20,20)));
	drawcont.add(new Types::Ellipse(Point2f(z[7].x, z[7].y), Size2f(20,20)));
	drawcont.add(new Types::Ellipse(Point2f(z[8].x, z[8].y), Size2f(20,20)));
	drawcont.add(new Types::Ellipse(Point2f(z[9].x, z[9].y), Size2f(20,20)));


}

void KW_MAP::derivatives(int indexR, int indexC, double a, double b, double c, double d, double e, int sig)
{
	double cosE = cos(e);
	double sinE = sin(e);

	if(sig==3)
	{

		H[indexR][indexC] = 1;
		H[indexR + 3][indexC] = sinE;
	//	H[indexR + 4][indexC] = cosE;
		H[indexR + 4][indexC] = d * cosE;

		indexC += 1;
		H[indexR + 1][indexC] = 1;
		H[indexR + 3][indexC] = cosE;
	//	H[indexR + 4][indexC] = - sinE;
		H[indexR + 4][indexC] = - d * sinE;

		indexC += 1;

	}

	H[indexR][indexC] = 1;
	H[indexR + 2][indexC] = 0.5 * cosE;
	//H[indexR + 4][indexC] = -0.5 * sinE;
	H[indexR + 4][indexC] = -0.5 * c * sinE;

	indexC += 1;
	H[indexR + 1][indexC] = 1;
	H[indexR + 2][indexC] = -0.5 * sinE;
	H[indexR + 4][indexC] = -0.5 * c * cosE;
	//H[indexR + 4][indexC] = -0.5 * cosE;

	if(sig == 1)
	{
		indexC += 1;
		H[indexR][indexC] = 1;
		H[indexR + 2][indexC] = cosE;
		H[indexR + 3][indexC] = sinE;
	//	H[indexR + 4][indexC] = - sinE + cosE;
		H[indexR + 4][indexC] = - c * sinE + d * cosE;


		indexC += 1;
		H[indexR + 1][indexC] = 1;
		H[indexR + 2][indexC] = - sinE;
		H[indexR + 3][indexC] = cosE;
	//	H[indexR + 4][indexC] = - cosE - sinE;
		H[indexR + 4][indexC] = - c * cosE - d * sinE;

	}

}


void KW_MAP::calculateH()
{
	for(int i = 0; i < 29; i++)
	{
		for(int j = 0; j < 20; j++)
		{
			H[i][j]=0;
		}
		//cout<<"\n";
	}
	H[0][0] = 1;
	H[2][0] = 0.5;
	H[1][1] = 1;
	H[3][1] = 1;

	derivatives(4,2, state[4], state[5], state[6], state[7], state[8],1);
	derivatives(9,6, state[9], state[10], state[11], state[12], state[13],1);
	derivatives(14,10,state[14], state[15], state[16], state[17], state[18],1);
	derivatives(19,14,state[19], state[20], state[21], state[22], state[23],2);
	derivatives(24,16,state[24], state[25], state[26], state[27], state[28],3);

	for(int i = 0; i < 29; i++)
	{
		for(int j = 0; j < 20; j++)
		{
			//cout << setprecision(3)<<H[i][j]<<"\t";
		}
		//cout<<"\n";
	}
}

void KW_MAP::calculateDiff()
{
	LOG(LTRACE) << "KW_MAP::calculateDiff\n";

	cout<<"KW_MAP::calculateDiff\n";
	//różnica
	double D[20];
	double error = 0;
	unsigned int j = 0;

    for (unsigned int i = 0 ; i < z.size() * 2; i = i + 2)
    {
        D[i] = - z[j].x + charPoint[j].x ;
   //     cout<<"D:"<< D[i]<<"\n";
        D[i + 1] = - z[j].y + charPoint[j].y;
   //     cout<<"D:"<< D[i + 1]<<"\n";
        j += 1;
    }

    calculateH();

    double t[29];
    for (unsigned int i = 0; i < state.size(); i++)
    {
        t[i] = 0;
        for  (j = 0; j < z.size() * 2; j++)
        {

            //mnożenie macierzy H * roznica S
            t[i] += H[i][j] * D[j];

        }
       // cout << "\n";
        //wspolczynnik zapominania
        learnRate = 0.01;
        t[i] *= learnRate;
        //obliczony blad
        error += abs(t[i]);
        cout<<"t "<<t[i]<<"\n";
        diff.push_back(t[i]);
    }
}

void KW_MAP::updateState()
 {
	cout<<"KW_MAP::updateState\n";


     for (unsigned int i = 0; i < state.size(); i++)
     {

         state[i] = state[i] + diff[i];
     }
 }


KW_MAP::KW_MAP(const std::string & name) :
	Base::Component(name) {
	LOG(LTRACE) << "Hello KW_MAP\n";
	ileObrazkow = 0;
	P[0][0]=59.8301;
	P[0][1]=-0.575163;
	P[1][0]=-0.575163;
	P[0][2]=-13.451;
	P[2][0]=-13.451;
	P[0][3]=0.477124;
	P[3][0]=0.477124;
	P[0][4]=74.8366;
	P[4][0]=74.8366;
	P[0][5]=-37.8497;
	P[5][0]=-37.8497;
	P[0][6]=-6.75817;
	P[6][0]=-6.75817;
	P[0][7]=7.79739;
	P[7][0]=7.79739;
	P[0][8]=-0.141139;
	P[8][0]=-0.141139;
	P[0][9]=82.8366;
	P[9][0]=82.8366;
	P[0][10]=-18.5621;
	P[10][0]=-18.5621;
	P[0][11]=-10.5882;
	P[11][0]=-10.5882;
	P[0][12]=-9.40523;
	P[12][0]=-9.40523;
	P[0][13]=-0.0921031;
	P[13][0]=-0.0921031;
	P[0][14]=77.915;
	P[14][0]=77.915;
	P[0][15]=-6.86275;
	P[15][0]=-6.86275;
	P[0][16]=-14.719;
	P[16][0]=-14.719;
	P[0][17]=1.33333;
	P[17][0]=1.33333;
	P[0][18]=-0.0577324;
	P[18][0]=-0.0577324;
	P[0][19]=61.8235;
	P[19][0]=61.8235;
	P[0][20]=-2.35948;
	P[20][0]=-2.35948;
	P[0][21]=24.1046;
	P[21][0]=24.1046;
	P[0][22]=2.85621;
	P[22][0]=2.85621;
	P[0][23]=-0.0699018;
	P[23][0]=-0.0699018;
	P[0][24]=59.3922;
	P[24][0]=59.3922;
	P[0][25]=3.51634;
	P[25][0]=3.51634;
	P[0][26]=17.9477;
	P[26][0]=17.9477;
	P[0][27]=10.1176;
	P[27][0]=10.1176;
	P[0][28]=-0.0672494;
	P[28][0]=-0.0672494;
	P[1][1]=0.379085;
	P[1][2]=0.156863;
	P[2][1]=0.156863;
	P[1][3]=-0.163399;
	P[3][1]=-0.163399;
	P[1][4]=-0.679739;
	P[4][1]=-0.679739;
	P[1][5]=0.535948;
	P[5][1]=0.535948;
	P[1][6]=0.601307;
	P[6][1]=0.601307;
	P[1][7]=-0.0522876;
	P[7][1]=-0.0522876;
	P[1][8]=1.09772e-05;
	P[8][1]=1.09772e-05;
	P[1][9]=-0.973856;
	P[9][1]=-0.973856;
	P[1][10]=0.228758;
	P[10][1]=0.228758;
	P[1][11]=0;
	P[11][1]=0;
	P[1][12]=-0.163399;
	P[12][1]=-0.163399;
	P[1][13]=0.00155437;
	P[13][1]=0.00155437;
	P[1][14]=-1.40523;
	P[14][1]=-1.40523;
	P[1][15]=-0.0196078;
	P[15][1]=-0.0196078;
	P[1][16]=1.97386;
	P[16][1]=1.97386;
	P[1][17]=0.431373;
	P[17][1]=0.431373;
	P[1][18]=-0.000309872;
	P[18][1]=-0.000309872;
	P[1][19]=0.764706;
	P[19][1]=0.764706;
	P[1][20]=0.339869;
	P[20][1]=0.339869;
	P[1][21]=-2.02614;
	P[21][1]=-2.02614;
	P[1][22]=0.124183;
	P[22][1]=0.124183;
	P[1][23]=-0.000658766;
	P[23][1]=-0.000658766;
	P[1][24]=-0.686275;
	P[24][1]=-0.686275;
	P[1][25]=-0.202614;
	P[25][1]=-0.202614;
	P[1][26]=-0.575163;
	P[26][1]=-0.575163;
	P[1][27]=-0.0588235;
	P[27][1]=-0.0588235;
	P[1][28]=0.00323997;
	P[28][1]=0.00323997;
	P[2][2]=17.0588;
	P[2][3]=1.07843;
	P[3][2]=1.07843;
	P[2][4]=-1.13725;
	P[4][2]=-1.13725;
	P[2][5]=-1.72549;
	P[5][2]=-1.72549;
	P[2][6]=-14.9804;
	P[6][2]=-14.9804;
	P[2][7]=-1.01961;
	P[7][2]=-1.01961;
	P[2][8]=0.0113829;
	P[8][2]=0.0113829;
	P[2][9]=-0.137255;
	P[9][2]=-0.137255;
	P[2][10]=0.54902;
	P[10][2]=0.54902;
	P[2][11]=-14.1176;
	P[11][2]=-14.1176;
	P[2][12]=0.490196;
	P[12][2]=0.490196;
	P[2][13]=0.00779605;
	P[13][2]=0.00779605;
	P[2][14]=-3.19608;
	P[14][2]=-3.19608;
	P[2][15]=-0.823529;
	P[15][2]=-0.823529;
	P[2][16]=-2.7451;
	P[16][2]=-2.7451;
	P[2][17]=0.176471;
	P[17][2]=0.176471;
	P[2][18]=-0.00151393;
	P[18][2]=-0.00151393;
	P[2][19]=-8.76471;
	P[19][2]=-8.76471;
	P[2][20]=0.509804;
	P[20][2]=0.509804;
	P[2][21]=-2.15686;
	P[21][2]=-2.15686;
	P[2][22]=-0.254902;
	P[22][2]=-0.254902;
	P[2][23]=0.0178924;
	P[23][2]=0.0178924;
	P[2][24]=-0.294118;
	P[24][2]=-0.294118;
	P[2][25]=5.66667;
	P[25][2]=5.66667;
	P[2][26]=-28.5098;
	P[26][2]=-28.5098;
	P[2][27]=-2.70588;
	P[27][2]=-2.70588;
	P[2][28]=0.0355193;
	P[28][2]=0.0355193;
	P[3][3]=0.565359;
	P[3][4]=0.836601;
	P[4][3]=0.836601;
	P[3][5]=-0.143791;
	P[5][3]=-0.143791;
	P[3][6]=0.300654;
	P[6][3]=0.300654;
	P[3][7]=0.326797;
	P[7][3]=0.326797;
	P[3][8]=-0.00118527;
	P[8][3]=-0.00118527;
	P[3][9]=1.98366;
	P[9][3]=1.98366;
	P[3][10]=-0.326797;
	P[10][3]=-0.326797;
	P[3][11]=-1.17647;
	P[11][3]=-1.17647;
	P[3][12]=0.212418;
	P[12][3]=0.212418;
	P[3][13]=-0.00174907;
	P[13][3]=-0.00174907;
	P[3][14]=2.76797;
	P[14][3]=2.76797;
	P[3][15]=-0.215686;
	P[15][3]=-0.215686;
	P[3][16]=-2.36601;
	P[16][3]=-2.36601;
	P[3][17]=-0.284314;
	P[17][3]=-0.284314;
	P[3][18]=-0.00196699;
	P[18][3]=-0.00196699;
	P[3][19]=0.0294118;
	P[19][3]=0.0294118;
	P[3][20]=-0.124183;
	P[20][3]=-0.124183;
	P[3][21]=2.04575;
	P[21][3]=2.04575;
	P[3][22]=0.00326797;
	P[22][3]=0.00326797;
	P[3][23]=0.000145284;
	P[23][3]=0.000145284;
	P[3][24]=2.0098;
	P[24][3]=2.0098;
	P[3][25]=1.31046;
	P[25][3]=1.31046;
	P[3][26]=-1.28758;
	P[26][3]=-1.28758;
	P[3][27]=0.294118;
	P[27][3]=0.294118;
	P[3][28]=-0.00357112;
	P[28][3]=-0.00357112;
	P[4][4]=134.967;
	P[4][5]=-78.5229;
	P[5][4]=-78.5229;
	P[4][6]=-58.5752;
	P[6][4]=-58.5752;
	P[4][7]=8.53595;
	P[7][4]=8.53595;
	P[4][8]=-0.254026;
	P[8][4]=-0.254026;
	P[4][9]=124.026;
	P[9][4]=124.026;
	P[4][10]=-26.8301;
	P[10][4]=-26.8301;
	P[4][11]=-28.3529;
	P[11][4]=-28.3529;
	P[4][12]=-14.6928;
	P[12][4]=-14.6928;
	P[4][13]=-0.133181;
	P[13][4]=-0.133181;
	P[4][14]=110.007;
	P[14][4]=110.007;
	P[4][15]=-11.4314;
	P[15][4]=-11.4314;
	P[4][16]=-18.9673;
	P[16][4]=-18.9673;
	P[4][17]=3.01961;
	P[17][4]=3.01961;
	P[4][18]=-0.0869846;
	P[18][4]=-0.0869846;
	P[4][19]=86.4118;
	P[19][4]=86.4118;
	P[4][20]=-2.95425;
	P[20][4]=-2.95425;
	P[4][21]=30.6797;
	P[21][4]=30.6797;
	P[4][22]=3.83007;
	P[22][4]=3.83007;
	P[4][23]=-0.0919034;
	P[23][4]=-0.0919034;
	P[4][24]=89.8431;
	P[24][4]=89.8431;
	P[4][25]=10.8562;
	P[25][4]=10.8562;
	P[4][26]=-2.81046;
	P[26][4]=-2.81046;
	P[4][27]=13.8235;
	P[27][4]=13.8235;
	P[4][28]=-0.068696;
	P[28][4]=-0.068696;
	P[5][5]=49.0458;
	P[5][6]=44.7974;
	P[6][5]=44.7974;
	P[5][7]=-4.30719;
	P[7][5]=-4.30719;
	P[5][8]=0.147317;
	P[8][5]=0.147317;
	P[5][9]=-69.2876;
	P[9][5]=-69.2876;
	P[5][10]=15.1307;
	P[10][5]=15.1307;
	P[5][11]=21.5294;
	P[11][5]=21.5294;
	P[5][12]=7.56209;
	P[12][5]=7.56209;
	P[5][13]=0.0733354;
	P[13][5]=0.0733354;
	P[5][14]=-55.1895;
	P[14][5]=-55.1895;
	P[5][15]=6.09804;
	P[15][5]=6.09804;
	P[5][16]=5.81699;
	P[16][5]=5.81699;
	P[5][17]=-1.80392;
	P[17][5]=-1.80392;
	P[5][18]=0.0448454;
	P[18][5]=0.0448454;
	P[5][19]=-47;
	P[19][5]=-47;
	P[5][20]=1.32026;
	P[20][5]=1.32026;
	P[5][21]=-13.0065;
	P[21][5]=-13.0065;
	P[5][22]=-1.60131;
	P[22][5]=-1.60131;
	P[5][23]=0.0497395;
	P[23][5]=0.0497395;
	P[5][24]=-47.6863;
	P[24][5]=-47.6863;
	P[5][25]=-6.06536;
	P[25][5]=-6.06536;
	P[5][26]=6.44444;
	P[26][5]=6.44444;
	P[5][27]=-7.17647;
	P[27][5]=-7.17647;
	P[5][28]=0.0292512;
	P[28][5]=0.0292512;
	P[6][6]=74.7712;
	P[6][7]=0.0915033;
	P[7][6]=0.0915033;
	P[6][8]=0.0913725;
	P[8][6]=0.0913725;
	P[6][9]=-35.3987;
	P[9][6]=-35.3987;
	P[6][10]=6.02614;
	P[10][6]=6.02614;
	P[6][11]=20;
	P[11][6]=20;
	P[6][12]=4.30065;
	P[12][6]=4.30065;
	P[6][13]=0.0386039;
	P[13][6]=0.0386039;
	P[6][14]=-17.3203;
	P[14][6]=-17.3203;
	P[6][15]=3.37255;
	P[15][6]=3.37255;
	P[6][16]=-6.48366;
	P[16][6]=-6.48366;
	P[6][17]=-1.72549;
	P[17][6]=-1.72549;
	P[6][18]=0.0207086;
	P[18][6]=0.0207086;
	P[6][19]=-20.8235;
	P[19][6]=-20.8235;
	P[6][20]=-0.300654;
	P[20][6]=-0.300654;
	P[6][21]=5.28105;
	P[21][6]=5.28105;
	P[6][22]=0.20915;
	P[22][6]=0.20915;
	P[6][23]=0.0135383;
	P[23][6]=0.0135383;
	P[6][24]=-21.9608;
	P[24][6]=-21.9608;
	P[6][25]=-4.71895;
	P[25][6]=-4.71895;
	P[6][26]=28.183;
	P[26][6]=28.183;
	P[6][27]=-1.88235;
	P[27][6]=-1.88235;
	P[6][28]=-0.0261226;
	P[28][6]=-0.0261226;
	P[7][7]=2.33987;
	P[7][8]=-0.0146677;
	P[8][7]=-0.0146677;
	P[7][9]=10.4771;
	P[9][7]=10.4771;
	P[7][10]=-2.1634;
	P[10][7]=-2.1634;
	P[7][11]=-2.11765;
	P[11][7]=-2.11765;
	P[7][12]=-0.379085;
	P[12][7]=-0.379085;
	P[7][13]=-0.00850956;
	P[13][7]=-0.00850956;
	P[7][14]=9.98693;
	P[14][7]=9.98693;
	P[7][15]=-0.901961;
	P[15][7]=-0.901961;
	P[7][16]=-0.418301;
	P[16][7]=-0.418301;
	P[7][17]=0.196078;
	P[17][7]=0.196078;
	P[7][18]=-0.00828857;
	P[18][7]=-0.00828857;
	P[7][19]=10.0588;
	P[19][7]=10.0588;
	P[7][20]=0.0849673;
	P[20][7]=0.0849673;
	P[7][21]=2.40523;
	P[21][7]=2.40523;
	P[7][22]=0.163399;
	P[22][7]=0.163399;
	P[7][23]=-0.0132319;
	P[23][7]=-0.0132319;
	P[7][24]=8.43137;
	P[24][7]=8.43137;
	P[7][25]=0.816993;
	P[25][7]=0.816993;
	P[7][26]=1.26797;
	P[26][7]=1.26797;
	P[7][27]=1.23529;
	P[27][7]=1.23529;
	P[7][28]=-0.00791758;
	P[28][7]=-0.00791758;
	P[8][8]=0.00053061;
	P[8][9]=-0.219402;
	P[9][8]=-0.219402;
	P[8][10]=0.0497243;
	P[10][8]=0.0497243;
	P[8][11]=0.0459302;
	P[11][8]=0.0459302;
	P[8][12]=0.025789;
	P[12][8]=0.025789;
	P[8][13]=0.000231049;
	P[13][8]=0.000231049;
	P[8][14]=-0.20024;
	P[14][8]=-0.20024;
	P[8][15]=0.0207764;
	P[15][8]=0.0207764;
	P[8][16]=0.0354618;
	P[16][8]=0.0354618;
	P[8][17]=-0.00616154;
	P[17][8]=-0.00616154;
	P[8][18]=0.000156116;
	P[18][8]=0.000156116;
	P[8][19]=-0.159382;
	P[19][8]=-0.159382;
	P[8][20]=0.00575166;
	P[20][8]=0.00575166;
	P[8][21]=-0.0645988;
	P[21][8]=-0.0645988;
	P[8][22]=-0.00793123;
	P[22][8]=-0.00793123;
	P[8][23]=0.000187822;
	P[23][8]=0.000187822;
	P[8][24]=-0.16393;
	P[24][8]=-0.16393;
	P[8][25]=-0.0208618;
	P[25][8]=-0.0208618;
	P[8][26]=-0.016176;
	P[26][8]=-0.016176;
	P[8][27]=-0.0288875;
	P[27][8]=-0.0288875;
	P[8][28]=0.000176437;
	P[28][8]=0.000176437;
	P[9][9]=177.232;
	P[9][10]=-41.9477;
	P[10][9]=-41.9477;
	P[9][11]=-83.7647;
	P[11][9]=-83.7647;
	P[9][12]=-18.0163;
	P[12][9]=-18.0163;
	P[9][13]=-0.195891;
	P[13][9]=-0.195891;
	P[9][14]=125.389;
	P[14][9]=125.389;
	P[9][15]=-12.8431;
	P[15][9]=-12.8431;
	P[9][16]=-22.732;
	P[16][9]=-22.732;
	P[9][17]=2.57843;
	P[17][9]=2.57843;
	P[9][18]=-0.103651;
	P[18][9]=-0.103651;
	P[9][19]=94.4412;
	P[19][9]=94.4412;
	P[9][20]=-3.30719;
	P[20][9]=-3.30719;
	P[9][21]=30.9739;
	P[21][9]=30.9739;
	P[9][22]=4.38889;
	P[22][9]=4.38889;
	P[9][23]=-0.0895337;
	P[23][9]=-0.0895337;
	P[9][24]=99.5784;
	P[24][9]=99.5784;
	P[9][25]=13.768;
	P[25][9]=13.768;
	P[9][26]=-12.7516;
	P[26][9]=-12.7516;
	P[9][27]=12.2941;
	P[27][9]=12.2941;
	P[9][28]=-0.0548689;
	P[28][9]=-0.0548689;
	P[10][10]=10.5752;
	P[10][11]=22.4706;
	P[11][10]=22.4706;
	P[10][12]=4.14379;
	P[12][10]=4.14379;
	P[10][13]=0.046305;
	P[13][10]=0.046305;
	P[10][14]=-27.9869;
	P[14][10]=-27.9869;
	P[10][15]=2.90196;
	P[15][10]=2.90196;
	P[10][16]=5.47712;
	P[16][10]=5.47712;
	P[10][17]=-0.490196;
	P[17][10]=-0.490196;
	P[10][18]=0.0230481;
	P[18][10]=0.0230481;
	P[10][19]=-20.6471;
	P[19][10]=-20.6471;
	P[10][20]=0.915033;
	P[20][10]=0.915033;
	P[10][21]=-7.81699;
	P[21][10]=-7.81699;
	P[10][22]=-1.10458;
	P[22][10]=-1.10458;
	P[10][23]=0.0206851;
	P[23][10]=0.0206851;
	P[10][24]=-21.7255;
	P[24][10]=-21.7255;
	P[10][25]=-2.81699;
	P[25][10]=-2.81699;
	P[10][26]=3.32026;
	P[26][10]=3.32026;
	P[10][27]=-2.70588;
	P[27][10]=-2.70588;
	P[10][28]=0.00958616;
	P[28][10]=0.00958616;
	P[11][11]=88;
	P[11][12]=5.17647;
	P[12][11]=5.17647;
	P[11][13]=0.08195;
	P[13][11]=0.08195;
	P[11][14]=-26.5882;
	P[14][11]=-26.5882;
	P[11][15]=4.35294;
	P[15][11]=4.35294;
	P[11][16]=-1.17647;
	P[16][11]=-1.17647;
	P[11][17]=-1.41176;
	P[17][11]=-1.41176;
	P[11][18]=0.0317296;
	P[18][11]=0.0317296;
	P[11][19]=-22.7059;
	P[19][11]=-22.7059;
	P[11][20]=-0.470588;
	P[20][11]=-0.470588;
	P[11][21]=1.88235;
	P[21][11]=1.88235;
	P[11][22]=-0.352941;
	P[22][11]=-0.352941;
	P[11][23]=0.012086;
	P[23][11]=0.012086;
	P[11][24]=-25.5294;
	P[24][11]=-25.5294;
	P[11][25]=-9.05882;
	P[25][11]=-9.05882;
	P[11][26]=33.8824;
	P[26][11]=33.8824;
	P[11][27]=1.05882;
	P[27][11]=1.05882;
	P[11][28]=-0.0308351;
	P[28][11]=-0.0308351;
	P[12][12]=3.74183;
	P[12][13]=0.0230356;
	P[13][12]=0.0230356;
	P[12][14]=-14.7614;
	P[14][12]=-14.7614;
	P[12][15]=1.60784;
	P[15][12]=1.60784;
	P[12][16]=3.39869;
	P[16][12]=3.39869;
	P[12][17]=-0.637255;
	P[17][12]=-0.637255;
	P[12][18]=0.012893;
	P[18][12]=0.012893;
	P[12][19]=-10.2059;
	P[19][12]=-10.2059;
	P[12][20]=0.522876;
	P[20][12]=0.522876;
	P[12][21]=-3.95425;
	P[21][12]=-3.95425;
	P[12][22]=-0.761438;
	P[22][12]=-0.761438;
	P[12][23]=0.0100826;
	P[23][12]=0.0100826;
	P[12][24]=-10.2843;
	P[24][12]=-10.2843;
	P[12][25]=-0.454248;
	P[25][12]=-0.454248;
	P[12][26]=0.594771;
	P[26][12]=0.594771;
	P[12][27]=-1.17647;
	P[27][12]=-1.17647;
	P[12][28]=0.00229741;
	P[28][12]=0.00229741;
	P[13][13]=0.000245146;
	P[13][14]=-0.136085;
	P[14][13]=-0.136085;
	P[13][15]=0.0129843;
	P[15][13]=0.0129843;
	P[13][16]=0.0283463;
	P[16][13]=0.0283463;
	P[13][17]=-0.00180869;
	P[17][13]=-0.00180869;
	P[13][18]=0.000111556;
	P[18][13]=0.000111556;
	P[13][19]=-0.0971168;
	P[19][13]=-0.0971168;
	P[13][20]=0.00521851;
	P[20][13]=0.00521851;
	P[13][21]=-0.0349918;
	P[21][13]=-0.0349918;
	P[13][22]=-0.00572456;
	P[22][13]=-0.00572456;
	P[13][23]=8.73077e-05;
	P[23][13]=8.73077e-05;
	P[13][24]=-0.104676;
	P[24][13]=-0.104676;
	P[13][25]=-0.0117412;
	P[25][13]=-0.0117412;
	P[13][26]=-0.000405776;
	P[26][13]=-0.000405776;
	P[13][27]=-0.0154329;
	P[27][13]=-0.0154329;
	P[13][28]=7.78719e-05;
	P[28][13]=7.78719e-05;
	P[14][14]=126.369;
	P[14][15]=-11.7255;
	P[15][14]=-11.7255;
	P[14][16]=-45.1242;
	P[16][14]=-45.1242;
	P[14][17]=0.401961;
	P[17][14]=0.401961;
	P[14][18]=-0.0913249;
	P[18][14]=-0.0913249;
	P[14][19]=76.0882;
	P[19][14]=76.0882;
	P[14][20]=-4.79739;
	P[20][14]=-4.79739;
	P[14][21]=55.1699;
	P[21][14]=55.1699;
	P[14][22]=5.01634;
	P[22][14]=5.01634;
	P[14][23]=-0.0907228;
	P[23][14]=-0.0907228;
	P[14][24]=90.5784;
	P[24][14]=90.5784;
	P[14][25]=12.3758;
	P[25][14]=12.3758;
	P[14][26]=-0.555556;
	P[26][14]=-0.555556;
	P[14][27]=12.8824;
	P[27][14]=12.8824;
	P[14][28]=-0.073355;
	P[28][14]=-0.073355;
	P[15][15]=1.29412;
	P[15][16]=3.13725;
	P[16][15]=3.13725;
	P[15][17]=-0.294118;
	P[17][15]=-0.294118;
	P[15][18]=0.00959501;
	P[18][15]=0.00959501;
	P[15][19]=-7.94118;
	P[19][15]=-7.94118;
	P[15][20]=0.27451;
	P[20][15]=0.27451;
	P[15][21]=-3.80392;
	P[21][15]=-3.80392;
	P[15][22]=-0.490196;
	P[22][15]=-0.490196;
	P[15][23]=0.00838061;
	P[23][15]=0.00838061;
	P[15][24]=-8.94118;
	P[24][15]=-8.94118;
	P[15][25]=-1.39216;
	P[25][15]=-1.39216;
	P[15][26]=2.31373;
	P[26][15]=2.31373;
	P[15][27]=-1.11765;
	P[27][15]=-1.11765;
	P[15][28]=0.00304013;
	P[28][15]=0.00304013;
	P[16][16]=46.8497;
	P[16][17]=3.56863;
	P[17][16]=3.56863;
	P[16][18]=0.0186734;
	P[18][16]=0.0186734;
	P[16][19]=3.52941;
	P[19][16]=3.52941;
	P[16][20]=4.24837;
	P[20][16]=4.24837;
	P[16][21]=-49.6209;
	P[21][16]=-49.6209;
	P[16][22]=-2.30065;
	P[22][16]=-2.30065;
	P[16][23]=0.0173756;
	P[23][16]=0.0173756;
	P[16][24]=-20.0784;
	P[24][16]=-20.0784;
	P[16][25]=-6.3268;
	P[25][16]=-6.3268;
	P[16][26]=3.86928;
	P[26][16]=3.86928;
	P[16][27]=-2.47059;
	P[27][16]=-2.47059;
	P[16][28]=0.024193;
	P[28][16]=0.024193;
	P[17][17]=0.735294;
	P[17][18]=-0.00256776;
	P[18][17]=-0.00256776;
	P[17][19]=3.97059;
	P[19][17]=3.97059;
	P[17][20]=0.431373;
	P[20][17]=0.431373;
	P[17][21]=-3.54902;
	P[21][17]=-3.54902;
	P[17][22]=0.107843;
	P[22][17]=0.107843;
	P[17][23]=-0.00245137;
	P[23][17]=-0.00245137;
	P[17][24]=1.55882;
	P[24][17]=1.55882;
	P[17][25]=-0.166667;
	P[25][17]=-0.166667;
	P[17][26]=-0.72549;
	P[26][17]=-0.72549;
	P[17][27]=0.117647;
	P[27][17]=0.117647;
	P[17][28]=0.00268172;
	P[28][17]=0.00268172;
	P[18][18]=7.82583e-05;
	P[18][19]=-0.0677176;
	P[19][18]=-0.0677176;
	P[18][20]=0.0018413;
	P[20][18]=0.0018413;
	P[18][21]=-0.0242665;
	P[21][18]=-0.0242665;
	P[18][22]=-0.0037999;
	P[22][18]=-0.0037999;
	P[18][23]=6.96975e-05;
	P[23][18]=6.96975e-05;
	P[18][24]=-0.0708356;
	P[24][18]=-0.0708356;
	P[18][25]=-0.00976234;
	P[25][18]=-0.00976234;
	P[18][26]=0.00819007;
	P[26][18]=0.00819007;
	P[18][27]=-0.00959279;
	P[27][18]=-0.00959279;
	P[18][28]=3.89867e-05;
	P[28][18]=3.89867e-05;
	P[19][19]=78.3824;
	P[19][20]=-0.117647;
	P[20][19]=-0.117647;
	P[19][21]=6.05882;
	P[21][19]=6.05882;
	P[19][22]=2.55882;
	P[22][19]=2.55882;
	P[19][23]=-0.0797819;
	P[23][19]=-0.0797819;
	P[19][24]=65.8529;
	P[24][19]=65.8529;
	P[19][25]=4.32353;
	P[25][19]=4.32353;
	P[19][26]=8.05882;
	P[26][19]=8.05882;
	P[19][27]=10.2353;
	P[27][19]=10.2353;
	P[19][28]=-0.049758;
	P[28][19]=-0.049758;
	P[20][20]=0.653595;
	P[20][21]=-4.57516;
	P[21][20]=-4.57516;
	P[20][22]=-0.20915;
	P[22][20]=-0.20915;
	P[20][23]=0.00113886;
	P[23][20]=0.00113886;
	P[20][24]=-2.45098;
	P[24][20]=-2.45098;
	P[20][25]=-0.339869;
	P[25][20]=-0.339869;
	P[20][26]=-1.00654;
	P[26][20]=-1.00654;
	P[20][27]=-0.411765;
	P[27][20]=-0.411765;
	P[20][28]=0.00497758;
	P[28][20]=0.00497758;
	P[21][21]=58.732;
	P[21][22]=2.69935;
	P[22][21]=2.69935;
	P[21][23]=-0.0421824;
	P[23][21]=-0.0421824;
	P[21][24]=26.2157;
	P[24][21]=26.2157;
	P[21][25]=5.49673;
	P[25][21]=5.49673;
	P[21][26]=2.57516;
	P[26][21]=2.57516;
	P[21][27]=4;
	P[27][21]=4;
	P[21][28]=-0.0346055;
	P[28][21]=-0.0346055;
	P[22][22]=0.486928;
	P[22][23]=-0.00381208;
	P[23][22]=-0.00381208;
	P[22][24]=3.10784;
	P[24][22]=3.10784;
	P[22][25]=0.25817;
	P[25][22]=0.25817;
	P[22][26]=-0.202614;
	P[26][22]=-0.202614;
	P[22][27]=0.647059;
	P[27][22]=0.647059;
	P[22][28]=-0.000898075;
	P[28][22]=-0.000898075;
	P[23][23]=0.000134753;
	P[23][24]=-0.0648897;
	P[24][23]=-0.0648897;
	P[23][25]=-0.00149784;
	P[25][23]=-0.00149784;
	P[23][26]=-0.0214536;
	P[26][23]=-0.0214536;
	P[23][27]=-0.011457;
	P[27][23]=-0.011457;
	P[23][28]=6.0391e-05;
	P[28][23]=6.0391e-05;
	P[24][24]=72.2647;
	P[24][25]=10.598;
	P[25][24]=10.598;
	P[24][26]=-2.60784;
	P[26][24]=-2.60784;
	P[24][27]=10.3529;
	P[27][24]=10.3529;
	P[24][28]=-0.0597272;
	P[28][24]=-0.0597272;
	P[25][25]=5.11438;
	P[25][26]=-8.30719;
	P[26][25]=-8.30719;
	P[25][27]=0.941176;
	P[27][25]=0.941176;
	P[25][28]=-0.00780906;
	P[28][25]=-0.00780906;
	P[26][26]=61.2418;
	P[26][27]=4;
	P[27][26]=4;
	P[26][28]=-0.0988842;
	P[28][26]=-0.0988842;
	P[27][27]=2.70588;
	P[27][28]=-0.0158912;
	P[28][27]=-0.0158912;
	P[28][28]=0.000281653;
	R[0][0]=51.5882;
	R[0][1]=1.35294;
	R[1][0]=1.35294;
	R[0][2]=71.5294;
	R[2][0]=71.5294;
	R[0][3]=-31.2353;
	R[3][0]=-31.2353;
	R[0][4]=61.5882;
	R[4][0]=61.5882;
	R[0][5]=-7.94118;
	R[5][0]=-7.94118;
	R[0][6]=74;
	R[6][0]=74;
	R[0][7]=-14.1176;
	R[7][0]=-14.1176;
	R[0][8]=52.9412;
	R[8][0]=52.9412;
	R[0][9]=-16.4118;
	R[9][0]=-16.4118;
	R[0][10]=-2.5277e+08;
	R[10][0]=-2.5277e+08;
	R[0][11]=-4.04355e+08;
	R[11][0]=-4.04355e+08;
	R[0][12]=5.07296e+07;
	R[12][0]=5.07296e+07;
	R[0][13]=-6.25807e+07;
	R[13][0]=-6.25807e+07;
	R[0][14]=-5.90283e+08;
	R[14][0]=-5.90283e+08;
	R[0][15]=-5.00497e+06;
	R[15][0]=-5.00497e+06;
	R[0][16]=261.294;
	R[16][0]=261.294;
	R[0][17]=298.588;
	R[17][0]=298.588;
	R[0][18]=1.68364e+08;
	R[18][0]=1.68364e+08;
	R[0][19]=-2.52433e+08;
	R[19][0]=-2.52433e+08;
	R[1][1]=1.35294;
	R[1][2]=1.17647;
	R[2][1]=1.17647;
	R[1][3]=0.411765;
	R[3][1]=0.411765;
	R[1][4]=1.52941;
	R[4][1]=1.52941;
	R[1][5]=0.882353;
	R[5][1]=0.882353;
	R[1][6]=0.470588;
	R[6][1]=0.470588;
	R[1][7]=0.117647;
	R[7][1]=0.117647;
	R[1][8]=0.411765;
	R[8][1]=0.411765;
	R[1][9]=0.352941;
	R[9][1]=0.352941;
	R[1][10]=-5.49985e+08;
	R[10][1]=-5.49985e+08;
	R[1][11]=-4.60814e+08;
	R[11][1]=-4.60814e+08;
	R[1][12]=-7.28036e+08;
	R[12][1]=-7.28036e+08;
	R[1][13]=-8.28015e+08;
	R[13][1]=-8.28015e+08;
	R[1][14]=4.38982e+07;
	R[14][1]=4.38982e+07;
	R[1][15]=-4.41615e+06;
	R[15][1]=-4.41615e+06;
	R[1][16]=189.176;
	R[16][1]=189.176;
	R[1][17]=259.294;
	R[17][1]=259.294;
	R[1][18]=-6.24241e+08;
	R[18][1]=-6.24241e+08;
	R[1][19]=-5.49687e+08;
	R[19][1]=-5.49687e+08;
	R[2][2]=107.882;
	R[2][3]=-49.1765;
	R[3][2]=-49.1765;
	R[2][4]=86.3529;
	R[4][2]=86.3529;
	R[2][5]=-11.3529;
	R[5][2]=-11.3529;
	R[2][6]=103;
	R[6][2]=103;
	R[2][7]=-19.5882;
	R[7][2]=-19.5882;
	R[2][8]=74.9412;
	R[8][2]=74.9412;
	R[2][9]=-24;
	R[9][2]=-24;
	R[2][10]=5.20063e+08;
	R[10][2]=5.20063e+08;
	R[2][11]=3.38755e+08;
	R[11][2]=3.38755e+08;
	R[2][12]=-1.27667e+08;
	R[12][2]=-1.27667e+08;
	R[2][13]=4.49943e+07;
	R[13][2]=4.49943e+07;
	R[2][14]=-5.20701e+08;
	R[14][2]=-5.20701e+08;
	R[2][15]=-3.53291e+06;
	R[15][2]=-3.53291e+06;
	R[2][16]=217.647;
	R[16][2]=217.647;
	R[2][17]=215.529;
	R[17][2]=215.529;
	R[2][18]=2.08014e+08;
	R[18][2]=2.08014e+08;
	R[2][19]=5.20302e+08;
	R[19][2]=5.20302e+08;
	R[3][3]=24;
	R[3][4]=-36.9412;
	R[4][3]=-36.9412;
	R[3][5]=5.23529;
	R[5][3]=5.23529;
	R[3][6]=-46.3529;
	R[6][3]=-46.3529;
	R[3][7]=9.11765;
	R[7][3]=9.11765;
	R[3][8]=-31.5294;
	R[8][3]=-31.5294;
	R[3][9]=10.6471;
	R[9][3]=10.6471;
	R[3][10]=-1.63506e+08;
	R[10][3]=-1.63506e+08;
	R[3][11]=-3.92372e+08;
	R[11][3]=-3.92372e+08;
	R[3][12]=4.16593e+07;
	R[12][3]=4.16593e+07;
	R[3][13]=1.49981e+07;
	R[13][3]=1.49981e+07;
	R[3][14]=2.47508e+08;
	R[14][3]=2.47508e+08;
	R[3][15]=-1.17764e+06;
	R[15][3]=-1.17764e+06;
	R[3][16]=20.2353;
	R[16][3]=20.2353;
	R[3][17]=64.9412;
	R[17][3]=64.9412;
	R[3][18]=6.93379e+07;
	R[18][3]=6.93379e+07;
	R[3][19]=-1.63426e+08;
	R[19][3]=-1.63426e+08;
	R[4][4]=82.8824;
	R[4][5]=-15.4706;
	R[5][4]=-15.4706;
	R[4][6]=86.1765;
	R[6][4]=86.1765;
	R[4][7]=-16.7647;
	R[7][4]=-16.7647;
	R[4][8]=66.1176;
	R[8][4]=66.1176;
	R[4][9]=-20.4118;
	R[9][4]=-20.4118;
	R[4][10]=-8.92645e+07;
	R[10][4]=-8.92645e+07;
	R[4][11]=-1.19832e+07;
	R[11][4]=-1.19832e+07;
	R[4][12]=-2.43575e+08;
	R[12][4]=-2.43575e+08;
	R[4][13]=-5.82869e+08;
	R[13][4]=-5.82869e+08;
	R[4][14]=-5.85146e+08;
	R[14][4]=-5.85146e+08;
	R[4][15]=-3.82732e+06;
	R[15][4]=-3.82732e+06;
	R[4][16]=217.353;
	R[16][4]=217.353;
	R[4][17]=229;
	R[17][4]=229;
	R[4][18]=-1.53619e+08;
	R[18][4]=-1.53619e+08;
	R[4][19]=-8.90065e+07;
	R[19][4]=-8.90065e+07;
	R[5][5]=6.76471;
	R[5][6]=-12.2941;
	R[6][5]=-12.2941;
	R[5][7]=3.05882;
	R[7][5]=3.05882;
	R[5][8]=-9.29412;
	R[8][5]=-9.29412;
	R[5][9]=3.76471;
	R[9][5]=3.76471;
	R[5][10]=-2.9734e+08;
	R[10][5]=-2.9734e+08;
	R[5][11]=-2.08169e+08;
	R[11][5]=-2.08169e+08;
	R[5][12]=-4.7539e+08;
	R[12][5]=-4.7539e+08;
	R[5][13]=-5.7537e+08;
	R[13][5]=-5.7537e+08;
	R[5][14]=2.96543e+08;
	R[14][5]=2.96543e+08;
	R[5][15]=-4.41615e+06;
	R[15][5]=-4.41615e+06;
	R[5][16]=182.059;
	R[16][5]=182.059;
	R[5][17]=258.529;
	R[17][5]=258.529;
	R[5][18]=-3.71596e+08;
	R[18][5]=-3.71596e+08;
	R[5][19]=-2.97042e+08;
	R[19][5]=-2.97042e+08;
	R[6][6]=115.471;
	R[6][7]=-23.0588;
	R[7][6]=-23.0588;
	R[6][8]=73.4706;
	R[8][6]=73.4706;
	R[6][9]=-25.2353;
	R[9][6]=-25.2353;
	R[6][10]=-1.04038e+08;
	R[10][6]=-1.04038e+08;
	R[6][11]=-9.8093e+07;
	R[11][6]=-9.8093e+07;
	R[6][12]=-6.21198e+08;
	R[12][6]=-6.21198e+08;
	R[6][13]=1.30072e+08;
	R[13][6]=1.30072e+08;
	R[6][14]=-3.17091e+08;
	R[14][6]=-3.17091e+08;
	R[6][15]=-294407;
	R[15][6]=-294407;
	R[6][16]=82.8824;
	R[16][6]=82.8824;
	R[6][17]=26.2941;
	R[17][6]=26.2941;
	R[6][18]=-1.08988e+08;
	R[18][6]=-1.08988e+08;
	R[6][19]=-1.04018e+08;
	R[19][6]=-1.04018e+08;
	R[7][7]=4.94118;
	R[7][8]=-13.0588;
	R[8][7]=-13.0588;
	R[7][9]=5;
	R[9][7]=5;
	R[7][10]=-2.08075e+08;
	R[10][7]=-2.08075e+08;
	R[7][11]=-1.96186e+08;
	R[11][7]=-1.96186e+08;
	R[7][12]=-2.31815e+08;
	R[12][7]=-2.31815e+08;
	R[7][13]=-2.45146e+08;
	R[13][7]=-2.45146e+08;
	R[7][14]=3.76399e+08;
	R[14][7]=3.76399e+08;
	R[7][15]=-588820;
	R[15][7]=-588820;
	R[7][16]=11.7059;
	R[16][7]=11.7059;
	R[7][17]=32.7059;
	R[17][7]=32.7059;
	R[7][18]=-2.17976e+08;
	R[18][7]=-2.17976e+08;
	R[7][19]=-2.08036e+08;
	R[19][7]=-2.08036e+08;
	R[8][8]=68.6471;
	R[8][9]=-20.1765;
	R[9][8]=-20.1765;
	R[8][10]=-1.33834e+08;
	R[10][8]=-1.33834e+08;
	R[8][11]=-3.21088e+08;
	R[11][8]=-3.21088e+08;
	R[8][12]=-5.1705e+08;
	R[12][8]=-5.1705e+08;
	R[8][13]=-8.50778e+07;
	R[13][8]=-8.50778e+07;
	R[8][14]=5.54325e+08;
	R[14][8]=5.54325e+08;
	R[8][15]=-3.23851e+06;
	R[15][8]=-3.23851e+06;
	R[8][16]=184;
	R[16][8]=184;
	R[8][17]=191.588;
	R[17][8]=191.588;
	R[8][18]=-4.40933e+08;
	R[18][8]=-4.40933e+08;
	R[8][19]=-1.33616e+08;
	R[19][8]=-1.33616e+08;
	R[9][9]=8.23529;
	R[9][10]=-4.60721e+08;
	R[10][9]=-4.60721e+08;
	R[9][11]=-4.48831e+08;
	R[11][9]=-4.48831e+08;
	R[9][12]=-2.31815e+08;
	R[12][9]=-2.31815e+08;
	R[9][13]=-2.45146e+08;
	R[13][9]=-2.45146e+08;
	R[9][14]=3.76399e+08;
	R[14][9]=3.76399e+08;
	R[9][15]=-588820;
	R[15][9]=-588820;
	R[9][16]=10.5882;
	R[16][9]=10.5882;
	R[9][17]=34.2353;
	R[17][9]=34.2353;
	R[9][18]=-2.17976e+08;
	R[18][9]=-2.17976e+08;
	R[9][19]=-4.60681e+08;
	R[19][9]=-4.60681e+08;
	R[10][10]=-6.3782e+08;
	R[10][11]=-2.91114e+08;
	R[11][10]=-2.91114e+08;
	R[10][12]=1.84317e+08;
	R[12][10]=1.84317e+08;
	R[10][13]=-1.44833e+08;
	R[13][10]=-1.44833e+08;
	R[10][14]=-3.76197e+08;
	R[14][10]=-3.76197e+08;
	R[10][15]=-3.57074e+08;
	R[15][10]=-3.57074e+08;
	R[10][16]=-4.14652e+08;
	R[16][10]=-4.14652e+08;
	R[10][17]=3.55512e+07;
	R[17][10]=3.55512e+07;
	R[10][18]=3.85985e+08;
	R[18][10]=3.85985e+08;
	R[10][19]=4.62815e+06;
	R[19][10]=4.62815e+06;
	R[11][11]=-2.83505e+08;
	R[11][12]=-6.08358e+08;
	R[12][11]=-6.08358e+08;
	R[11][13]=-1.02163e+08;
	R[13][11]=-1.02163e+08;
	R[11][14]=4.5461e+08;
	R[14][11]=4.5461e+08;
	R[11][15]=-7.38683e+08;
	R[15][11]=-7.38683e+08;
	R[11][16]=-3.48759e+08;
	R[16][11]=-3.48759e+08;
	R[11][17]=-1.13173e+07;
	R[17][11]=-1.13173e+07;
	R[11][18]=-6.77209e+08;
	R[18][11]=-6.77209e+08;
	R[11][19]=4.6527e+08;
	R[19][11]=4.6527e+08;
	R[12][12]=2.91105e+08;
	R[12][13]=-2.93019e+08;
	R[13][12]=-2.93019e+08;
	R[12][14]=6.04494e+08;
	R[14][12]=6.04494e+08;
	R[12][15]=7.3089e+08;
	R[15][12]=7.3089e+08;
	R[12][16]=3.33475e+07;
	R[16][12]=3.33475e+07;
	R[12][17]=-2.73816e+08;
	R[17][12]=-2.73816e+08;
	R[12][18]=-4.82652e+08;
	R[18][12]=-4.82652e+08;
	R[12][19]=8.29971e+07;
	R[19][12]=8.29971e+07;
	R[13][13]=1.22786e+09;
	R[13][14]=3.24956e+08;
	R[14][13]=3.24956e+08;
	R[13][15]=8.47346e+08;
	R[15][13]=8.47346e+08;
	R[13][16]=2.3504e+08;
	R[16][13]=2.3504e+08;
	R[13][17]=2.76311e+08;
	R[17][13]=2.76311e+08;
	R[13][18]=9.73535e+07;
	R[18][13]=9.73535e+07;
	R[13][19]=-3.54385e+08;
	R[19][13]=-3.54385e+08;
	R[14][14]=7.59444e+08;
	R[14][15]=1.02081e+09;
	R[15][14]=1.02081e+09;
	R[14][16]=3.79347e+08;
	R[16][14]=3.79347e+08;
	R[14][17]=-6.23769e+07;
	R[17][14]=-6.23769e+07;
	R[14][18]=-4.22824e+08;
	R[18][14]=-4.22824e+08;
	R[14][19]=7.2494e+08;
	R[19][14]=7.2494e+08;
	R[15][15]=1.53396e+09;
	R[15][16]=-1.12936e+09;
	R[16][15]=-1.12936e+09;
	R[15][17]=-1.54918e+09;
	R[17][15]=-1.54918e+09;
	R[15][18]=-1.86561e+08;
	R[18][15]=-1.86561e+08;
	R[15][19]=3.21844e+08;
	R[19][15]=3.21844e+08;
	R[16][16]=48137.9;
	R[16][17]=65973.9;
	R[17][16]=65973.9;
	R[16][18]=3.02071e+08;
	R[18][16]=3.02071e+08;
	R[16][19]=-3.38495e+08;
	R[19][16]=-3.38495e+08;
	R[17][17]=90491.9;
	R[17][18]=-2.43463e+08;
	R[18][17]=-2.43463e+08;
	R[17][19]=1.40018e+08;
	R[19][17]=1.40018e+08;
	R[18][18]=-1.26606e+07;
	R[18][19]=1.82235e+08;
	R[19][18]=1.82235e+08;
	R[19][19]=-1.40648e+07;
}

}//: namespace KW_MAP
}//: namespace Processors
