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

KW_MAP::KW_MAP(const std::string & name) :
	Base::Component(name) {
	LOG(LTRACE) << "Hello KW_MAP\n";
	ileObrazkow = 0;
}

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
			stateToCharPoint();
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
		H[indexR + 4][indexC] = cosE;
	//	H[indexR + 4][indexC] = d * cosE;

		indexC += 1;
		H[indexR + 1][indexC] = 1;
		H[indexR + 3][indexC] = cosE;
		H[indexR + 4][indexC] = - sinE;
	//	H[indexR + 4][indexC] = - d * sinE;

		indexC += 1;

	}

	H[indexR][indexC] = 1;
	H[indexR + 2][indexC] = 0.5 * cosE;
	H[indexR + 4][indexC] = -0.5 * sinE;
	//H[indexR + 4][indexC] = -0.5 * c * sinE;

	indexC += 1;
	H[indexR + 1][indexC] = 1;
	H[indexR + 2][indexC] = -0.5 * sinE;
	//H[indexR + 4][indexC] = -0.5 * c * cosE;
	H[indexR + 4][indexC] = -0.5 * cosE;

	if(sig == 1)
	{
		indexC += 1;
		H[indexR][indexC] = 1;
		H[indexR + 2][indexC] = cosE;
		H[indexR + 3][indexC] = sinE;
		H[indexR + 4][indexC] = - sinE + cosE;
	//	H[indexR + 4][indexC] = - c * sinE + d * cosE;


		indexC += 1;
		H[indexR + 1][indexC] = 1;
		H[indexR + 2][indexC] = - sinE;
		H[indexR + 3][indexC] = cosE;
		H[indexR + 4][indexC] = - cosE - sinE;
	//	H[indexR + 4][indexC] = - c * cosE - d * sinE;

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
       //  cout<<"nowy state"<<state[i]<<"\n";
     }
 }




}//: namespace KW_MAP
}//: namespace Processors
