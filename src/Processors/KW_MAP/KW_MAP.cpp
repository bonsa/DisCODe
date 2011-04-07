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
#include <vector>

namespace Processors {
namespace KW_MAP {

using namespace cv;

KW_MAP::KW_MAP(const std::string & name) : Base::Component(name)
{
	LOG(LTRACE) << "Hello KW_MAP\n";
}

KW_MAP::~KW_MAP()
{
	LOG(LTRACE) << "Good bye KW_MAP\n";
}



bool KW_MAP::onInit()
{
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

	return true;
}

bool KW_MAP::onFinish()
{
	LOG(LTRACE) << "KW_MAP::finish\n";

	return true;
}


bool KW_MAP::onStep()
{
	LOG(LTRACE) << "KW_MAP::step\n";

	blobs_ready = img_ready = false;

	try {

		getCharPoints();



		newImage->raise();

		return true;
	} catch (...) {
		LOG(LERROR) << "KW_MAP::getCharPoints failed\n";
		return false;
	}
}

bool KW_MAP::onStop()
{
	return true;
}

bool KW_MAP::onStart()
{
	return true;
}

void KW_MAP::onNewImage()
{
	LOG(LTRACE) << "KW_MAP::onNewImage\n";

	img_ready = true;
	tsl_img = in_img.read();
	//co robi tak linijka?
	tsl_img = tsl_img.clone();
	if (blobs_ready && img_ready)
		onStep();
}

void KW_MAP::onNewBlobs()
{
	LOG(LTRACE) << "KW_MAP::onNewBlobs\n";

	blobs_ready = true;
	blobs = in_blobs.read();
	if (blobs_ready && img_ready)
		onStep();
}


void KW_MAP::getCharPoints()
{

	LOG(LTRACE) << "KW_MAP::getCharPoints\n";

	try {

		int id = 0;
		//numerElements - liczba punktów wchodzących w skład konturu
		// i, ii - indeksy
		unsigned int numerElements ;
		std::ofstream plik("/home/kasia/Test.txt");
		Types::Blobs::Blob *currentBlob;
		Types::Blobs::BlobResult result;
		CvSeq * contour;
		CvSeqReader reader;
		CvPoint actualPoint;
		// wektor zawierający punkty konturu
		vector<CvPoint> contourPoints;
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

		Types::DrawableContainer drawcont;

		MaxArea = 0;
		MaxY = 0;

		//największy blob to dłoń
		for ( int i = 0; i < blobs.GetNumBlobs(); i++ )
		{
			currentBlob = blobs.GetBlob(i);

			Area = currentBlob->Area();
			if (Area > MaxArea)
			{
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
			for (int j = 0; j < contour->total; j=j+1)
			{
				CV_READ_SEQ_ELEM( actualPoint, reader);

				if (j%10 == 1) {
					//plik << actualPoint.x << " " << actualPoint.y << std::endl;
					contourPoints.push_back(cvPoint(actualPoint.x, actualPoint.y));
					if (actualPoint.y > MaxY)
					{
						MaxY = actualPoint.y;
					}
					cnt++;
				}
			}

			//środek cięzkości
			// calculate moments
			m00 = currentBlob->Moment(0,0);
			m01 = currentBlob->Moment(0,1);
			m10 = currentBlob->Moment(1,0);

			CenterOfGravity_x = m10/m00;
			CenterOfGravity_y = m01/m00;

			MINDIST = (MaxY-CenterOfGravity_y)*(MaxY-CenterOfGravity_y)*4/9;
			//przesuniety punkt środka ciężkości
			CenterOfGravity_y += (MaxY-CenterOfGravity_y)*2/3;
			charPoint.push_back(cvPoint(CenterOfGravity_x,CenterOfGravity_y));

			numerElements = contourPoints.size();

			//******************************************************************
			//obliczenie roznicy miedzy punktami z odwodu a środkiem ciężkosci
			for(unsigned int i = 0; i < numerElements; i++)
			{
				TempDist = (contourPoints[i].x - CenterOfGravity_x)*(contourPoints[i].x - CenterOfGravity_x)+(contourPoints[i].y - CenterOfGravity_y)*(contourPoints[i].y - CenterOfGravity_y);
				if(TempDist > MINDIST)
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

			for(unsigned int i=1; i < numerElements - 2; i++)
			{
				plik << dist[i] << "\n";
				derivative.push_back(dist[i+1]- dist[i]);

				if (dist[i+1] > MINDIST && dist[i]> MINDIST )
				{
					if (lastMinDist == 1)
					{
						lastSign = 1;
						lastMinDist = 0;
					}
					//maksiumum, funkcja rosła i zaczeła maleć
					if (derivative[i] < 0 && lastSign == 1)
					{
						indexPoint.push_back(i);
						lastSign = -1;
					}
					//minimum
					else if (derivative[i] > 0 && lastSign == -1)
					{
						indexPoint.push_back(i);
						lastSign = 1;
					}
				}
				else
				{
					lastMinDist = 1;
				}
			}

			Types::Ellipse * el = new Types::Ellipse(Point2f(CenterOfGravity_x, CenterOfGravity_y), Size2f(20,20));
			drawcont.add(el);
			Types::Ellipse * el2 = new Types::Ellipse(Point2f(last_x, last_y), Size2f(7,7));
			drawcont.add(el2);

			last_x = CenterOfGravity_x;
			last_y = CenterOfGravity_y;

			int idLeftPoint = 0;
			int xLeftPoint = 1000000;
			for (unsigned int i=0; i < indexPoint.size(); i++)
			{
				//znajdujemy punkt najbardziej wysynięty na lewo, czyli wierzchołek małego palca
				if (xLeftPoint > contourPoints[indexPoint[i]].x)
				{
					xLeftPoint = contourPoints[indexPoint[i]].x;
					idLeftPoint = i;
				}

			}

			for (int i=idLeftPoint; i >= 0; i--)
			{
				charPoint.push_back(cvPoint(contourPoints[indexPoint[i]].x, contourPoints[indexPoint[i]].y));
				drawcont.add(new Types::Ellipse(Point2f(contourPoints[indexPoint[i]].x, contourPoints[indexPoint[i]].y), Size2f(10,10)));
			}

			for (int i=indexPoint.size() - 1; i > idLeftPoint; i--)
			{
				charPoint.push_back(cvPoint(contourPoints[indexPoint[i]].x, contourPoints[indexPoint[i]].y));
				drawcont.add(new Types::Ellipse(Point2f(contourPoints[indexPoint[i]].x, contourPoints[indexPoint[i]].y), Size2f(10,10)));
			}

			//plik <<"Punkt środka cieżkosci: "<< CenterOfGravity_x <<" "<< CenterOfGravity_y;

		result.AddBlob(blobs.GetBlob(id));
		out_signs.write(result);
		out_draw.write(drawcont);

	} catch (...) {
		LOG(LERROR) << "KW_MAP::getCharPoints failed\n";

	}
}

CvPoint KW_MAP::rot(CvPoint p, double angle, CvPoint p0)
{
    CvPoint t;
    t.x = p0.x + (int)((double)(p.x - p0.x) * cos(angle) - (double)(p.y-p0.y) * sin(angle));
    t.y = p0.y + (int)((double)(p.x - p0.x) * sin(angle) + (double)(p.y-p0.y) * cos(angle));
    return t;
}


}//: namespace KW_MAP
}//: namespace Processors
