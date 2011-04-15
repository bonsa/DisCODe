/*!
 * \file KW_MAP_P_R.cpp
 * \brief Estymacja MAP, obliczanie macierzy kowariancji P i R
 * \author kwasak
 * \date 2011-04-14
 */

#include <memory>
#include <string>
#include <math.h> 

#include "KW_MAP_P_R.hpp"
#include "Logger.hpp"
#include "Types/Ellipse.hpp"
#include "Types/Line.hpp"
#include "Types/Rectangle.hpp"
#include <vector>
#include <iomanip>

namespace Processors {
namespace KW_MAP_P_R{

using namespace cv;

KW_MAP_P_R::KW_MAP_P_R(const std::string & name) :
	Base::Component(name) {
	LOG(LTRACE) << "Hello KW_MAP\n";
	ileObrazkow = 0;
}

KW_MAP_P_R::~KW_MAP_P_R() {
	LOG(LTRACE) << "Good bye KW_MAP\n";
}

bool KW_MAP_P_R::onInit() {
	LOG(LTRACE) << "KW_MAP::initialize\n";

	h_onNewImage.setup(this, &KW_MAP_P_R::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	h_onNewBlobs.setup(this, &KW_MAP_P_R::onNewBlobs);
	registerHandler("onNewBlobs", &h_onNewBlobs);

	h_calculate.setup(this, &KW_MAP_P_R::calculate);
	registerHandler("calculate", &h_calculate);

	registerStream("in_blobs", &in_blobs);
	registerStream("in_img", &in_img);

	newImage = registerEvent("newImage");

	registerStream("out_signs", &out_signs);
	registerStream("out_draw", &out_draw);

	learnRate = 0.01;

	for( int i = 0; i < 29; i++)
	{
		pMean[i] = 0;
		//cout<<pMean[i]<<"\n";
	}

	for (int i = 0; i < 20; i++)
	{
		rMean[i] = 0;
		//cout<<rMean[i]<<"\n";
	}

	first = true;

	return true;
}

bool KW_MAP_P_R::onFinish() {
	LOG(LTRACE) << "KW_MAP_P_R::finish\n";

	for (unsigned int i = 0; i < 29; i++)
    {
		pMean[i] = pMean[i]/ileObrazkow;
		//cout<<"pMean["<< i <<"] = "<< pMean[i] <<"\n";
    }

	for (unsigned int i = 0; i < 20; i++)
    {
		rMean[i] = (int)(rMean[i]/ileObrazkow);
		//cout<<"rMean["<< i <<"] = "<< rMean[i] <<"\n";
    }
/*
	for (unsigned int i = 0; i < 20; i++)
    {
		for(unsigned int j = 0; j <18; j++)
		{
			cout<<"nChar["<< i <<"]["<< j <<"] ="<< nChar[i][j] <<"\n";
		}
    }

	for (unsigned int i = 0; i < 29; i++)
    {
		for(unsigned int j = 0; j <18; j++)
		{
			cout<<"nStates["<< i <<"]["<< j <<"] ="<< nStates[i][j] <<"\n";
		}
    }
    */
	return true;
}

bool KW_MAP_P_R::onStep() {
	LOG(LTRACE) << "KW_MAP_P_R::step\n";

	blobs_ready = img_ready = false;


	try {
		ileObrazkow = ileObrazkow + 1;
		cout<<"ilosc obrazkow"<<ileObrazkow<<"\n" ;

		drawcont.clear();
		z.clear();
		charPoint.clear();
		diff.clear();
		state.clear();

		getCharPoints();
		 // z --> s, z pomiarów oblicza stan
		charPointsToState();

		out_draw.write(drawcont);
		newImage->raise();

		return true;
	} catch (...) {
		LOG(LERROR) << "KW_MAP::getCharPoints failed\n";
		return false;
	}
}

bool KW_MAP_P_R::onStop() {
	return true;
}

bool KW_MAP_P_R::onStart() {
	return true;
}

void KW_MAP_P_R::onNewImage() {
	LOG(LTRACE) << "KW_MAP_P_R::onNewImage\n";

	img_ready = true;
	tsl_img = in_img.read();
	//co robi tak linijka?
	tsl_img = tsl_img.clone();
	if (blobs_ready && img_ready)
		onStep();
}

void KW_MAP_P_R::onNewBlobs() {
	LOG(LTRACE) << "KW_MAP_P_R::onNewBlobs\n";

	blobs_ready = true;
	blobs = in_blobs.read();
	if (blobs_ready && img_ready)
		onStep();
}

void KW_MAP_P_R::getCharPoints() {

	LOG(LTRACE) << "KW_MAP_P_R::getCharPoints\n";

	try {

		int id = 0;
		//numerElements - liczba punktów wchodzących w skład konturu
		// i, ii - indeksy
		unsigned int numerElements;
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
		//id ostatenio wyznaczonego ekstremum
		int idLastExtreme;

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
		idLastExtreme = 0;
		if (derivative[0] > 0)
			lastSign = 1;
		else
			lastSign = -1;

		lastMinDist = 0;
		//pierwszy punkt kontury to wierzchołek punktu środkowego.
		indexPoint.push_back(0);

		for (unsigned int i = 1; i < numerElements - 2; i++) {

			derivative.push_back(dist[i + 1] - dist[i]);

			if (dist[i + 1] > MINDIST && dist[i] > MINDIST) {
				if (lastMinDist == 1) {
					lastSign = 1;
					lastMinDist = 0;
				}
				//maksiumum, funkcja rosła i zaczeła maleć
				if (derivative[i] < 0 && lastSign == 1) {
					if(((contourPoints[i].x - contourPoints[idLastExtreme].x)	* (contourPoints[i].x - contourPoints[idLastExtreme].x )	+ (contourPoints[i].y - contourPoints[idLastExtreme].y) * (contourPoints[i].y - contourPoints[idLastExtreme].y)) > 900)
					{
						indexPoint.push_back(i);
						lastSign = -1;
						idLastExtreme = i;
					}
				}
				//minimum
				else if (derivative[i] > 0 && lastSign == -1) {
					if(((contourPoints[i].x - contourPoints[idLastExtreme].x)	* (contourPoints[i].x - contourPoints[idLastExtreme].x )	+ (contourPoints[i].y - contourPoints[idLastExtreme].y) * (contourPoints[i].y - contourPoints[idLastExtreme].y)) > 900)
					{
						indexPoint.push_back(i);
						lastSign = 1;
						idLastExtreme = i;
					}
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
/*
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

	*/
		for (unsigned i = 0; i < charPoint.size(); i++)
		{
			drawcont.add(new Types::Ellipse(Point2f(charPoint[i].x, charPoint[i].y), Size2f(10,10)));
		}

		result.AddBlob(blobs.GetBlob(id));
		out_signs.write(result);

	} catch (...) {
		LOG(LERROR) << "KW_MAP::getCharPoints failed\n";

	}
}

void KW_MAP_P_R::charPointsToState() {
	LOG(LTRACE) << "KW_MAP_P_R::charPointsToState\n";

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

	for(unsigned int i = 0, j = 0; i < charPoint.size(); i++)
	{
		rMean[j] += charPoint[i].x;
		rMean[j+1] += charPoint[i].y;
		cout<<rMean[j]<<"\n";
		cout<<rMean[j+1]<<"\n";
		nChar[j][ileObrazkow-1] =  charPoint[i].x;
		nChar[j+1][ileObrazkow-1] =  charPoint[i].y;
		j = j + 2;
	//	cout << "charPoint size: " << charPoint.size() << endl;
	}

	for(unsigned int i = 0; i < state.size(); i++)
	{
		pMean[i] += state[i];
		nStates[i][ileObrazkow-1] =  state[i];
	//	cout<<pMean[i]<<"\n";
	//	cout << "State size: " << state.size() << endl;
	}

}

cv::Point KW_MAP_P_R::rot(cv::Point p, double angle, cv::Point p0) {
		cv::Point t;
		t.x = p0.x + (int) ((double) (p.x - p0.x) * cos(angle) - (double) (p.y - p0.y) * sin(angle));
		t.y = p0.y + (int) ((double) (p.x - p0.x) * sin(angle) + (double) (p.y - p0.y) * cos(angle));
		return t;
	}

//p2 - czubek palca, p1 - punkt miedzy palcami
void KW_MAP_P_R::fingerToState(cv::Point p2, cv::Point p1, int sig) {

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

void KW_MAP_P_R::calculate()
{
	cout<<"MAMAMA!\n";
	std::ofstream plik("/home/kasia/Test.txt");
	for(unsigned i = 0; i < state.size(); i++)
	{
		meanStates.push_back(pMean[i]/ileObrazkow);
	}

	for(unsigned i = 0; i < charPoint.size(); i++)
	{
		meanChar.push_back(rMean[i]/ileObrazkow);
	}

	for(int i = 0; i<29; i++)
	{
	  for(int j = i; j<29 ; j++)
	  {
		  P[i][j] = 0;
		  for(int k = 0; k < ileObrazkow; k++)
		  {
			  P[i][j] += (nStates[i][k]-meanStates[i])*(nStates[j][k]-meanStates[j]);
		  }

		  P[i][j] /= (ileObrazkow - 1);
	//	  plik<<"P["<<i<<"]["<<j<<"]="<<P[i][j]<<";\n";

		  if (i!=j)
		  {
			  //macierz kowariancji jest macierza symetryczna
			  P[j][i] = P[i][j];
		//  	  plik<<"P["<<j<<"]["<<i<<"]="<<P[j][i]<<";\n";
		  }
	  }
	}


	for(int i = 0; i<20; i++)
	{
	  for(int j = i; j<20 ; j++)
	  {
		  R[i][j] = 0;
		  for(int k = 0; k < ileObrazkow; k++)
		  {
			  R[i][j] += (nChar[i][k]-meanChar[i])*(nChar[j][k]-meanChar[j]);
		  }

		  R[i][j] /= (ileObrazkow - 1);
		  //plik<<"R["<<i<<"]["<<j<<"]="<<R[i][j]<<";\n";
		  if (i!=j)
		  {
			  //macierz kowariancji jest macierza symetryczna
			  R[j][i] = R[i][j];
			//  plik<<"R["<<j<<"]["<<i<<"]="<<R[j][i]<<";\n";
		  }
	  }
	}

	for(int i = 0; i<20; i++)
	{
		for(int j = 0; j<20; j++)
		{
			 plik<<"R["<<i<<"]["<<j<<"]="<<R[i][j]<<";\n";
		}
	}
	cout<<"MAMAMA2!\n";

	cv::Size sizeR = Size(20,20);		//rozmiar obrazka

	invR.create(sizeR, CV_32FC1);		//8bitów, 0-255, 1 kanał

	if (invR.isContinuous())   {
		sizeR.width *= sizeR.height;
		sizeR.height = 1;
	}

	for (int i = 0; i < sizeR.height; i++) {

			// when the arrays are continuous,
			// the outer loop is executed only once
			// if not - it's executed for each row

			// get pointer to beggining of i-th row of input image
			float* R_p = invR.ptr <float> (i);

			//oznacza, które wiersza jest aktualnie przepisywany
			int row = 0;
			int col = 0;
			for(int j = 0 ; j < sizeR.width ; j++)
			{
				R_p[j] = R[row][col];
				col += 1;
				if(col == 20)
				{
					col = 0;
					row = row + 1;
				}
			}
	}
	cv::Mat inv;
	cout <<"ok: "<<invR.at<float>(0,0)<<"\n";
	cout <<"OK: "<<invR.at<float>(19,19)<<"\n";
	cout <<"OK: "<<invR.at<float>(17,0)<<"\n";
	cout <<"OK: "<<invR.at<float>(11,2)<<"\n";

	//inv = invR.inv();
	//odwracanie macierzy
/*	cv::invert(invR, inv, DECOMP_LU);

	cout <<"ok: "<<inv.at<float>(0,0)<<"\n";
	cout <<"OK: "<<inv.at<float>(19,19)<<"\n";
	cout <<"OK: "<<inv.at<float>(17,0)<<"\n";
	cout <<"OK: "<<inv.at<float>(11,2)<<"\n";
*/
	//int row = invR.row();
//	cout<<"ROzmiar"<<row;
	//cout <<"OK: "<<invR.at<float>(13,19)<<"\n";
	plik.close();
}


}//: namespace KW_MAP_P_R
}//: namespace Processors
