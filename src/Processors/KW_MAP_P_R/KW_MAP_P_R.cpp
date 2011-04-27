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
	nrStates = 29;
	nrChar = 20;
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

	for( int i = 0; i < 29; i++)
	{
		pMean[i] = 0;
	}

	for (int i = 0; i < 20; i++)
	{
		rMean[i] = 0;
	}

	first = true;

	return true;
}

bool KW_MAP_P_R::onFinish() {
	LOG(LTRACE) << "KW_MAP_P_R::finish\n";

/*
	for (unsigned int i = 0; i < 29; i++)
    {
		pMean[i] = pMean[i]/ileObrazkow;
		cout<<"pMean["<< i <<"] = "<< pMean[i] <<"\n";
    }

	for (unsigned int i = 0; i < 20; i++)
    {
		rMean[i] = (int)(rMean[i]/ileObrazkow);
		//cout<<"rMean["<< i <<"] = "<< rMean[i] <<"\n";
    }
    */
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
		meanChar.clear();
		meanStates.clear();

		// wyznaczenie punktów charakterystycznych na aktualnym obrazku
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

		// id największego bloba czyli dłoni
		int id = 0;
		//numerElements - liczba punktów wchodzących w skład konturu
		unsigned int numerElements;
		//aktualny blob
		Types::Blobs::Blob *currentBlob;
		//blob o największej powierzchni czyli dłoń
		Types::Blobs::BlobResult result;
		//kontur bloba
		CvSeq * contour;
		//reader bloba
		CvSeqReader reader;
		//punkt, na którym akltualnie jest przeprowadzana operacja
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
		//kontener przechowujący elementy, które mozna narysować
		Types::DrawableContainer signs;
		// momenty służace do obliczenia środka ciężkości
		double m00, m10, m01;
		//powierzchnia bloba, powierzchnia największego bloba, współrzedne punktu środka ciezkości, największa wartośc współrzędnej Y
		double Area, MaxArea, CenterOfGravity_x, CenterOfGravity_y, MaxY;

		MaxArea = 0;
		MaxY = 0;

		//największy blob to dłoń
		for (int i = 0; i < blobs.GetNumBlobs(); i++) {
			currentBlob = blobs.GetBlob(i);

			Area = currentBlob->Area();
			// szukanie bloba o najwiekszej powierzchni
			if (Area > MaxArea) {
				MaxArea = Area;
				// id największego bloba, czyli dłoni
				id = i;
			}
		}

		//obliczenia tylko dla najwiekszego blobu, czyli dloni
		currentBlob = blobs.GetBlob(id);
		//wyznaczenie punktów znajdujących się na konturze bloba
		contour = currentBlob->GetExternalContour()->GetContourPoints();
		cvStartReadSeq(contour, &reader);

		for (int j = 0; j < contour->total; j = j + 1) {
			CV_READ_SEQ_ELEM( actualPoint, reader);

			if (j % 10 == 1) {
				//plik << actualPoint.x << " " << actualPoint.y << std::endl;
				contourPoints.push_back(cv::Point(actualPoint.x, actualPoint.y));
				//wyznaczenie największej wartości współrzędnej Y
				if (actualPoint.y > MaxY) {
					MaxY = actualPoint.y;
				}
			}
		}

		// calculate moments
		m00 = currentBlob->Moment(0, 0);
		m01 = currentBlob->Moment(0, 1);
		m10 = currentBlob->Moment(1, 0);

		//środek cięzkości
		CenterOfGravity_x = m10 / m00;
		CenterOfGravity_y = m01 / m00;

		// obliczenie wartości minimalej odległości pomiedzy środkiem ciezkości a punktem na konturze, która bedzie określać czy dany punkt bedzie brany pod uwagę podczas szukania ekstremów (punktów charakterystycznych)
		MINDIST = (MaxY - CenterOfGravity_y) * (MaxY - CenterOfGravity_y) * 4/ 9;
		//przesuniety punkt środka ciężkości, pierwszy punkt charakterystyczny
		charPoint.push_back(cv::Point(CenterOfGravity_x, CenterOfGravity_y + (MaxY - CenterOfGravity_y) * 4 / 5));

		// przesunięcie punktu ciezkości w celu ułatwienia wyznaczenie punktów charakterystycznych
		CenterOfGravity_y += (MaxY - CenterOfGravity_y) * 2 / 3;

		// określenie liczny punktów wchodzących w skład konturu
		numerElements = contourPoints.size();

		//******************************************************************
		//obliczenie roznicy miedzy punktami z obwodu a środkiem ciężkosci
		for (unsigned int i = 0; i < numerElements; i++) {
			TempDist = (contourPoints[i].x - CenterOfGravity_x)	* (contourPoints[i].x - CenterOfGravity_x)	+ (contourPoints[i].y - CenterOfGravity_y) * (contourPoints[i].y - CenterOfGravity_y);
			if (TempDist > MINDIST)
				dist.push_back(TempDist);
			else
				//jeśli odległość jest mniejsza niż MINDIST oznacza to, że jest to dolna cześć dłoni i nie znajdują się tam żadnego punkty charakterystyczne poza przesuniętym środkiem ciężkości, dlatego te punkty można ominąć
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

		//1 -oznacza, że ostatni element z konturu należał do dolnej czesci dłoni
		lastMinDist = 0;
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
			charPoint.push_back(cv::Point(contourPoints[indexPoint[i]].x, contourPoints[indexPoint[i]].y));
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

	fingerToState(charPoint[1], charPoint[2], 1);
	fingerToState(charPoint[3], charPoint[4], 1);
	fingerToState(charPoint[5], charPoint[6], 1);
	fingerToState(charPoint[7], charPoint[6], -1);
	fingerToState(charPoint[9], charPoint[8], -1);

	//zapamiętanie kolejnych wyznaczonych punktów charakterystycznych w tablicy nChar, sumowanie aktualnych punktów charakterystycznych w wektorze rMean
	for(unsigned int i = 0, j = 0; i < charPoint.size(); i++)
	{
		if (ileObrazkow <= 18)
		{
			//dodanie wartości aktualnych punktów charakterystycznych do tablicy rMean
			rMean[j] += charPoint[i].x;
			rMean[j+1] += charPoint[i].y;
		//	cout<<rMean[j]<<"\n";
		//	cout<<rMean[j+1]<<"\n";
		}

		//zapamiętanie kolejnych wyznaczonych punktów charakterystycznych w tablicy nChar
		nChar[j][ileObrazkow-1] =  charPoint[i].x;
		nChar[j+1][ileObrazkow-1] =  charPoint[i].y;
		j = j + 2;
	//	cout << "charPoint size: " << charPoint.size() << endl;
	}
	//zapamiętanie kolejnych wyznaczonych parametrów wektora stanu w tablicy nStates, sumowanie dotychczasownych wartości wektora stanu w wektorze pMean
	for(unsigned int i = 0; i < state.size(); i++)
	{
		if (ileObrazkow <= 18)
		{
			//dodanie aktualnych wartości parametrów wektra stanu do tablicy pMean
			pMean[i] += state[i];
			//zapamiętanie kolejnych wyznaczonych parametrów wektora stanu w tablicy nStates
			nStates[i][ileObrazkow-1] =  state[i];
		//	cout<<pMean[i]<<"\n";
		//	cout << "State size: " << state.size() << endl;
		}
	}

}

//punkcja obracająca punkt p o kąt angle według układu współrzędnych znajdującym się w punkcie p0
cv::Point KW_MAP_P_R::rot(cv::Point p, double angle, cv::Point p0) {
		cv::Point t;
		t.x = p0.x + (int) ((double) (p.x - p0.x) * cos(angle) - (double) (p.y - p0.y) * sin(angle));
		t.y = p0.y + (int) ((double) (p.x - p0.x) * sin(angle) + (double) (p.y - p0.y) * cos(angle));
		return t;
	}

//funkcja obliczająca parametry stanu na podstawie punktów charakterystycznych
//p2 - czubek palca, p1 - punkt miedzy palcami
void KW_MAP_P_R::fingerToState(cv::Point p2, cv::Point p1, int sig) {
	LOG(LTRACE) << "KW_MAP::fingerToState\n";

	//tg kąta nachylenia
	double dx = p2.x - charPoint[0].x;
	double dy = p2.y - charPoint[0].y;
	//argument kąta nachylenia
	double angle = atan2(dy, dx);
	double rotangle = angle + M_PI_2;

	//obrót punktów charakterystycznych o kąt nachylenia względem układu współrzędnych znajdującym się w punkcie dołu dłoni
	cv::Point pt1 = rot(p1, -rotangle, charPoint[0]);
	cv::Point pt2 = rot(p2, -rotangle, charPoint[0]);

	cv::Point statePoint;
//
//	if (sig == 1)
//		statePoint.x = pt2.x - (pt1.x - pt2.x);
//	else if (sig == -1)
//		statePoint.x = pt1.x;


	int width = abs(2 * (pt1.x - pt2.x));
	int height = abs(pt1.y - pt2.y);

	statePoint.x = pt2.x;
	statePoint.y = pt2.y + 0.5 * height;

	//obrót to poprzedniej pozycji
	statePoint = rot(statePoint, rotangle, charPoint[0]);

	//górny lewy wierzchołek
	state.push_back(statePoint.x);
	state.push_back(statePoint.y);
	//szerokosc
	state.push_back(width);
	//wysokosc
	state.push_back(height);
	state.push_back(angle);

	/*
	LOG(LTRACE) << "KW_MAP::fingerToState\n";

	double uj = (double) (-p2.x + charPoint[0].x) / (-p2.y + charPoint[0].y);
	double angle = atan(uj);
	cv::Point pt1 = rot(p1, angle, charPoint[0]);
	cv::Point pt2 = rot(p2, angle, charPoint[0]);

	cv::Point statePoint;

	if(sig == 1)
		statePoint.x = pt2.x - (pt1.x - pt2.x);
	else if (sig == -1)
		statePoint.x = pt1.x;

	statePoint.y = pt2.y;
	int width = abs(2 * (pt1.x - pt2.x));
	int height = abs(pt1.y - pt2.y);

	//obrót to poprzedniej pozycji
	angle = -angle;
	statePoint = rot(statePoint, angle, charPoint[0]);

	//górny lewy wierzchołek
	state.push_back(statePoint.x);
	state.push_back(statePoint.y);
	//szerokosc
	state.push_back(width);
	//wysokosc
	state.push_back(height);
	state.push_back(-angle);
	*/
}

/*!
 * Event handler function. wywołuje akcję obliczania P, invP, R, invR
 */
void KW_MAP_P_R::calculate()
{
	//zapisywanie macierzy P, invP, R, invR do pliku, tworzenie pliku
	std::ofstream plik("/home/kasia/Test.txt");

	//wyznaczenie średniej wartości parametrów stan dla dotychczasowych obrazków
	for(unsigned i = 0; i < nrStates; i++)
	{
		meanStates.push_back(pMean[i]/ileObrazkow);
		plik<<"pMean["<< i <<"] = "<< meanStates[i] <<"\n";
	}

	//wyznaczenie średniej  dla dotychczasowych obrazków
	for(unsigned i = 0; i < nrChar; i++)
	{
		meanChar.push_back(rMean[i]/ileObrazkow);

	//	cout<<"SIZE"<<charPoint.size()<<"\n";
	//	cout<<"meanChar"<<meanChar[i]<<"\n";
	}

	//*************Wyliczenie macierzy P i macierzy odwrotnej P
	//przepisanie średnich wartości stanów do macierzy

	cv::Mat PSamples(cv::Size(ileObrazkow, nrStates), CV_64FC1);

	plik<<"\n ";
	plik<<"PSamples\n ";
	for (unsigned int i = 0; i< nrStates; i++)
	{
		for(int j = 0; j< ileObrazkow; j++)
		{
			PSamples.at<float>(i,j) = nStates[i][j];
			plik<<setprecision(3)<<nStates[i][j]<<" \t";
			// plik do zapisu
			//std::ofstream plik("/home/kasia/Test.txt");

		}
		plik<<"\n";
	}


	cout<<"row"<<PSamples.rows<<"\n";
	cout<<"col"<<PSamples.cols<<"\n";

	cout<<nStates[0][0]<<"\n";
	cout<<PSamples.at<float>(0,0)<<"\n";
	cout<<nStates[1][0]<<"\n";
	cout<<PSamples.at<float>(1,0)<<"\n";
	cout<<nStates[0][3]<<"\n";
	cout<<PSamples.at<float>(0,3)<<"\n";
	cout<<nStates[19][13]<<"\n";
	cout<<PSamples.at<float>(19,13)<<"\n";

/*
	cv:: Mat mean;

	// obliczenie macierzy kowariancji P
	calcCovarMatrix(PSamples, P, mean, CV_COVAR_COLS| CV_COVAR_NORMAL);

	// podzial elementow macierzy P przez liczbe próbek - 1, w wyniku otrzymujemy macierz kowariancji
	for (int i = 0; i< P.rows; i++)
	{
		for(int j = 0; j< P.cols; j++)
			{
				P.at<double>(i,j) /= (ileObrazkow - 1);
				//plik<<"P["<<i<<"]["<<j<<"]="<<P.at<double>(i,j)<<";\n";
				//plik<<"P("<<i+1<<","<<j+1<<")="<<P.at<double>(i,j)<<";\t";
				//plik<<P.at<double>(i,j)<<";\t";
		}
		plik << "\n";
	}

	for (int i = 0; i< P.rows; i++)
	{
		for(int j = 0; j< P.cols; j++)
			{
				//plik<<"P["<<i<<"]["<<j<<"]="<<P.at<double>(i,j)<<";\n";
				//plik<<"MATP("<<j+1<<","<<i+1<<")="<<P.at<double>(i,j)<<";\n";
		}
	}
	cv::invert(P, invP, DECOMP_LU);
	for (int i = 0; i< invP.rows; i++)
	{
		for(int j = 0; j< invP.cols; j++)
			{
				//plik<<"invP["<<i<<"]["<<j<<"]="<<invP.at<double>(i,j)<<";\n";
			//plik<<invP.at<double>(i,j)<<";\t";
		}
		plik << "\n";
	}
*/
	//*************Wyliczenie macierzy R i macierzy odwrotnej R**************************************/

	cv::Mat RSamples(cv::Size(ileObrazkow, nrChar), CV_64FC1);

	plik<<"\n ";
	plik<<"RSamples\n ";
	for (unsigned int i = 0; i<  nrChar; i++)
	{
		for(int j = 0; j< ileObrazkow; j++)
		{
			RSamples.at<float>(i,j) = nChar[i][j];
			plik<<nChar[i][j]<<"\t ";
		}
		plik<<"\n ";
	}

	cout<<"row"<<RSamples.rows<<"\n";
	cout<<"col"<<RSamples.cols<<"\n";
cout<<nChar[0][0]<<"\n";
cout<<RSamples.at<float>(0,0)<<"\n";
cout<<nChar[1][0]<<"\n";
cout<<RSamples.at<float>(1,0)<<"\n";
cout<<nChar[0][3]<<"\n";
cout<<RSamples.at<float>(0,3)<<"\n";
cout<<nChar[19][13]<<"\n";
cout<<RSamples.at<float>(19,13)<<"\n";



/*
	cout<<"rows"<<PSamples.rows<<"\n";
	cout<<"cols"<<PSamples.cols<<"\n";

	cout<<"rows"<<P.rows<<"\n";
	cout<<"cols"<<P.cols<<"\n";

	cout<<"P"<<P.at<double>(0,0)<<"\n";

	for (int i = 0; i< P.rows; i++)
	{
		for(int j = 0; j< P.cols; j++)
		{
			P.at<double>(i,j) /= (ileObrazkow - 1);
		}
	}
	cout<<"P"<<P.at<double>(0,0)<<"\n";

	//najpierw ilosc kolumna potem wierszy
	cv::Size Test = Size(3,5);		//rozmiar obrazka
	cv:: Mat MatTest;
	MatTest.create(Test, CV_64FC1);		//8bitów, 0-255, 1 kanał
*/
/*
	MatTest.at<double>(0,0) = 4;
	MatTest.at<double>(0,1) = 2;
	MatTest.at<double>(0,2) = 0.6;
	MatTest.at<double>(1,0) = 4.2;
	MatTest.at<double>(1,1) = 2.1;
	MatTest.at<double>(1,2) = 0.59;
	MatTest.at<double>(2,0) = 3.9;
	MatTest.at<double>(2,1) = 2.0;
	MatTest.at<double>(2,2) = 0.58;
	MatTest.at<double>(3,0) = 4.3;
	MatTest.at<double>(3,1) = 2.1;
	MatTest.at<double>(3,2) = 0.62;
	MatTest.at<double>(4,0) = 4.1;
	MatTest.at<double>(4,1) = 2.2;
	MatTest.at<double>(4,2) = 0.63;

	cv:: Mat TestCov;
	calcCovarMatrix(MatTest,TestCov, mean, CV_COVAR_ROWS| CV_COVAR_NORMAL );
	cout<<"rows"<<TestCov.rows<<"\n";
	cout<<"cols"<<TestCov.cols<<"\n";

	for(unsigned int i = 0; i<3; i++)
	{
		for(unsigned int j = 0; j<3; j++)
		{
			//cout<<i<<","<<j<<": "<<TestCov.at<double>(i,j)/4<<"\n";
		}
	}
*/

	/*
	for(unsigned int i = 0; i< nrStates; i++)
	{
	  for(unsigned int j = i; j < nrStates ; j++)
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

	*/
/*
	for(unsigned int i = 0; i < nrStates; i++)
	{
		for(unsigned int j = 0; j < nrStates; j++)
		{
			 plik<<"P("<<i+1<<","<<j+1<<")="<<P[i][j]<<";\n";
		}
	}


	for(unsigned int i = 0; i < nrChar; i++)
	{
	  for(unsigned int j = i; j < nrChar ; j++)
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

	for(unsigned int i = 0; i < nrChar; i++)
	{
		for(unsigned int j = 0; j < nrChar; j++)
		{
			 plik<<"R["<<i<<"]["<<j<<"]="<<R[i][j]<<";\n";
		}
	}
	cout<<"MAMAMA2!\n";

	cv::Size sizeR = Size(nrChar,nrChar);		//rozmiar obrazka

	invR.create(sizeR, CV_32FC1);		//8bitów, 0-255, 1 kanał

	if (invR.isContinuous())   {
		sizeR.width *= sizeR.height;
		sizeR.height = 1;
	}

	//przepisanie macierzy R do macierzy invR
	for (int i = 0; i < sizeR.height; i++) {

			// when the arrays are continuous,
			// the outer loop is executed only once
			// if not - it's executed for each row

			// get pointer to beggining of i-th row of input image
			float* R_p = invR.ptr <float> (i);

			//oznacza, które wiersza jest aktualnie przepisywany
			int row = 0;
			unsigned int  col = 0;
			for(int j = 0 ; j < sizeR.width ; j++)
			{
				R_p[j] = R[row][col];
				col += 1;
				if(col == nrChar)
				{
					col = 0;
					row = row + 1;
				}
			}
	}


	cv::Mat inv;
	//odwracanie macierzy
	double d = cv::invert(invR, inv, DECOMP_LU);
	for(unsigned int i = 0; i < nrChar; i++)
	{
		for(unsigned int j = 0; j < nrChar; j++)
		{
			 plik<<"invR["<<i<<"]["<<j<<"]="<<inv.at<float>(i,j)<<";\n";
		}
	}
	plik << "d=" << d << "\n";

	cv::Size sizeP = Size(nrStates,nrStates);		//rozmiar obrazka

	invP.create(sizeP, CV_32FC1);		//8bitów, 0-255, 1 kanał

	if (invP.isContinuous())   {
		sizeP.width *= sizeP.height;
		sizeP.height = 1;
	}

	for (int i = 0; i < sizeP.height; i++) {

			// when the arrays are continuous,
			// the outer loop is executed only once
			// if not - it's executed for each row

			// get pointer to beggining of i-th row of input image
			float* P_p = invP.ptr <float> (i);

			//oznacza, które wiersza jest aktualnie przepisywany
			int row = 0;
			unsigned int  col = 0;
			for(int j = 0 ; j < sizeP.width ; j++)
			{
				P_p[j] = P[row][col];
				col += 1;
				if(col == nrStates)
				{
					col = 0;
					row = row + 1;
				}
			}
	}

	cv::Mat inv2;
	//odwracanie macierzy
	d = cv::invert(invP, inv2,  DECOMP_SVD);
	for(unsigned int i = 0; i<nrStates; i++)
	{
		for(unsigned int j = 0; j<nrStates; j++)
		{
			 plik<<"invP["<<i<<"]["<<j<<"]="<<inv2.at<float>(i,j)<<";\n";
	//		 cout<<"invP["<<i<<"]["<<j<<"]="<<inv2.at<float>(i,j)<<";\n";
		}
	}

	plik << "d=" << d << "\n";

	plik.close();
*/
}


}//: namespace KW_MAP_P_R
}//: namespace Processors
