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

	factor = 0.01;
	nrChar = 20;
	nrStates = 29;

	// czy warunek stopu jest spełniony
	STOP = false;

	//pierwsze uruchomienie komponentu
	first = true;

	return true;
}

bool KW_MAP::onFinish() {
	LOG(LTRACE) << "KW_MAP::finish\n";

	return true;
}

bool KW_MAP::onStep() {
	LOG(LTRACE) << "KW_MAP::step\n";

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

			getCharPoints();
			projectionMeasurePoints();

			if (first == true) {
				// char --> s, z pomiarów oblicza stan
				charPointsToState();
				first = false;
			} else {
				// s --> z
				stateToCharPoint();
				projectionEstimatedPoints();
				calculateDiff();
				updateState();
			}
		}
		projectionStates();
		stopCondition();

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

		// id największego bloba, czyli dłoni
		int id = 0;
		//numerElements - liczba punktów wchodzących w skład konturu
		unsigned int numerElements;
		// plik do zapisu
		std::ofstream plik("/home/kasia/Test.txt");
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
		double m00, m10, m01;
		//powierzchnia bloba, powiedzchnia największego bloba, współrzędne środka ciężkości, maksymalna wartośc współrzędnej Y
		double Area, MaxArea, CenterOfGravity_x, CenterOfGravity_y, MaxY;

		MaxArea = 0;
		MaxY = 0;

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
		//kontur największego bloba
		contour = currentBlob->GetExternalContour()->GetContourPoints();
		cvStartReadSeq(contour, &reader);

		for (int j = 0; j < contour->total; j = j + 1) {
			CV_READ_SEQ_ELEM( actualPoint, reader);

			if (j % 10 == 1) {
				//wpisanie punktów z konturu do wektora
				contourPoints.push_back(cv::Point(actualPoint.x, actualPoint.y));
				//szukanie max wartoścy y
				if (actualPoint.y > MaxY) {
					MaxY = actualPoint.y;
				}
			}
		}

		// calculate moments
		m00 = currentBlob->Moment(0, 0);
		m01 = currentBlob->Moment(0, 1);
		m10 = currentBlob->Moment(1, 0);

		//obliczenie środka cięzkości
		CenterOfGravity_x = m10 / m00;
		CenterOfGravity_y = m01 / m00;

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

		result.AddBlob(blobs.GetBlob(id));
		out_signs.write(result);

	} catch (...) {
		LOG(LERROR) << "KW_MAP::getCharPoints failed\n";

	}
}

//projekcja na obraz punktów charakterystycznych z aktualnego obrazka
void KW_MAP::projectionMeasurePoints() {
//	LOG(LTRACE) << "KW_MAP::projectionMeasurePoints\n";

	Types::Ellipse * el;
	for (unsigned int i = 0; i < nrChar / 2; i++) {
		el = new Types::Ellipse(Point2f(charPoint[i].x, charPoint[i].y),
				Size2f(10, 10));
		el->setCol(CV_RGB(0,0,255));
		drawcont.add(el);
	}

}

void KW_MAP::charPointsToState() {
//	LOG(LTRACE) << "KW_MAP::charPointsToState\n";

	//obliczanie parametrów prostokąta opisującego wewnętrzą część dłoni
	//wspolrzedna x lewego gornego punktu
	state.push_back(charPoint[0].x - (charPoint[8].x - charPoint[0].x));
	//wspolrzedna y lewego gornego punktu
	state.push_back(charPoint[6].y);
	//szerokosc
	state.push_back(abs(2 * (charPoint[8].x - charPoint[0].x)));
	//wysokosc
	state.push_back(abs(charPoint[0].y - charPoint[6].y));

	//tworzenie na podstawie punktów charekterystycznych parametrów stanu
	fingerToState(charPoint[1], charPoint[2], 1);
	fingerToState(charPoint[3], charPoint[4], 1);
	fingerToState(charPoint[5], charPoint[6], 1);
	fingerToState(charPoint[7], charPoint[6], -1);
	fingerToState(charPoint[9], charPoint[8], -1);
/*
	for (unsigned int i = 0; i < nrStates; i++) {
		cout << i << " states\t" << state[i] << "\n";
	}
*/
}

//punkcja obracająca punkt p o kąt angle według układu współrzędnych znajdującym się w punkcie p0
cv::Point KW_MAP::rot(cv::Point p, double angle, cv::Point p0) {
	cv::Point t;
	t.x = p0.x + (int) ((double) (p.x - p0.x) * cos(angle) - (double) (p.y
			- p0.y) * sin(angle));
	t.y = p0.y + (int) ((double) (p.x - p0.x) * sin(angle) + (double) (p.y
			- p0.y) * cos(angle));
	return t;
}

//funkcja obliczająca parametry stanu na podstawie punktów charakterystycznych
//p2 - czubek palca, p1 - punkt miedzy palcami
void KW_MAP::fingerToState(cv::Point p2, cv::Point p1, int sig) {

	LOG(LTRACE) << "KW_MAP::fingerToState\n";

	//tg kąta nachylenia
	double uj = (double) (-p2.x + charPoint[0].x) / (-p2.y + charPoint[0].y);
	//argument kąta nachylenia
	double angle = atan(uj);
	//obrót punktów charakterystycznych o kąt nachylenia względem układu współrzędnych znajdującym się w punkcie dołu dłoni
	cv::Point pt1 = rot(p1, angle, charPoint[0]);
	cv::Point pt2 = rot(p2, angle, charPoint[0]);

	// 4 punkty stanu opisują palec według chłopaków
	cv::Point statePoint;
	cv::Point statePoint2;
	cv::Point statePoint3;
	cv::Point statePoint4;

	if (sig == 1)
		statePoint.x = pt2.x - (pt1.x - pt2.x);
	else if (sig == -1)
		statePoint.x = pt1.x;

	statePoint.y = pt2.y;
	int width = abs(2 * (pt1.x - pt2.x));
	int height = abs(pt1.y - pt2.y);

	//obór to poprzedniej pozycji
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
}

//******************************************SPRAWDŹ CZY DZIALA***********************************************

//funkcja obliczajaca punkty charakterystyczne na podstawie wektora stanu
void KW_MAP::stateToFinger(double s1, double s2, double s3, double s4,
		double angle, int sig) {
	LOG(LTRACE) << "KW_MAP::stateToFinger\n";

	cv::Point tempPoint;

	if (sig == 1) {
		tempPoint.x = s1 + 0.5 * s3 * cos(angle);
		tempPoint.y = s2 - 0.5 * s3 * sin(angle);
		z.push_back(tempPoint);

		tempPoint.x = s1 + s3 * cos(angle) + s4 * sin(angle);
		tempPoint.y = s2 - s3 * sin(angle) + s4 * cos(angle);
		z.push_back(tempPoint);
	}
	if (sig == 2) {
		tempPoint.x = s1 + 0.5 * s3 * cos(angle);
		tempPoint.y = s2 - 0.5 * s3 * sin(angle);
		z.push_back(tempPoint);
	}
	if (sig == 3) {
		tempPoint.x = s1 + s4 * sin(angle);
		tempPoint.y = s2 + s4 * cos(angle);
		z.push_back(tempPoint);

		tempPoint.x = s1 + 0.5 * s3 * cos(angle);
		tempPoint.y = s2 - 0.5 * s3 * sin(angle);
		z.push_back(tempPoint);
	}
}

void KW_MAP::stateToCharPoint() {
	LOG(LTRACE) << "KW_MAP::stateToCharPoint\n";

	cv::Point rotPoint;
	cv::Point tempPoint;
	// punkt dołu dłoni
	z.push_back(cv::Point((state[0] + 0.5 * state[2]), (state[1] + state[3])));

	//punkty pierwszego palca od lewej
	stateToFinger(state[4], state[5], state[6], state[7], state[8], 1);
	//punkty drugiego palca od lewej
	stateToFinger(state[9], state[10], state[11], state[12], state[13], 1);
	//punkty środkowego palca
	stateToFinger(state[14], state[15], state[16], state[17], state[18], 1);
	//punkty czwartego palca od lewej
	stateToFinger(state[19], state[20], state[21], state[22], state[23], 2);
	//punkty kciuka
	stateToFinger(state[24], state[25], state[26], state[27], state[28], 3);

}
//projekcja na obraz estymowanych punktów charakterystycznych
void KW_MAP::projectionEstimatedPoints()
{
	//projekcja na obraz punktów charakterystycznych
	Types::Ellipse * elE;
	for (unsigned int i = 0; i < nrChar / 2; i++) {
		elE = new Types::Ellipse(Point2f(z[i].x, z[i].y), Size2f(20, 20));
		elE->setCol(CV_RGB(0,255,0));
		drawcont.add(elE);
	}

	//projekcia linii miedzy dołem dłoni a czubami palców
	Types::Line * elL;
	for (unsigned int i = 1; i < nrChar / 2; i += 2) {
		elL = new Types::Line(cv::Point(z[0].x, z[0].y), cv::Point(z[i].x,
				z[i].y));
		elL->setCol(CV_RGB(255,0,0));
		drawcont.add(elL);
	}
}

// obliczenie części jakobianu, macierzy H,
void KW_MAP::derivatives(int indexR, int indexC, double a, double b, double c,
		double d, double e, int sig) {
	double cosE = cos(e);
	double sinE = sin(e);

	if (sig == 3) {

		H[indexR][indexC] = 1;
		H[indexR + 3][indexC] = sinE;
		H[indexR + 4][indexC] = d * cosE;

		indexC += 1;
		H[indexR + 1][indexC] = 1;
		H[indexR + 3][indexC] = cosE;
		H[indexR + 4][indexC] = -d * sinE;

		indexC += 1;
	}

	H[indexR][indexC] = 1;
	H[indexR + 2][indexC] = 0.5 * cosE;
	H[indexR + 4][indexC] = -0.5 * c * sinE;

	indexC += 1;
	H[indexR + 1][indexC] = 1;
	H[indexR + 2][indexC] = -0.5 * sinE;
	H[indexR + 4][indexC] = -0.5 * c * cosE;

	if (sig == 1) {
		indexC += 1;
		H[indexR][indexC] = 1;
		H[indexR + 2][indexC] = cosE;
		H[indexR + 3][indexC] = sinE;
		H[indexR + 4][indexC] = -c * sinE + d * cosE;

		indexC += 1;
		H[indexR + 1][indexC] = 1;
		H[indexR + 2][indexC] = -sinE;
		H[indexR + 3][indexC] = cosE;
		H[indexR + 4][indexC] = -c * cosE - d * sinE;
	}
}

// funkcja obliczająca macierz jakobianu H
void KW_MAP::calculateH() {
	for (int i = 0; i < 29; i++) {
		for (int j = 0; j < 20; j++) {
			H[i][j] = 0;
		}
		//cout<<"\n";
	}
	H[0][0] = 1;
	H[2][0] = 0.5;
	H[1][1] = 1;
	H[3][1] = 1;

	derivatives(4, 2, state[4], state[5], state[6], state[7], state[8], 1);
	derivatives(9, 6, state[9], state[10], state[11], state[12], state[13], 1);
	derivatives(14, 10, state[14], state[15], state[16], state[17], state[18],
			1);
	derivatives(19, 14, state[19], state[20], state[21], state[22], state[23],
			2);
	derivatives(24, 16, state[24], state[25], state[26], state[27], state[28],
			3);

	//wyswietlanie macierzy H
	for (int i = 0; i < 29; i++) {
		for (int j = 0; j < 20; j++) {
			//cout << setprecision(3)<<H[i][j]<<"\t";
		}
		//cout<<"\n";
	}
}

//obliczenie o ile zmieni się wektor stanu
void KW_MAP::calculateDiff() {
	LOG(LTRACE) << "KW_MAP::calculateDiff\n";

	cout << "KW_MAP::calculateDiff\n";
	//różnica miedzy wektorami h(s) i z
	double D[20];
	unsigned j = 0;
	float error2 = 0;

	for (unsigned int i = 0; i < nrChar; i = i + 2)
	{
		//różnica miedzy punktami charakterystycznymi aktualnego obraz
		D[i] = z[j].x - charPoint[j].x;
		D[i + 1] = z[j].y - charPoint[j].y;
		j += 1;
	}

	//obliczenie macierzy jakobianu H
	calculateH();

	double t[nrChar];
	for (unsigned int i = 0; i < nrChar; i++) {
		t[i] = 0;
		for (unsigned int j = 0; j < nrChar; j++) {
			//t = iloraz odwrotnej macierzy R * roznica D
			t[i] += invR[i][j] * D[j];
		}
	}

	for (unsigned int i = 0; i < nrStates; i++) {
		t1[i] = 0;
		for (unsigned int j = 0; j < nrChar; j++) {
			//t1 = iloraz macierzy H * t
			t1[i] += H[i][j] * t[j];
		}
	}

	double t2[nrStates];
	for (unsigned int i = 0; i < nrStates; i++) {
		t2[i] = 0;
		for (unsigned int j = 0; j < nrStates; j++) {
			//mnożenie macierzy P * t1
			t2[i] += P[i][j] * t1[j];
		}
		diff.push_back(t2[i]);
		error2 += abs(t2[i]);
	}

	cout<<"ERROR2: "<<error2<<"\n";
	if(error2 < 3)
	{
		cout<< "STOP is true\n";
		STOP = true;
	}
}
// aktualizacja wektora stanu i macierzy P
void KW_MAP::updateState() {
	cout << "KW_MAP::updateState\n";

	for (unsigned int i = 0; i < nrStates; i++) {
		state[i] = state[i] - diff[i];
//		cout << i << " states\t" << state[i] << "\n";
	}

	for (unsigned int i = 0; i < nrStates; i++) {
		for (unsigned int j = 0; j < nrStates; j++) {
			P[i][j] *= (1 - factor);
		}
	}
}

void KW_MAP::stopCondition()
{
	cout << "KW_MAP::stopCondition()\n";
	for (unsigned int i = 0; i < nrStates; i++)
	{
		diffStates.push_back(state[i] - pMean[i]);
	}

	double t2[nrStates];
	for (unsigned int i = 0; i < nrStates; i++)
	{
		t2[i] = 0;
		for (unsigned int j = 0; j < nrStates; j++) {
			//mnożenie macierzy odwrotnej P * roznica S
			t2[i] += invP[i][j] * diffStates[j];
		}
	}

	double error = 0;
	for (unsigned int i = 0; i < nrStates; i++)
	{
		T.push_back(t1[i]+t2[i]);
		error += abs(T[i]);
	}

	//warunek końca estymacji MAP
	cout<<"ERROR: "<<error<<"\n";
	if(error < 0.00000000000001)
	{
		cout<< "STOP is true\n";
		STOP = true;
	}

}

//projekcja estymowanego stanu na obraz
void KW_MAP::projectionStates() {
	cv::Point statePoint1;
	cv::Point statePoint2;
	cv::Point statePoint3;
	cv::Point statePoint4;

	statePoint1.x = statePoint4.x = state[0];
	statePoint1.y = statePoint2.y = state[1];
	statePoint2.x = statePoint3.x = state[0] + state[2];
	statePoint4.y = statePoint3.y = state[1] + state[3];

	//projekcja środa dłoni na obraz
	drawcont.add(new Types::Line(cv::Point(statePoint1.x, statePoint1.y),
			cv::Point(statePoint2.x, statePoint2.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint2.x, statePoint2.y),
			cv::Point(statePoint3.x, statePoint3.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint3.x, statePoint3.y),
			cv::Point(statePoint4.x, statePoint4.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint4.x, statePoint4.y),
			cv::Point(statePoint1.x, statePoint1.y)));

	statePoint1.x = state[4];
	statePoint1.y = state[5];

	statePoint2.x = state[4] + state[6] * cos(state[8]);
	statePoint2.y = state[5] - state[6] * sin(state[8]);

	statePoint3.x = state[4] + state[6] * cos(state[8]) + state[7] * sin(
			state[8]);
	statePoint3.y = state[5] - state[6] * sin(state[8]) + state[7] * cos(
			state[8]);

	statePoint4.x = statePoint3.x - state[6] * cos(state[8]);
	statePoint4.y = statePoint3.y + state[6] * sin(state[8]);

	//projekcja lewego (małego) palca na obraz
	drawcont.add(new Types::Line(cv::Point(statePoint1.x, statePoint1.y),
			cv::Point(statePoint2.x, statePoint2.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint2.x, statePoint2.y),
			cv::Point(statePoint3.x, statePoint3.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint3.x, statePoint3.y),
			cv::Point(statePoint4.x, statePoint4.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint4.x, statePoint4.y),
			cv::Point(statePoint1.x, statePoint1.y)));

	statePoint1.x = state[9];
	statePoint1.y = state[10];

	statePoint2.x = state[9] + state[11] * cos(state[13]);
	statePoint2.y = state[10] - state[11] * sin(state[13]);

	statePoint3.x = state[9] + state[11] * cos(state[13]) + state[12] * sin(
			state[13]);
	statePoint3.y = state[10] - state[11] * sin(state[13]) + state[12] * cos(
			state[13]);

	statePoint4.x = statePoint3.x - state[11] * cos(state[13]);
	statePoint4.y = statePoint3.y + state[11] * sin(state[13]);

	//projekcja drugiego palca (wskazującego) od lewej na obraz
	drawcont.add(new Types::Line(cv::Point(statePoint1.x, statePoint1.y),
			cv::Point(statePoint2.x, statePoint2.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint2.x, statePoint2.y),
			cv::Point(statePoint3.x, statePoint3.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint3.x, statePoint3.y),
			cv::Point(statePoint4.x, statePoint4.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint4.x, statePoint4.y),
			cv::Point(statePoint1.x, statePoint1.y)));

	statePoint1.x = state[14];
	statePoint1.y = state[15];

	statePoint2.x = state[14] + state[16] * cos(state[18]);
	statePoint2.y = state[15] - state[16] * sin(state[18]);

	statePoint3.x = state[14] + state[16] * cos(state[18]) + state[17] * sin(
			state[18]);
	statePoint3.y = state[15] - state[16] * sin(state[18]) + state[17] * cos(
			state[18]);

	statePoint4.x = statePoint3.x - state[16] * cos(state[18]);
	statePoint4.y = statePoint3.y + state[16] * sin(state[18]);

	//projekcja środkowego palca od lewej na obraz
	drawcont.add(new Types::Line(cv::Point(statePoint1.x, statePoint1.y),
			cv::Point(statePoint2.x, statePoint2.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint2.x, statePoint2.y),
			cv::Point(statePoint3.x, statePoint3.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint3.x, statePoint3.y),
			cv::Point(statePoint4.x, statePoint4.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint4.x, statePoint4.y),
			cv::Point(statePoint1.x, statePoint1.y)));

	statePoint1.x = state[19];
	statePoint1.y = state[20];

	statePoint4.x = state[19] + state[22] * sin(state[23]);
	statePoint4.y = state[20] + state[22] * cos(state[23]);

	statePoint2.x = state[19] + state[21] * cos(state[23]);
	statePoint2.y = state[20] - state[21] * sin(state[23]);

	statePoint3.x = statePoint4.x + state[21] * cos(state[23]);
	statePoint3.y = statePoint4.y - state[21] * sin(state[23]);

	//projekcja drugiego palca od prawej
	drawcont.add(new Types::Line(cv::Point(statePoint1.x, statePoint1.y),
			cv::Point(statePoint2.x, statePoint2.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint2.x, statePoint2.y),
			cv::Point(statePoint3.x, statePoint3.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint3.x, statePoint3.y),
			cv::Point(statePoint4.x, statePoint4.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint4.x, statePoint4.y),
			cv::Point(statePoint1.x, statePoint1.y)));

	statePoint1.x = state[24];
	statePoint1.y = state[25];

	statePoint4.x = state[24] + state[27] * sin(state[28]);
	statePoint4.y = state[25] + state[27] * cos(state[28]);

	statePoint2.x = state[24] + state[26] * cos(state[28]);
	statePoint2.y = state[25] - state[26] * sin(state[28]);

	statePoint3.x = statePoint4.x + state[26] * cos(state[28]);
	statePoint3.y = statePoint4.y - state[26] * sin(state[28]);

	//projekcja pierwszego palca (kciuka) od prawej
	drawcont.add(new Types::Line(cv::Point(statePoint1.x, statePoint1.y),
			cv::Point(statePoint2.x, statePoint2.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint2.x, statePoint2.y),
			cv::Point(statePoint3.x, statePoint3.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint3.x, statePoint3.y),
			cv::Point(statePoint4.x, statePoint4.y)));
	drawcont.add(new Types::Line(cv::Point(statePoint4.x, statePoint4.y),
			cv::Point(statePoint1.x, statePoint1.y)));
}

//konstruktor
KW_MAP::KW_MAP(const std::string & name) :
	Base::Component(name) {
	LOG(LTRACE) << "Hello KW_MAP\n";
	ileObrazkow = 0;

	pMean[0] = 126.778;
	pMean[1] = 296.556;;
	pMean[2] = 158.333;
	pMean[3] = 172.278;
	pMean[4] = 20.5556;
	pMean[5] = 280.889;
	pMean[6] = 53.7778;
	pMean[7] = 117.889;
	pMean[8] = 0.677179;
	pMean[9] = 116.056;
	pMean[10] = 186.111;
	pMean[11] = 32;
	pMean[12] = 125.278;
	pMean[13] = 0.254681;
	pMean[14] = 153.389;
	pMean[15] = 163.333;
	pMean[16] = 60.4444;
	pMean[17] = 139.167;
	pMean[18] = 0.0754777;
	pMean[19] = 251.5;
	pMean[20] = 173.222;
	pMean[21] = 39.4444;
	pMean[22] = 127.389;
	pMean[23] = -0.219808;
	pMean[24] = 368.167;
	pMean[25] = 311.944;
	pMean[26] = 70.7778;
	pMean[27] = 102;
	pMean[28] = -0.959125;

	P[0][0] = 59.8301;
	P[0][1] = -0.575163;
	P[0][2] = -13.451;
	P[0][3] = 0.477124;
	P[0][4] = 74.8366;
	P[0][5] = -37.8497;
	P[0][6] = -6.75817;
	P[0][7] = 7.79739;
	P[0][8] = -0.141139;
	P[0][9] = 82.8366;
	P[0][10] = -18.5621;
	P[0][11] = -10.5882;
	P[0][12] = -9.40523;
	P[0][13] = -0.0921031;
	P[0][14] = 77.915;
	P[0][15] = -6.86275;
	P[0][16] = -14.719;
	P[0][17] = 1.33333;
	P[0][18] = -0.0577324;
	P[0][19] = 61.8235;
	P[0][20] = -2.35948;
	P[0][21] = 24.1046;
	P[0][22] = 2.85621;
	P[0][23] = -0.0699018;
	P[0][24] = 59.3922;
	P[0][25] = 3.51634;
	P[0][26] = 17.9477;
	P[0][27] = 10.1176;
	P[0][28] = -0.0672494;
	P[1][0] = -0.575163;
	P[1][1] = 0.379085;
	P[1][2] = 0.156863;
	P[1][3] = -0.163399;
	P[1][4] = -0.679739;
	P[1][5] = 0.535948;
	P[1][6] = 0.601307;
	P[1][7] = -0.0522876;
	P[1][8] = 1.09772e-05;
	P[1][9] = -0.973856;
	P[1][10] = 0.228758;
	P[1][11] = 0;
	P[1][12] = -0.163399;
	P[1][13] = 0.00155437;
	P[1][14] = -1.40523;
	P[1][15] = -0.0196078;
	P[1][16] = 1.97386;
	P[1][17] = 0.431373;
	P[1][18] = -0.000309872;
	P[1][19] = 0.764706;
	P[1][20] = 0.339869;
	P[1][21] = -2.02614;
	P[1][22] = 0.124183;
	P[1][23] = -0.000658766;
	P[1][24] = -0.686275;
	P[1][25] = -0.202614;
	P[1][26] = -0.575163;
	P[1][27] = -0.0588235;
	P[1][28] = 0.00323997;
	P[2][0] = -13.451;
	P[2][1] = 0.156863;
	P[2][2] = 17.0588;
	P[2][3] = 1.07843;
	P[2][4] = -1.13725;
	P[2][5] = -1.72549;
	P[2][6] = -14.9804;
	P[2][7] = -1.01961;
	P[2][8] = 0.0113829;
	P[2][9] = -0.137255;
	P[2][10] = 0.54902;
	P[2][11] = -14.1176;
	P[2][12] = 0.490196;
	P[2][13] = 0.00779605;
	P[2][14] = -3.19608;
	P[2][15] = -0.823529;
	P[2][16] = -2.7451;
	P[2][17] = 0.176471;
	P[2][18] = -0.00151393;
	P[2][19] = -8.76471;
	P[2][20] = 0.509804;
	P[2][21] = -2.15686;
	P[2][22] = -0.254902;
	P[2][23] = 0.0178924;
	P[2][24] = -0.294118;
	P[2][25] = 5.66667;
	P[2][26] = -28.5098;
	P[2][27] = -2.70588;
	P[2][28] = 0.0355193;
	P[3][0] = 0.477124;
	P[3][1] = -0.163399;
	P[3][2] = 1.07843;
	P[3][3] = 0.565359;
	P[3][4] = 0.836601;
	P[3][5] = -0.143791;
	P[3][6] = 0.300654;
	P[3][7] = 0.326797;
	P[3][8] = -0.00118527;
	P[3][9] = 1.98366;
	P[3][10] = -0.326797;
	P[3][11] = -1.17647;
	P[3][12] = 0.212418;
	P[3][13] = -0.00174907;
	P[3][14] = 2.76797;
	P[3][15] = -0.215686;
	P[3][16] = -2.36601;
	P[3][17] = -0.284314;
	P[3][18] = -0.00196699;
	P[3][19] = 0.0294118;
	P[3][20] = -0.124183;
	P[3][21] = 2.04575;
	P[3][22] = 0.00326797;
	P[3][23] = 0.000145284;
	P[3][24] = 2.0098;
	P[3][25] = 1.31046;
	P[3][26] = -1.28758;
	P[3][27] = 0.294118;
	P[3][28] = -0.00357112;
	P[4][0] = 74.8366;
	P[4][1] = -0.679739;
	P[4][2] = -1.13725;
	P[4][3] = 0.836601;
	P[4][4] = 134.967;
	P[4][5] = -78.5229;
	P[4][6] = -58.5752;
	P[4][7] = 8.53595;
	P[4][8] = -0.254026;
	P[4][9] = 124.026;
	P[4][10] = -26.8301;
	P[4][11] = -28.3529;
	P[4][12] = -14.6928;
	P[4][13] = -0.133181;
	P[4][14] = 110.007;
	P[4][15] = -11.4314;
	P[4][16] = -18.9673;
	P[4][17] = 3.01961;
	P[4][18] = -0.0869846;
	P[4][19] = 86.4118;
	P[4][20] = -2.95425;
	P[4][21] = 30.6797;
	P[4][22] = 3.83007;
	P[4][23] = -0.0919034;
	P[4][24] = 89.8431;
	P[4][25] = 10.8562;
	P[4][26] = -2.81046;
	P[4][27] = 13.8235;
	P[4][28] = -0.068696;
	P[5][0] = -37.8497;
	P[5][1] = 0.535948;
	P[5][2] = -1.72549;
	P[5][3] = -0.143791;
	P[5][4] = -78.5229;
	P[5][5] = 49.0458;
	P[5][6] = 44.7974;
	P[5][7] = -4.30719;
	P[5][8] = 0.147317;
	P[5][9] = -69.2876;
	P[5][10] = 15.1307;
	P[5][11] = 21.5294;
	P[5][12] = 7.56209;
	P[5][13] = 0.0733354;
	P[5][14] = -55.1895;
	P[5][15] = 6.09804;
	P[5][16] = 5.81699;
	P[5][17] = -1.80392;
	P[5][18] = 0.0448454;
	P[5][19] = -47;
	P[5][20] = 1.32026;
	P[5][21] = -13.0065;
	P[5][22] = -1.60131;
	P[5][23] = 0.0497395;
	P[5][24] = -47.6863;
	P[5][25] = -6.06536;
	P[5][26] = 6.44444;
	P[5][27] = -7.17647;
	P[5][28] = 0.0292512;
	P[6][0] = -6.75817;
	P[6][1] = 0.601307;
	P[6][2] = -14.9804;
	P[6][3] = 0.300654;
	P[6][4] = -58.5752;
	P[6][5] = 44.7974;
	P[6][6] = 74.7712;
	P[6][7] = 0.0915033;
	P[6][8] = 0.0913725;
	P[6][9] = -35.3987;
	P[6][10] = 6.02614;
	P[6][11] = 20;
	P[6][12] = 4.30065;
	P[6][13] = 0.0386039;
	P[6][14] = -17.3203;
	P[6][15] = 3.37255;
	P[6][16] = -6.48366;
	P[6][17] = -1.72549;
	P[6][18] = 0.0207086;
	P[6][19] = -20.8235;
	P[6][20] = -0.300654;
	P[6][21] = 5.28105;
	P[6][22] = 0.20915;
	P[6][23] = 0.0135383;
	P[6][24] = -21.9608;
	P[6][25] = -4.71895;
	P[6][26] = 28.183;
	P[6][27] = -1.88235;
	P[6][28] = -0.0261226;
	P[7][0] = 7.79739;
	P[7][1] = -0.0522876;
	P[7][2] = -1.01961;
	P[7][3] = 0.326797;
	P[7][4] = 8.53595;
	P[7][5] = -4.30719;
	P[7][6] = 0.0915033;
	P[7][7] = 2.33987;
	P[7][8] = -0.0146677;
	P[7][9] = 10.4771;
	P[7][10] = -2.1634;
	P[7][11] = -2.11765;
	P[7][12] = -0.379085;
	P[7][13] = -0.00850956;
	P[7][14] = 9.98693;
	P[7][15] = -0.901961;
	P[7][16] = -0.418301;
	P[7][17] = 0.196078;
	P[7][18] = -0.00828857;
	P[7][19] = 10.0588;
	P[7][20] = 0.0849673;
	P[7][21] = 2.40523;
	P[7][22] = 0.163399;
	P[7][23] = -0.0132319;
	P[7][24] = 8.43137;
	P[7][25] = 0.816993;
	P[7][26] = 1.26797;
	P[7][27] = 1.23529;
	P[7][28] = -0.00791758;
	P[8][0] = -0.141139;
	P[8][1] = 1.09772e-05;
	P[8][2] = 0.0113829;
	P[8][3] = -0.00118527;
	P[8][4] = -0.254026;
	P[8][5] = 0.147317;
	P[8][6] = 0.0913725;
	P[8][7] = -0.0146677;
	P[8][8] = 0.00053061;
	P[8][9] = -0.219402;
	P[8][10] = 0.0497243;
	P[8][11] = 0.0459302;
	P[8][12] = 0.025789;
	P[8][13] = 0.000231049;
	P[8][14] = -0.20024;
	P[8][15] = 0.0207764;
	P[8][16] = 0.0354618;
	P[8][17] = -0.00616154;
	P[8][18] = 0.000156116;
	P[8][19] = -0.159382;
	P[8][20] = 0.00575166;
	P[8][21] = -0.0645988;
	P[8][22] = -0.00793123;
	P[8][23] = 0.000187822;
	P[8][24] = -0.16393;
	P[8][25] = -0.0208618;
	P[8][26] = -0.016176;
	P[8][27] = -0.0288875;
	P[8][28] = 0.000176437;
	P[9][0] = 82.8366;
	P[9][1] = -0.973856;
	P[9][2] = -0.137255;
	P[9][3] = 1.98366;
	P[9][4] = 124.026;
	P[9][5] = -69.2876;
	P[9][6] = -35.3987;
	P[9][7] = 10.4771;
	P[9][8] = -0.219402;
	P[9][9] = 177.232;
	P[9][10] = -41.9477;
	P[9][11] = -83.7647;
	P[9][12] = -18.0163;
	P[9][13] = -0.195891;
	P[9][14] = 125.389;
	P[9][15] = -12.8431;
	P[9][16] = -22.732;
	P[9][17] = 2.57843;
	P[9][18] = -0.103651;
	P[9][19] = 94.4412;
	P[9][20] = -3.30719;
	P[9][21] = 30.9739;
	P[9][22] = 4.38889;
	P[9][23] = -0.0895337;
	P[9][24] = 99.5784;
	P[9][25] = 13.768;
	P[9][26] = -12.7516;
	P[9][27] = 12.2941;
	P[9][28] = -0.0548689;
	P[10][0] = -18.5621;
	P[10][1] = 0.228758;
	P[10][2] = 0.54902;
	P[10][3] = -0.326797;
	P[10][4] = -26.8301;
	P[10][5] = 15.1307;
	P[10][6] = 6.02614;
	P[10][7] = -2.1634;
	P[10][8] = 0.0497243;
	P[10][9] = -41.9477;
	P[10][10] = 10.5752;
	P[10][11] = 22.4706;
	P[10][12] = 4.14379;
	P[10][13] = 0.046305;
	P[10][14] = -27.9869;
	P[10][15] = 2.90196;
	P[10][16] = 5.47712;
	P[10][17] = -0.490196;
	P[10][18] = 0.0230481;
	P[10][19] = -20.6471;
	P[10][20] = 0.915033;
	P[10][21] = -7.81699;
	P[10][22] = -1.10458;
	P[10][23] = 0.0206851;
	P[10][24] = -21.7255;
	P[10][25] = -2.81699;
	P[10][26] = 3.32026;
	P[10][27] = -2.70588;
	P[10][28] = 0.00958616;
	P[11][0] = -10.5882;
	P[11][1] = 0;
	P[11][2] = -14.1176;
	P[11][3] = -1.17647;
	P[11][4] = -28.3529;
	P[11][5] = 21.5294;
	P[11][6] = 20;
	P[11][7] = -2.11765;
	P[11][8] = 0.0459302;
	P[11][9] = -83.7647;
	P[11][10] = 22.4706;
	P[11][11] = 88;
	P[11][12] = 5.17647;
	P[11][13] = 0.08195;
	P[11][14] = -26.5882;
	P[11][15] = 4.35294;
	P[11][16] = -1.17647;
	P[11][17] = -1.41176;
	P[11][18] = 0.0317296;
	P[11][19] = -22.7059;
	P[11][20] = -0.470588;
	P[11][21] = 1.88235;
	P[11][22] = -0.352941;
	P[11][23] = 0.012086;
	P[11][24] = -25.5294;
	P[11][25] = -9.05882;
	P[11][26] = 33.8824;
	P[11][27] = 1.05882;
	P[11][28] = -0.0308351;
	P[12][0] = -9.40523;
	P[12][1] = -0.163399;
	P[12][2] = 0.490196;
	P[12][3] = 0.212418;
	P[12][4] = -14.6928;
	P[12][5] = 7.56209;
	P[12][6] = 4.30065;
	P[12][7] = -0.379085;
	P[12][8] = 0.025789;
	P[12][9] = -18.0163;
	P[12][10] = 4.14379;
	P[12][11] = 5.17647;
	P[12][12] = 3.74183;
	P[12][13] = 0.0230356;
	P[12][14] = -14.7614;
	P[12][15] = 1.60784;
	P[12][16] = 3.39869;
	P[12][17] = -0.637255;
	P[12][18] = 0.012893;
	P[12][19] = -10.2059;
	P[12][20] = 0.522876;
	P[12][21] = -3.95425;
	P[12][22] = -0.761438;
	P[12][23] = 0.0100826;
	P[12][24] = -10.2843;
	P[12][25] = -0.454248;
	P[12][26] = 0.594771;
	P[12][27] = -1.17647;
	P[12][28] = 0.00229741;
	P[13][0] = -0.0921031;
	P[13][1] = 0.00155437;
	P[13][2] = 0.00779605;
	P[13][3] = -0.00174907;
	P[13][4] = -0.133181;
	P[13][5] = 0.0733354;
	P[13][6] = 0.0386039;
	P[13][7] = -0.00850956;
	P[13][8] = 0.000231049;
	P[13][9] = -0.195891;
	P[13][10] = 0.046305;
	P[13][11] = 0.08195;
	P[13][12] = 0.0230356;
	P[13][13] = 0.000245146;
	P[13][14] = -0.136085;
	P[13][15] = 0.0129843;
	P[13][16] = 0.0283463;
	P[13][17] = -0.00180869;
	P[13][18] = 0.000111556;
	P[13][19] = -0.0971168;
	P[13][20] = 0.00521851;
	P[13][21] = -0.0349918;
	P[13][22] = -0.00572456;
	P[13][23] = 8.73077e-05;
	P[13][24] = -0.104676;
	P[13][25] = -0.0117412;
	P[13][26] = -0.000405776;
	P[13][27] = -0.0154329;
	P[13][28] = 7.78719e-05;
	P[14][0] = 77.915;
	P[14][1] = -1.40523;
	P[14][2] = -3.19608;
	P[14][3] = 2.76797;
	P[14][4] = 110.007;
	P[14][5] = -55.1895;
	P[14][6] = -17.3203;
	P[14][7] = 9.98693;
	P[14][8] = -0.20024;
	P[14][9] = 125.389;
	P[14][10] = -27.9869;
	P[14][11] = -26.5882;
	P[14][12] = -14.7614;
	P[14][13] = -0.136085;
	P[14][14] = 126.369;
	P[14][15] = -11.7255;
	P[14][16] = -45.1242;
	P[14][17] = 0.401961;
	P[14][18] = -0.0913249;
	P[14][19] = 76.0882;
	P[14][20] = -4.79739;
	P[14][21] = 55.1699;
	P[14][22] = 5.01634;
	P[14][23] = -0.0907228;
	P[14][24] = 90.5784;
	P[14][25] = 12.3758;
	P[14][26] = -0.555556;
	P[14][27] = 12.8824;
	P[14][28] = -0.073355;
	P[15][0] = -6.86275;
	P[15][1] = -0.0196078;
	P[15][2] = -0.823529;
	P[15][3] = -0.215686;
	P[15][4] = -11.4314;
	P[15][5] = 6.09804;
	P[15][6] = 3.37255;
	P[15][7] = -0.901961;
	P[15][8] = 0.0207764;
	P[15][9] = -12.8431;
	P[15][10] = 2.90196;
	P[15][11] = 4.35294;
	P[15][12] = 1.60784;
	P[15][13] = 0.0129843;
	P[15][14] = -11.7255;
	P[15][15] = 1.29412;
	P[15][16] = 3.13725;
	P[15][17] = -0.294118;
	P[15][18] = 0.00959501;
	P[15][19] = -7.94118;
	P[15][20] = 0.27451;
	P[15][21] = -3.80392;
	P[15][22] = -0.490196;
	P[15][23] = 0.00838061;
	P[15][24] = -8.94118;
	P[15][25] = -1.39216;
	P[15][26] = 2.31373;
	P[15][27] = -1.11765;
	P[15][28] = 0.00304013;
	P[16][0] = -14.719;
	P[16][1] = 1.97386;
	P[16][2] = -2.7451;
	P[16][3] = -2.36601;
	P[16][4] = -18.9673;
	P[16][5] = 5.81699;
	P[16][6] = -6.48366;
	P[16][7] = -0.418301;
	P[16][8] = 0.0354618;
	P[16][9] = -22.732;
	P[16][10] = 5.47712;
	P[16][11] = -1.17647;
	P[16][12] = 3.39869;
	P[16][13] = 0.0283463;
	P[16][14] = -45.1242;
	P[16][15] = 3.13725;
	P[16][16] = 46.8497;
	P[16][17] = 3.56863;
	P[16][18] = 0.0186734;
	P[16][19] = 3.52941;
	P[16][20] = 4.24837;
	P[16][21] = -49.6209;
	P[16][22] = -2.30065;
	P[16][23] = 0.0173756;
	P[16][24] = -20.0784;
	P[16][25] = -6.3268;
	P[16][26] = 3.86928;
	P[16][27] = -2.47059;
	P[16][28] = 0.024193;
	P[17][0] = 1.33333;
	P[17][1] = 0.431373;
	P[17][2] = 0.176471;
	P[17][3] = -0.284314;
	P[17][4] = 3.01961;
	P[17][5] = -1.80392;
	P[17][6] = -1.72549;
	P[17][7] = 0.196078;
	P[17][8] = -0.00616154;
	P[17][9] = 2.57843;
	P[17][10] = -0.490196;
	P[17][11] = -1.41176;
	P[17][12] = -0.637255;
	P[17][13] = -0.00180869;
	P[17][14] = 0.401961;
	P[17][15] = -0.294118;
	P[17][16] = 3.56863;
	P[17][17] = 0.735294;
	P[17][18] = -0.00256776;
	P[17][19] = 3.97059;
	P[17][20] = 0.431373;
	P[17][21] = -3.54902;
	P[17][22] = 0.107843;
	P[17][23] = -0.00245137;
	P[17][24] = 1.55882;
	P[17][25] = -0.166667;
	P[17][26] = -0.72549;
	P[17][27] = 0.117647;
	P[17][28] = 0.00268172;
	P[18][0] = -0.0577324;
	P[18][1] = -0.000309872;
	P[18][2] = -0.00151393;
	P[18][3] = -0.00196699;
	P[18][4] = -0.0869846;
	P[18][5] = 0.0448454;
	P[18][6] = 0.0207086;
	P[18][7] = -0.00828857;
	P[18][8] = 0.000156116;
	P[18][9] = -0.103651;
	P[18][10] = 0.0230481;
	P[18][11] = 0.0317296;
	P[18][12] = 0.012893;
	P[18][13] = 0.000111556;
	P[18][14] = -0.0913249;
	P[18][15] = 0.00959501;
	P[18][16] = 0.0186734;
	P[18][17] = -0.00256776;
	P[18][18] = 7.82583e-05;
	P[18][19] = -0.0677176;
	P[18][20] = 0.0018413;
	P[18][21] = -0.0242665;
	P[18][22] = -0.0037999;
	P[18][23] = 6.96975e-05;
	P[18][24] = -0.0708356;
	P[18][25] = -0.00976234;
	P[18][26] = 0.00819007;
	P[18][27] = -0.00959279;
	P[18][28] = 3.89867e-05;
	P[19][0] = 61.8235;
	P[19][1] = 0.764706;
	P[19][2] = -8.76471;
	P[19][3] = 0.0294118;
	P[19][4] = 86.4118;
	P[19][5] = -47;
	P[19][6] = -20.8235;
	P[19][7] = 10.0588;
	P[19][8] = -0.159382;
	P[19][9] = 94.4412;
	P[19][10] = -20.6471;
	P[19][11] = -22.7059;
	P[19][12] = -10.2059;
	P[19][13] = -0.0971168;
	P[19][14] = 76.0882;
	P[19][15] = -7.94118;
	P[19][16] = 3.52941;
	P[19][17] = 3.97059;
	P[19][18] = -0.0677176;
	P[19][19] = 78.3824;
	P[19][20] = -0.117647;
	P[19][21] = 6.05882;
	P[19][22] = 2.55882;
	P[19][23] = -0.0797819;
	P[19][24] = 65.8529;
	P[19][25] = 4.32353;
	P[19][26] = 8.05882;
	P[19][27] = 10.2353;
	P[19][28] = -0.049758;
	P[20][0] = -2.35948;
	P[20][1] = 0.339869;
	P[20][2] = 0.509804;
	P[20][3] = -0.124183;
	P[20][4] = -2.95425;
	P[20][5] = 1.32026;
	P[20][6] = -0.300654;
	P[20][7] = 0.0849673;
	P[20][8] = 0.00575166;
	P[20][9] = -3.30719;
	P[20][10] = 0.915033;
	P[20][11] = -0.470588;
	P[20][12] = 0.522876;
	P[20][13] = 0.00521851;
	P[20][14] = -4.79739;
	P[20][15] = 0.27451;
	P[20][16] = 4.24837;
	P[20][17] = 0.431373;
	P[20][18] = 0.0018413;
	P[20][19] = -0.117647;
	P[20][20] = 0.653595;
	P[20][21] = -4.57516;
	P[20][22] = -0.20915;
	P[20][23] = 0.00113886;
	P[20][24] = -2.45098;
	P[20][25] = -0.339869;
	P[20][26] = -1.00654;
	P[20][27] = -0.411765;
	P[20][28] = 0.00497758;
	P[21][0] = 24.1046;
	P[21][1] = -2.02614;
	P[21][2] = -2.15686;
	P[21][3] = 2.04575;
	P[21][4] = 30.6797;
	P[21][5] = -13.0065;
	P[21][6] = 5.28105;
	P[21][7] = 2.40523;
	P[21][8] = -0.0645988;
	P[21][9] = 30.9739;
	P[21][10] = -7.81699;
	P[21][11] = 1.88235;
	P[21][12] = -3.95425;
	P[21][13] = -0.0349918;
	P[21][14] = 55.1699;
	P[21][15] = -3.80392;
	P[21][16] = -49.6209;
	P[21][17] = -3.54902;
	P[21][18] = -0.0242665;
	P[21][19] = 6.05882;
	P[21][20] = -4.57516;
	P[21][21] = 58.732;
	P[21][22] = 2.69935;
	P[21][23] = -0.0421824;
	P[21][24] = 26.2157;
	P[21][25] = 5.49673;
	P[21][26] = 2.57516;
	P[21][27] = 4;
	P[21][28] = -0.0346055;
	P[22][0] = 2.85621;
	P[22][1] = 0.124183;
	P[22][2] = -0.254902;
	P[22][3] = 0.00326797;
	P[22][4] = 3.83007;
	P[22][5] = -1.60131;
	P[22][6] = 0.20915;
	P[22][7] = 0.163399;
	P[22][8] = -0.00793123;
	P[22][9] = 4.38889;
	P[22][10] = -1.10458;
	P[22][11] = -0.352941;
	P[22][12] = -0.761438;
	P[22][13] = -0.00572456;
	P[22][14] = 5.01634;
	P[22][15] = -0.490196;
	P[22][16] = -2.30065;
	P[22][17] = 0.107843;
	P[22][18] = -0.0037999;
	P[22][19] = 2.55882;
	P[22][20] = -0.20915;
	P[22][21] = 2.69935;
	P[22][22] = 0.486928;
	P[22][23] = -0.00381208;
	P[22][24] = 3.10784;
	P[22][25] = 0.25817;
	P[22][26] = -0.202614;
	P[22][27] = 0.647059;
	P[22][28] = -0.000898075;
	P[23][0] = -0.0699018;
	P[23][1] = -0.000658766;
	P[23][2] = 0.0178924;
	P[23][3] = 0.000145284;
	P[23][4] = -0.0919034;
	P[23][5] = 0.0497395;
	P[23][6] = 0.0135383;
	P[23][7] = -0.0132319;
	P[23][8] = 0.000187822;
	P[23][9] = -0.0895337;
	P[23][10] = 0.0206851;
	P[23][11] = 0.012086;
	P[23][12] = 0.0100826;
	P[23][13] = 8.73077e-05;
	P[23][14] = -0.0907228;
	P[23][15] = 0.00838061;
	P[23][16] = 0.0173756;
	P[23][17] = -0.00245137;
	P[23][18] = 6.96975e-05;
	P[23][19] = -0.0797819;
	P[23][20] = 0.00113886;
	P[23][21] = -0.0421824;
	P[23][22] = -0.00381208;
	P[23][23] = 0.000134753;
	P[23][24] = -0.0648897;
	P[23][25] = -0.00149784;
	P[23][26] = -0.0214536;
	P[23][27] = -0.011457;
	P[23][28] = 6.0391e-05;
	P[24][0] = 59.3922;
	P[24][1] = -0.686275;
	P[24][2] = -0.294118;
	P[24][3] = 2.0098;
	P[24][4] = 89.8431;
	P[24][5] = -47.6863;
	P[24][6] = -21.9608;
	P[24][7] = 8.43137;
	P[24][8] = -0.16393;
	P[24][9] = 99.5784;
	P[24][10] = -21.7255;
	P[24][11] = -25.5294;
	P[24][12] = -10.2843;
	P[24][13] = -0.104676;
	P[24][14] = 90.5784;
	P[24][15] = -8.94118;
	P[24][16] = -20.0784;
	P[24][17] = 1.55882;
	P[24][18] = -0.0708356;
	P[24][19] = 65.8529;
	P[24][20] = -2.45098;
	P[24][21] = 26.2157;
	P[24][22] = 3.10784;
	P[24][23] = -0.0648897;
	P[24][24] = 72.2647;
	P[24][25] = 10.598;
	P[24][26] = -2.60784;
	P[24][27] = 10.3529;
	P[24][28] = -0.0597272;
	P[25][0] = 3.51634;
	P[25][1] = -0.202614;
	P[25][2] = 5.66667;
	P[25][3] = 1.31046;
	P[25][4] = 10.8562;
	P[25][5] = -6.06536;
	P[25][6] = -4.71895;
	P[25][7] = 0.816993;
	P[25][8] = -0.0208618;
	P[25][9] = 13.768;
	P[25][10] = -2.81699;
	P[25][11] = -9.05882;
	P[25][12] = -0.454248;
	P[25][13] = -0.0117412;
	P[25][14] = 12.3758;
	P[25][15] = -1.39216;
	P[25][16] = -6.3268;
	P[25][17] = -0.166667;
	P[25][18] = -0.00976234;
	P[25][19] = 4.32353;
	P[25][20] = -0.339869;
	P[25][21] = 5.49673;
	P[25][22] = 0.25817;
	P[25][23] = -0.00149784;
	P[25][24] = 10.598;
	P[25][25] = 5.11438;
	P[25][26] = -8.30719;
	P[25][27] = 0.941176;
	P[25][28] = -0.00780906;
	P[26][0] = 17.9477;
	P[26][1] = -0.575163;
	P[26][2] = -28.5098;
	P[26][3] = -1.28758;
	P[26][4] = -2.81046;
	P[26][5] = 6.44444;
	P[26][6] = 28.183;
	P[26][7] = 1.26797;
	P[26][8] = -0.016176;
	P[26][9] = -12.7516;
	P[26][10] = 3.32026;
	P[26][11] = 33.8824;
	P[26][12] = 0.594771;
	P[26][13] = -0.000405776;
	P[26][14] = -0.555556;
	P[26][15] = 2.31373;
	P[26][16] = 3.86928;
	P[26][17] = -0.72549;
	P[26][18] = 0.00819007;
	P[26][19] = 8.05882;
	P[26][20] = -1.00654;
	P[26][21] = 2.57516;
	P[26][22] = -0.202614;
	P[26][23] = -0.0214536;
	P[26][24] = -2.60784;
	P[26][25] = -8.30719;
	P[26][26] = 61.2418;
	P[26][27] = 4;
	P[26][28] = -0.0988842;
	P[27][0] = 10.1176;
	P[27][1] = -0.0588235;
	P[27][2] = -2.70588;
	P[27][3] = 0.294118;
	P[27][4] = 13.8235;
	P[27][5] = -7.17647;
	P[27][6] = -1.88235;
	P[27][7] = 1.23529;
	P[27][8] = -0.0288875;
	P[27][9] = 12.2941;
	P[27][10] = -2.70588;
	P[27][11] = 1.05882;
	P[27][12] = -1.17647;
	P[27][13] = -0.0154329;
	P[27][14] = 12.8824;
	P[27][15] = -1.11765;
	P[27][16] = -2.47059;
	P[27][17] = 0.117647;
	P[27][18] = -0.00959279;
	P[27][19] = 10.2353;
	P[27][20] = -0.411765;
	P[27][21] = 4;
	P[27][22] = 0.647059;
	P[27][23] = -0.011457;
	P[27][24] = 10.3529;
	P[27][25] = 0.941176;
	P[27][26] = 4;
	P[27][27] = 2.70588;
	P[27][28] = -0.0158912;
	P[28][0] = -0.0672494;
	P[28][1] = 0.00323997;
	P[28][2] = 0.0355193;
	P[28][3] = -0.00357112;
	P[28][4] = -0.068696;
	P[28][5] = 0.0292512;
	P[28][6] = -0.0261226;
	P[28][7] = -0.00791758;
	P[28][8] = 0.000176437;
	P[28][9] = -0.0548689;
	P[28][10] = 0.00958616;
	P[28][11] = -0.0308351;
	P[28][12] = 0.00229741;
	P[28][13] = 7.78719e-05;
	P[28][14] = -0.073355;
	P[28][15] = 0.00304013;
	P[28][16] = 0.024193;
	P[28][17] = 0.00268172;
	P[28][18] = 3.89867e-05;
	P[28][19] = -0.049758;
	P[28][20] = 0.00497758;
	P[28][21] = -0.0346055;
	P[28][22] = -0.000898075;
	P[28][23] = 6.0391e-05;
	P[28][24] = -0.0597272;
	P[28][25] = -0.00780906;
	P[28][26] = -0.0988842;
	P[28][27] = -0.0158912;
	P[28][28] = 0.000281653;

	invR[0][0] = 0.260714;
	invR[0][1] = -0.0729677;
	invR[0][2] = -0.205537;
	invR[0][3] = -0.308433;
	invR[0][4] = -0.0367005;
	invR[0][5] = -0.122784;
	invR[0][6] = 0.138836;
	invR[0][7] = 0.885158;
	invR[0][8] = -0.131774;
	invR[0][9] = -0.35644;
	invR[0][10] = -2.70255e-10;
	invR[0][11] = -1.21634e-09;
	invR[0][12] = 1.27938e-09;
	invR[0][13] = -3.69512e-10;
	invR[0][14] = 2.21011e-10;
	invR[0][15] = -2.41861e-10;
	invR[0][16] = -0.0767415;
	invR[0][17] = 0.085342;
	invR[0][18] = -7.78731e-10;
	invR[0][19] = -0.0277858;
	invR[1][0] = -0.0729677;
	invR[1][1] = 0.443132;
	invR[1][2] = -0.0279749;
	invR[1][3] = -0.232021;
	invR[1][4] = -0.190515;
	invR[1][5] = -0.211904;
	invR[1][6] = -0.220663;
	invR[1][7] = -0.93373;
	invR[1][8] = 0.249471;
	invR[1][9] = 0.163253;
	invR[1][10] = 1.29269e-10;
	invR[1][11] = -8.68691e-10;
	invR[1][12] = -1.91492e-09;
	invR[1][13] = 5.56391e-10;
	invR[1][14] = -1.98792e-09;
	invR[1][15] = -2.72453e-10;
	invR[1][16] = -0.0170554;
	invR[1][17] = 0.0191557;
	invR[1][18] = 1.00753e-09;
	invR[1][19] = -0.0063642;
	invR[2][0] = -0.205537;
	invR[2][1] = -0.0279749;
	invR[2][2] = 0.336433;
	invR[2][3] = 0.423332;
	invR[2][4] = 0.0255745;
	invR[2][5] = 0.151889;
	invR[2][6] = -0.118851;
	invR[2][7] = -0.389494;
	invR[2][8] = 0.034417;
	invR[2][9] = 0.268637;
	invR[2][10] = 1.01092e-09;
	invR[2][11] = -3.7475e-10;
	invR[2][12] = -1.49026e-09;
	invR[2][13] = 9.90954e-10;
	invR[2][14] = -1.13071e-09;
	invR[2][15] = 1.71105e-10;
	invR[2][16] = 0.117484;
	invR[2][17] = -0.130724;
	invR[2][18] = 5.11386e-10;
	invR[2][19] = 0.0426103;
	invR[3][0] = -0.308433;
	invR[3][1] = -0.232021;
	invR[3][2] = 0.423332;
	invR[3][3] = 0.77959;
	invR[3][4] = 0.163592;
	invR[3][5] = 0.372714;
	invR[3][6] = -0.0669591;
	invR[3][7] = -0.367149;
	invR[3][8] = -0.0102726;
	invR[3][9] = 0.264436;
	invR[3][10] = 1.33576e-09;
	invR[3][11] = -5.98948e-10;
	invR[3][12] = -1.51588e-09;
	invR[3][13] = 1.04943e-09;
	invR[3][14] = -1.28124e-09;
	invR[3][15] = 2.5583e-10;
	invR[3][16] = 0.175974;
	invR[3][17] = -0.195576;
	invR[3][18] = 6.69838e-10;
	invR[3][19] = 0.0635954;
	invR[4][0] = -0.0367005;
	invR[4][1] = -0.190515;
	invR[4][2] = 0.0255745;
	invR[4][3] = 0.163592;
	invR[4][4] = 0.123997;
	invR[4][5] = 0.0960225;
	invR[4][6] = 0.0866532;
	invR[4][7] = 0.232926;
	invR[4][8] = -0.084178;
	invR[4][9] = 0.00651101;
	invR[4][10] = 3.84292e-11;
	invR[4][11] = 7.25704e-10;
	invR[4][12] = 8.66852e-10;
	invR[4][13] = -1.0452e-09;
	invR[4][14] = 7.74612e-10;
	invR[4][15] = 3.64328e-10;
	invR[4][16] = 0.0149693;
	invR[4][17] = -0.0172666;
	invR[4][18] = -1.17879e-11;
	invR[4][19] = 0.00603964;
	invR[5][0] = -0.122784;
	invR[5][1] = -0.211904;
	invR[5][2] = 0.151889;
	invR[5][3] = 0.372714;
	invR[5][4] = 0.0960225;
	invR[5][5] = 0.356361;
	invR[5][6] = 0.00364136;
	invR[5][7] = -0.301026;
	invR[5][8] = -0.00866035;
	invR[5][9] = 0.126851;
	invR[5][10] = 9.85564e-11;
	invR[5][11] = 9.67646e-10;
	invR[5][12] = 1.25289e-09;
	invR[5][13] = -1.50986e-09;
	invR[5][14] = 6.94556e-10;
	invR[5][15] = 4.15014e-10;
	invR[5][16] = 0.0736715;
	invR[5][17] = -0.0797885;
	invR[5][18] = -8.78723e-11;
	invR[5][19] = 0.0245349;
	invR[6][0] = 0.138836;
	invR[6][1] = -0.220663;
	invR[6][2] = -0.118851;
	invR[6][3] = -0.0669591;
	invR[6][4] = 0.0866532;
	invR[6][5] = 0.00364136;
	invR[6][6] = 0.177143;
	invR[6][7] = 0.793473;
	invR[6][8] = -0.172507;
	invR[6][9] = -0.220149;
	invR[6][10] = 2.49895e-11;
	invR[6][11] = 1.1965e-10;
	invR[6][12] = -5.40203e-10;
	invR[6][13] = 4.43717e-10;
	invR[6][14] = 5.45666e-10;
	invR[6][15] = -1.74835e-10;
	invR[6][16] = -0.0423299;
	invR[6][17] = 0.0464282;
	invR[6][18] = -5.36357e-10;
	invR[6][19] = -0.0146808;
	invR[7][0] = 0.885158;
	invR[7][1] = -0.93373;
	invR[7][2] = -0.389494;
	invR[7][3] = -0.367149;
	invR[7][4] = 0.232926;
	invR[7][5] = -0.301026;
	invR[7][6] = 0.793473;
	invR[7][7] = 5.0059;
	invR[7][8] = -0.93462;
	invR[7][9] = -1.45298;
	invR[7][10] = -5.21157e-10;
	invR[7][11] = -3.13268e-10;
	invR[7][12] = -2.44468e-10;
	invR[7][13] = 9.20815e-10;
	invR[7][14] = 2.35677e-09;
	invR[7][15] = -7.29632e-10;
	invR[7][16] = -0.149667;
	invR[7][17] = 0.161423;
	invR[7][18] = -2.18291e-09;
	invR[7][19] = -0.0491724;
	invR[8][0] = -0.131774;
	invR[8][1] = 0.249471;
	invR[8][2] = 0.034417;
	invR[8][3] = -0.0102726;
	invR[8][4] = -0.084178;
	invR[8][5] = -0.00866036;
	invR[8][6] = -0.172507;
	invR[8][7] = -0.93462;
	invR[8][8] = 0.209207;
	invR[8][9] = 0.209648;
	invR[8][10] = -1.68871e-10;
	invR[8][11] = -1.61658e-11;
	invR[8][12] = 1.47377e-10;
	invR[8][13] = -6.25818e-11;
	invR[8][14] = 7.07223e-11;
	invR[8][15] = 1.08444e-10;
	invR[8][16] = 0.018088;
	invR[8][17] = -0.019131;
	invR[8][18] = 1.66264e-10;
	invR[8][19] = 0.00556499;
	invR[9][0] = -0.35644;
	invR[9][1] = 0.163253;
	invR[9][2] = 0.268637;
	invR[9][3] = 0.264436;
	invR[9][4] = 0.00651101;
	invR[9][5] = 0.126851;
	invR[9][6] = -0.220149;
	invR[9][7] = -1.45298;
	invR[9][8] = 0.209648;
	invR[9][9] = 0.644463;
	invR[9][10] = -3.89563e-10;
	invR[9][11] = 2.0587e-11;
	invR[9][12] = 2.77585e-10;
	invR[9][13] = 2.02637e-10;
	invR[9][14] = 3.20126e-10;
	invR[9][15] = 5.12157e-10;
	invR[9][16] = 0.0762791;
	invR[9][17] = -0.0845029;
	invR[9][18] = 6.62821e-11;
	invR[9][19] = 0.0272936;
	invR[10][0] = -2.70255e-10;
	invR[10][1] = 1.29269e-10;
	invR[10][2] = 1.01092e-09;
	invR[10][3] = 1.33576e-09;
	invR[10][4] = 3.84292e-11;
	invR[10][5] = 9.85564e-11;
	invR[10][6] = 2.49895e-11;
	invR[10][7] = -5.21157e-10;
	invR[10][8] = -1.68871e-10;
	invR[10][9] = -3.89563e-10;
	invR[10][10] = -3.28889e-18;
	invR[10][11] = 1.79091e-18;
	invR[10][12] = 3.77456e-18;
	invR[10][13] = -7.24207e-19;
	invR[10][14] = 3.10432e-18;
	invR[10][15] = 7.06775e-19;
	invR[10][16] = -5.20718e-10;
	invR[10][17] = 6.30165e-10;
	invR[10][18] = -1.89781e-18;
	invR[10][19] = -2.39626e-10;
	invR[11][0] = -1.21634e-09;
	invR[11][1] = -8.68691e-10;
	invR[11][2] = -3.7475e-10;
	invR[11][3] = -5.98948e-10;
	invR[11][4] = 7.25704e-10;
	invR[11][5] = 9.67646e-10;
	invR[11][6] = 1.1965e-10;
	invR[11][7] = -3.13268e-10;
	invR[11][8] = -1.61658e-11;
	invR[11][9] = 2.05869e-11;
	invR[11][10] = 1.79091e-18;
	invR[11][11] = -2.5585e-18;
	invR[11][12] = -1.01953e-18;
	invR[11][13] = -3.79723e-20;
	invR[11][14] = -1.48509e-18;
	invR[11][15] = -1.07072e-19;
	invR[11][16] = 4.68499e-10;
	invR[11][17] = -5.75426e-10;
	invR[11][18] = 1.8012e-18;
	invR[11][19] = 2.24052e-10;
	invR[12][0] = 1.27938e-09;
	invR[12][1] = -1.91492e-09;
	invR[12][2] = -1.49026e-09;
	invR[12][3] = -1.51588e-09;
	invR[12][4] = 8.66852e-10;
	invR[12][5] = 1.25289e-09;
	invR[12][6] = -5.40203e-10;
	invR[12][7] = -2.44468e-10;
	invR[12][8] = 1.47377e-10;
	invR[12][9] = 2.77585e-10;
	invR[12][10] = 3.77456e-18;
	invR[12][11] = -1.01953e-18;
	invR[12][12] = -7.50828e-18;
	invR[12][13] = 3.31032e-18;
	invR[12][14] = -4.83111e-18;
	invR[12][15] = -2.13136e-18;
	invR[12][16] = -3.6278e-10;
	invR[12][17] = 4.25325e-10;
	invR[12][18] = 1.38366e-18;
	invR[12][19] = -1.5324e-10;
	invR[13][0] = -3.69512e-10;
	invR[13][1] = 5.56391e-10;
	invR[13][2] = 9.90954e-10;
	invR[13][3] = 1.04943e-09;
	invR[13][4] = -1.0452e-09;
	invR[13][5] = -1.50986e-09;
	invR[13][6] = 4.43717e-10;
	invR[13][7] = 9.20815e-10;
	invR[13][8] = -6.25818e-11;
	invR[13][9] = 2.02637e-10;
	invR[13][10] = -7.24207e-19;
	invR[13][11] = -3.79723e-20;
	invR[13][12] = 3.31032e-18;
	invR[13][13] = -2.40602e-18;
	invR[13][14] = 1.69098e-18;
	invR[13][15] = 1.20144e-18;
	invR[13][16] = 4.30878e-10;
	invR[13][17] = -5.29486e-10;
	invR[13][18] = 1.45859e-19;
	invR[13][19] = 2.06328e-10;
	invR[14][0] = 2.21011e-10;
	invR[14][1] = -1.98792e-09;
	invR[14][2] = -1.13071e-09;
	invR[14][3] = -1.28124e-09;
	invR[14][4] = 7.74612e-10;
	invR[14][5] = 6.94556e-10;
	invR[14][6] = 5.45666e-10;
	invR[14][7] = 2.35677e-09;
	invR[14][8] = 7.07223e-11;
	invR[14][9] = 3.20126e-10;
	invR[14][10] = 3.10432e-18;
	invR[14][11] = -1.48509e-18;
	invR[14][12] = -4.83111e-18;
	invR[14][13] = 1.69098e-18;
	invR[14][14] = -3.95982e-18;
	invR[14][15] = -1.12735e-18;
	invR[14][16] = -4.17023e-10;
	invR[14][17] = 4.61221e-10;
	invR[14][18] = 2.02688e-18;
	invR[14][19] = -1.48454e-10;
	invR[15][0] = -2.41861e-10;
	invR[15][1] = -2.72453e-10;
	invR[15][2] = 1.71105e-10;
	invR[15][3] = 2.5583e-10;
	invR[15][4] = 3.64328e-10;
	invR[15][5] = 4.15014e-10;
	invR[15][6] = -1.74835e-10;
	invR[15][7] = -7.29632e-10;
	invR[15][8] = 1.08444e-10;
	invR[15][9] = 5.12157e-10;
	invR[15][10] = 7.06775e-19;
	invR[15][11] = -1.07071e-19;
	invR[15][12] = -2.13136e-18;
	invR[15][13] = 1.20144e-18;
	invR[15][14] = -1.12735e-18;
	invR[15][15] = -3.77059e-14;
	invR[15][16] = -2.67156e-11;
	invR[15][17] = -1.8349e-10;
	invR[15][18] = -5.41825e-20;
	invR[15][19] = -3.74793e-10;
	invR[16][0] = -0.0767415;
	invR[16][1] = -0.0170554;
	invR[16][2] = 0.117484;
	invR[16][3] = 0.175974;
	invR[16][4] = 0.0149693;
	invR[16][5] = 0.0736715;
	invR[16][6] = -0.0423299;
	invR[16][7] = -0.149667;
	invR[16][8] = 0.018088;
	invR[16][9] = 0.0762791;
	invR[16][10] = -5.20718e-10;
	invR[16][11] = 4.68499e-10;
	invR[16][12] = -3.6278e-10;
	invR[16][13] = 4.30878e-10;
	invR[16][14] = -4.17023e-10;
	invR[16][15] = -2.67156e-11;
	invR[16][16] = 0.0465149;
	invR[16][17] = -0.0515979;
	invR[16][18] = 5.95044e-10;
	invR[16][19] = 0.0167117;
	invR[17][0] = 0.085342;
	invR[17][1] = 0.0191557;
	invR[17][2] = -0.130724;
	invR[17][3] = -0.195576;
	invR[17][4] = -0.0172666;
	invR[17][5] = -0.0797885;
	invR[17][6] = 0.0464282;
	invR[17][7] = 0.161423;
	invR[17][8] = -0.019131;
	invR[17][9] = -0.0845029;
	invR[17][10] = 6.30165e-10;
	invR[17][11] = -5.75426e-10;
	invR[17][12] = 4.25325e-10;
	invR[17][13] = -5.29486e-10;
	invR[17][14] = 4.61221e-10;
	invR[17][15] = -1.8349e-10;
	invR[17][16] = -0.0515979;
	invR[17][17] = 0.0572705;
	invR[17][18] = -1.66771e-10;
	invR[17][19] = -0.0185721;
	invR[18][0] = -7.78731e-10;
	invR[18][1] = 1.00753e-09;
	invR[18][2] = 5.11386e-10;
	invR[18][3] = 6.69838e-10;
	invR[18][4] = -1.17879e-11;
	invR[18][5] = -8.78723e-11;
	invR[18][6] = -5.36357e-10;
	invR[18][7] = -2.18291e-09;
	invR[18][8] = 1.66264e-10;
	invR[18][9] = 6.62821e-11;
	invR[18][10] = -1.89781e-18;
	invR[18][11] = 1.8012e-18;
	invR[18][12] = 1.38366e-18;
	invR[18][13] = 1.45859e-19;
	invR[18][14] = 2.02688e-18;
	invR[18][15] = -5.41824e-20;
	invR[18][16] = 5.95044e-10;
	invR[18][17] = -1.66771e-10;
	invR[18][18] = -2.39444e-18;
	invR[18][19] = -2.24538e-10;
	invR[19][0] = -0.0277858;
	invR[19][1] = -0.0063642;
	invR[19][2] = 0.0426103;
	invR[19][3] = 0.0635954;
	invR[19][4] = 0.00603964;
	invR[19][5] = 0.0245349;
	invR[19][6] = -0.0146808;
	invR[19][7] = -0.0491724;
	invR[19][8] = 0.00556499;
	invR[19][9] = 0.0272936;
	invR[19][10] = -2.39626e-10;
	invR[19][11] = 2.24052e-10;
	invR[19][12] = -1.5324e-10;
	invR[19][13] = 2.06328e-10;
	invR[19][14] = -1.48454e-10;
	invR[19][15] = -3.74793e-10;
	invR[19][16] = 0.0167117;
	invR[19][17] = -0.0185721;
	invR[19][18] = -2.24538e-10;
	invR[19][19] = 0.00603825;

	invP[0][0]=-292874;
	invP[0][1]=-1.54125e+06;
	invP[0][2]=-569015;
	invP[0][3]=-774310;
	invP[0][4]=-240072;
	invP[0][5]=986521;
	invP[0][6]=-487676;
	invP[0][7]=614577;
	invP[0][8]=-1.06269e+08;
	invP[0][9]=283958;
	invP[0][10]=-824107;
	invP[0][11]=317201;
	invP[0][12]=715550;
	invP[0][13]=-8.54881e+06;
	invP[0][14]=-360367;
	invP[0][15]=-1.20266e+06;
	invP[0][16]=-34359.6;
	invP[0][17]=5.08365e+06;
	invP[0][18]=-1.10865e+08;
	invP[0][19]=-113567;
	invP[0][20]=-321836;
	invP[0][21]=561864;
	invP[0][22]=-2.36099e+06;
	invP[0][23]=1.18763e+08;
	invP[0][24]=340199;
	invP[0][25]=-484580;
	invP[0][26]=-379090;
	invP[0][27]=534932;
	invP[0][28]=-1.3155e+08;
	invP[1][0]=-1.54125e+06;
	invP[1][1]=2.38185e+07;
	invP[1][2]=-3.34926e+06;
	invP[1][3]=-4.989e+06;
	invP[1][4]=-805082;
	invP[1][5]=5.94776e+06;
	invP[1][6]=-3.73489e+06;
	invP[1][7]=7.59658e+06;
	invP[1][8]=-3.85372e+08;
	invP[1][9]=1.01427e+06;
	invP[1][10]=-1.04139e+07;
	invP[1][11]=2.47613e+06;
	invP[1][12]=8.90876e+06;
	invP[1][13]=-3.54349e+08;
	invP[1][14]=-2.02586e+06;
	invP[1][15]=-8.74731e+06;
	invP[1][16]=-1.9751e+06;
	invP[1][17]=2.70338e+07;
	invP[1][18]=-1.13814e+09;
	invP[1][19]=142214;
	invP[1][20]=-1.12993e+07;
	invP[1][21]=3.38798e+06;
	invP[1][22]=-3.18892e+07;
	invP[1][23]=1.6517e+09;
	invP[1][24]=505357;
	invP[1][25]=-4.96173e+06;
	invP[1][26]=-2.26909e+06;
	invP[1][27]=2.74418e+06;
	invP[1][28]=-9.90344e+08;
	invP[2][0]=-569015;
	invP[2][1]=-3.34926e+06;
	invP[2][2]=-465061;
	invP[2][3]=-1.34035e+06;
	invP[2][4]=-312634;
	invP[2][5]=1.37352e+06;
	invP[2][6]=-591226;
	invP[2][7]=215429;
	invP[2][8]=-1.548e+08;
	invP[2][9]=397510;
	invP[2][10]=-1.81216e+06;
	invP[2][11]=643238;
	invP[2][12]=1.45203e+06;
	invP[2][13]=-6.73841e+07;
	invP[2][14]=-468265;
	invP[2][15]=-1.3028e+06;
	invP[2][16]=-91789;
	invP[2][17]=7.48445e+06;
	invP[2][18]=-2.43172e+08;
	invP[2][19]=-1893.64;
	invP[2][20]=866028;
	invP[2][21]=856819;
	invP[2][22]=-4.0299e+06;
	invP[2][23]=1.505e+08;
	invP[2][24]=324711;
	invP[2][25]=-688934;
	invP[2][26]=-432062;
	invP[2][27]=770015;
	invP[2][28]=-1.61648e+08;
	invP[3][0]=-774310;
	invP[3][1]=-4.989e+06;
	invP[3][2]=-1.34035e+06;
	invP[3][3]=-183020;
	invP[3][4]=-598924;
	invP[3][5]=2.95891e+06;
	invP[3][6]=-1.53087e+06;
	invP[3][7]=2.11222e+06;
	invP[3][8]=-1.37349e+08;
	invP[3][9]=1.13979e+06;
	invP[3][10]=-3.4039e+06;
	invP[3][11]=1.19247e+06;
	invP[3][12]=2.31028e+06;
	invP[3][13]=6.63789e+07;
	invP[3][14]=-1.31758e+06;
	invP[3][15]=-4.99088e+06;
	invP[3][16]=434400;
	invP[3][17]=2.09051e+07;
	invP[3][18]=-1.2391e+08;
	invP[3][19]=-564870;
	invP[3][20]=-2.18959e+06;
	invP[3][21]=2.58881e+06;
	invP[3][22]=-8.54856e+06;
	invP[3][23]=5.08475e+08;
	invP[3][24]=896675;
	invP[3][25]=-2.04622e+06;
	invP[3][26]=-1.09056e+06;
	invP[3][27]=2.86643e+06;
	invP[3][28]=-4.55006e+08;
	invP[4][0]=-240072;
	invP[4][1]=-805082;
	invP[4][2]=-312634;
	invP[4][3]=-598924;
	invP[4][4]=171031;
	invP[4][5]=840121;
	invP[4][6]=-356045;
	invP[4][7]=914016;
	invP[4][8]=-1.42196e+07;
	invP[4][9]=-21012.7;
	invP[4][10]=-1.38986e+06;
	invP[4][11]=209309;
	invP[4][12]=208261;
	invP[4][13]=9.51228e+06;
	invP[4][14]=-131291;
	invP[4][15]=-643600;
	invP[4][16]=-106340;
	invP[4][17]=3.16216e+06;
	invP[4][18]=5.21422e+07;
	invP[4][19]=151211;
	invP[4][20]=-94504.8;
	invP[4][21]=336128;
	invP[4][22]=-1.40267e+06;
	invP[4][23]=1.56448e+08;
	invP[4][24]=-146981;
	invP[4][25]=249421;
	invP[4][26]=-97067.3;
	invP[4][27]=462500;
	invP[4][28]=-5.43982e+07;
	invP[5][0]=986521;
	invP[5][1]=5.94776e+06;
	invP[5][2]=1.37352e+06;
	invP[5][3]=2.95891e+06;
	invP[5][4]=840121;
	invP[5][5]=-1.98675e+06;
	invP[5][6]=985267;
	invP[5][7]=-571210;
	invP[5][8]=2.93424e+08;
	invP[5][9]=-920688;
	invP[5][10]=2.05184e+06;
	invP[5][11]=-1.06738e+06;
	invP[5][12]=-2.38214e+06;
	invP[5][13]=6.40827e+07;
	invP[5][14]=941316;
	invP[5][15]=2.95135e+06;
	invP[5][16]=-147910;
	invP[5][17]=-1.20868e+07;
	invP[5][18]=6.88269e+08;
	invP[5][19]=382573;
	invP[5][20]=-1.79674e+06;
	invP[5][21]=-1.63516e+06;
	invP[5][22]=5.87109e+06;
	invP[5][23]=-3.00257e+08;
	invP[5][24]=-908874;
	invP[5][25]=1.22807e+06;
	invP[5][26]=933680;
	invP[5][27]=-892441;
	invP[5][28]=3.15968e+08;
	invP[6][0]=-487676;
	invP[6][1]=-3.73489e+06;
	invP[6][2]=-591226;
	invP[6][3]=-1.53087e+06;
	invP[6][4]=-356045;
	invP[6][5]=985267;
	invP[6][6]=-462350;
	invP[6][7]=245085;
	invP[6][8]=-1.33417e+08;
	invP[6][9]=412300;
	invP[6][10]=-1.23424e+06;
	invP[6][11]=540929;
	invP[6][12]=1.02674e+06;
	invP[6][13]=-1.86042e+07;
	invP[6][14]=-448083;
	invP[6][15]=-1.22832e+06;
	invP[6][16]=46916.9;
	invP[6][17]=5.94979e+06;
	invP[6][18]=-3.17062e+08;
	invP[6][19]=-137140;
	invP[6][20]=1.30596e+06;
	invP[6][21]=767623;
	invP[6][22]=-2.50827e+06;
	invP[6][23]=1.37947e+08;
	invP[6][24]=363189;
	invP[6][25]=-351790;
	invP[6][26]=-379479;
	invP[6][27]=492208;
	invP[6][28]=-1.30589e+08;
	invP[7][0]=614577;
	invP[7][1]=7.59658e+06;
	invP[7][2]=215429;
	invP[7][3]=2.11222e+06;
	invP[7][4]=914016;
	invP[7][5]=-571210;
	invP[7][6]=245085;
	invP[7][7]=1.5711e+06;
	invP[7][8]=2.40418e+08;
	invP[7][9]=-686305;
	invP[7][10]=965221;
	invP[7][11]=-839157;
	invP[7][12]=-1.73499e+06;
	invP[7][13]=1.00762e+08;
	invP[7][14]=606598;
	invP[7][15]=790016;
	invP[7][16]=-141175;
	invP[7][17]=-6.35095e+06;
	invP[7][18]=6.57907e+08;
	invP[7][19]=384785;
	invP[7][20]=-4.1269e+06;
	invP[7][21]=-829970;
	invP[7][22]=2.61428e+06;
	invP[7][23]=5.82456e+07;
	invP[7][24]=-904096;
	invP[7][25]=672209;
	invP[7][26]=280753;
	invP[7][27]=-203482;
	invP[7][28]=6.51565e+07;
	invP[8][0]=-1.06269e+08;
	invP[8][1]=-3.85372e+08;
	invP[8][2]=-1.548e+08;
	invP[8][3]=-1.37349e+08;
	invP[8][4]=-1.42196e+07;
	invP[8][5]=2.93424e+08;
	invP[8][6]=-1.33417e+08;
	invP[8][7]=2.40418e+08;
	invP[8][8]=-1.85437e+10;
	invP[8][9]=3.90866e+07;
	invP[8][10]=-3.10127e+08;
	invP[8][11]=7.80253e+07;
	invP[8][12]=1.72225e+08;
	invP[8][13]=-8.97436e+09;
	invP[8][14]=-7.31966e+07;
	invP[8][15]=-4.15931e+08;
	invP[8][16]=1.72884e+07;
	invP[8][17]=1.45734e+09;
	invP[8][18]=2.46743e+09;
	invP[8][19]=9.87622e+06;
	invP[8][20]=-1.14514e+08;
	invP[8][21]=1.88189e+08;
	invP[8][22]=-6.35797e+08;
	invP[8][23]=5.31317e+10;
	invP[8][24]=4.93502e+07;
	invP[8][25]=-9.01891e+07;
	invP[8][26]=-8.46126e+07;
	invP[8][27]=1.62332e+08;
	invP[8][28]=-3.2701e+10;
	invP[9][0]=283958;
	invP[9][1]=1.01427e+06;
	invP[9][2]=397510;
	invP[9][3]=1.13979e+06;
	invP[9][4]=-21012.7;
	invP[9][5]=-920688;
	invP[9][6]=412300;
	invP[9][7]=-686305;
	invP[9][8]=3.90866e+07;
	invP[9][9]=-159742;
	invP[9][10]=1.10836e+06;
	invP[9][11]=-251240;
	invP[9][12]=-427530;
	invP[9][13]=-3.75956e+07;
	invP[9][14]=216924;
	invP[9][15]=522952;
	invP[9][16]=22894.4;
	invP[9][17]=-3.73747e+06;
	invP[9][18]=1.11132e+08;
	invP[9][19]=-31739.3;
	invP[9][20]=151752;
	invP[9][21]=-487322;
	invP[9][22]=2.01306e+06;
	invP[9][23]=-1.49939e+08;
	invP[9][24]=-32629.4;
	invP[9][25]=5249.22;
	invP[9][26]=206604;
	invP[9][27]=-706775;
	invP[9][28]=8.83191e+07;
	invP[10][0]=-824107;
	invP[10][1]=-1.04139e+07;
	invP[10][2]=-1.81216e+06;
	invP[10][3]=-3.4039e+06;
	invP[10][4]=-1.38986e+06;
	invP[10][5]=2.05184e+06;
	invP[10][6]=-1.23424e+06;
	invP[10][7]=965221;
	invP[10][8]=-3.10127e+08;
	invP[10][9]=1.10836e+06;
	invP[10][10]=-2.25962e+06;
	invP[10][11]=1.22853e+06;
	invP[10][12]=2.70048e+06;
	invP[10][13]=-4.90804e+07;
	invP[10][14]=-1.35579e+06;
	invP[10][15]=-3.83364e+06;
	invP[10][16]=240814;
	invP[10][17]=1.8896e+07;
	invP[10][18]=-9.47768e+08;
	invP[10][19]=-858284;
	invP[10][20]=1.83828e+06;
	invP[10][21]=2.18423e+06;
	invP[10][22]=-6.46608e+06;
	invP[10][23]=3.44372e+08;
	invP[10][24]=1.25025e+06;
	invP[10][25]=-1.46926e+06;
	invP[10][26]=-1.28876e+06;
	invP[10][27]=1.72215e+06;
	invP[10][28]=-4.40138e+08;
	invP[11][0]=317201;
	invP[11][1]=2.47613e+06;
	invP[11][2]=643238;
	invP[11][3]=1.19247e+06;
	invP[11][4]=209309;
	invP[11][5]=-1.06738e+06;
	invP[11][6]=540929;
	invP[11][7]=-839157;
	invP[11][8]=7.80253e+07;
	invP[11][9]=-251240;
	invP[11][10]=1.22853e+06;
	invP[11][11]=-367907;
	invP[11][12]=-767696;
	invP[11][13]=1.04436e+06;
	invP[11][14]=385863;
	invP[11][15]=1.32258e+06;
	invP[11][16]=-39589.5;
	invP[11][17]=-6.44995e+06;
	invP[11][18]=1.84684e+08;
	invP[11][19]=145715;
	invP[11][20]=53910.1;
	invP[11][21]=-755818;
	invP[11][22]=2.61047e+06;
	invP[11][23]=-1.80067e+08;
	invP[11][24]=-245921;
	invP[11][25]=290723;
	invP[11][26]=393064;
	invP[11][27]=-816638;
	invP[11][28]=1.4796e+08;
	invP[12][0]=715550;
	invP[12][1]=8.90876e+06;
	invP[12][2]=1.45203e+06;
	invP[12][3]=2.31028e+06;
	invP[12][4]=208261;
	invP[12][5]=-2.38214e+06;
	invP[12][6]=1.02674e+06;
	invP[12][7]=-1.73499e+06;
	invP[12][8]=1.72225e+08;
	invP[12][9]=-427530;
	invP[12][10]=2.70048e+06;
	invP[12][11]=-767696;
	invP[12][12]=-921162;
	invP[12][13]=-6.33559e+06;
	invP[12][14]=719432;
	invP[12][15]=2.72652e+06;
	invP[12][16]=-131644;
	invP[12][17]=-1.32975e+07;
	invP[12][18]=1.82455e+08;
	invP[12][19]=485529;
	invP[12][20]=-1.54269e+06;
	invP[12][21]=-1.27982e+06;
	invP[12][22]=2.93232e+06;
	invP[12][23]=-2.19132e+08;
	invP[12][24]=-542812;
	invP[12][25]=-14907.1;
	invP[12][26]=797687;
	invP[12][27]=-1.29158e+06;
	invP[12][28]=2.75721e+08;
	invP[13][0]=-8.54881e+06;
	invP[13][1]=-3.54349e+08;
	invP[13][2]=-6.73841e+07;
	invP[13][3]=6.63789e+07;
	invP[13][4]=9.51228e+06;
	invP[13][5]=6.40827e+07;
	invP[13][6]=-1.86042e+07;
	invP[13][7]=1.00762e+08;
	invP[13][8]=-8.97436e+09;
	invP[13][9]=-3.75956e+07;
	invP[13][10]=-4.90804e+07;
	invP[13][11]=1.04436e+06;
	invP[13][12]=-6.33559e+06;
	invP[13][13]=-2.09148e+10;
	invP[13][14]=3.20152e+06;
	invP[13][15]=-2.3898e+08;
	invP[13][16]=1.1168e+07;
	invP[13][17]=3.07757e+08;
	invP[13][18]=3.73993e+10;
	invP[13][19]=-3.56228e+07;
	invP[13][20]=8.53613e+07;
	invP[13][21]=-1.22331e+06;
	invP[13][22]=1.22594e+08;
	invP[13][23]=-4.28268e+09;
	invP[13][24]=3.12273e+07;
	invP[13][25]=2.05921e+07;
	invP[13][26]=-2.37786e+07;
	invP[13][27]=-8.19409e+07;
	invP[13][28]=-4.01139e+09;
	invP[14][0]=-360367;
	invP[14][1]=-2.02586e+06;
	invP[14][2]=-468265;
	invP[14][3]=-1.31758e+06;
	invP[14][4]=-131291;
	invP[14][5]=941316;
	invP[14][6]=-448083;
	invP[14][7]=606598;
	invP[14][8]=-7.31966e+07;
	invP[14][9]=216924;
	invP[14][10]=-1.35579e+06;
	invP[14][11]=385863;
	invP[14][12]=719432;
	invP[14][13]=3.20152e+06;
	invP[14][14]=-254842;
	invP[14][15]=-866581;
	invP[14][16]=-15389.8;
	invP[14][17]=4.84505e+06;
	invP[14][18]=-1.76318e+08;
	invP[14][19]=-67803.9;
	invP[14][20]=433244;
	invP[14][21]=567610;
	invP[14][22]=-2.37609e+06;
	invP[14][23]=1.45136e+08;
	invP[14][24]=124326;
	invP[14][25]=-62232.2;
	invP[14][26]=-244626;
	invP[14][27]=623760;
	invP[14][28]=-9.69999e+07;
	invP[15][0]=-1.20266e+06;
	invP[15][1]=-8.74731e+06;
	invP[15][2]=-1.3028e+06;
	invP[15][3]=-4.99088e+06;
	invP[15][4]=-643600;
	invP[15][5]=2.95135e+06;
	invP[15][6]=-1.22832e+06;
	invP[15][7]=790016;
	invP[15][8]=-4.15931e+08;
	invP[15][9]=522952;
	invP[15][10]=-3.83364e+06;
	invP[15][11]=1.32258e+06;
	invP[15][12]=2.72652e+06;
	invP[15][13]=-2.3898e+08;
	invP[15][14]=-866581;
	invP[15][15]=-3.95445e+06;
	invP[15][16]=243305;
	invP[15][17]=1.17695e+07;
	invP[15][18]=-6.88411e+08;
	invP[15][19]=-364501;
	invP[15][20]=4.86818e+06;
	invP[15][21]=1.75045e+06;
	invP[15][22]=-6.42007e+06;
	invP[15][23]=3.47783e+08;
	invP[15][24]=988564;
	invP[15][25]=133242;
	invP[15][26]=-449654;
	invP[15][27]=494174;
	invP[15][28]=-1.26921e+08;
	invP[16][0]=-34359.5;
	invP[16][1]=-1.9751e+06;
	invP[16][2]=-91789;
	invP[16][3]=434400;
	invP[16][4]=-106340;
	invP[16][5]=-147910;
	invP[16][6]=46916.9;
	invP[16][7]=-141175;
	invP[16][8]=1.72884e+07;
	invP[16][9]=22894.4;
	invP[16][10]=240814;
	invP[16][11]=-39589.5;
	invP[16][12]=-131644;
	invP[16][13]=1.1168e+07;
	invP[16][14]=-15389.8;
	invP[16][15]=243305;
	invP[16][16]=-80227.8;
	invP[16][17]=1.93714e+06;
	invP[16][18]=5.31605e+07;
	invP[16][19]=-128771;
	invP[16][20]=-242038;
	invP[16][21]=-95525.1;
	invP[16][22]=559362;
	invP[16][23]=-9.9523e+07;
	invP[16][24]=153930;
	invP[16][25]=-76836.9;
	invP[16][26]=-81549;
	invP[16][27]=238708;
	invP[16][28]=-1.67892e+07;
	invP[17][0]=5.08365e+06;
	invP[17][1]=2.70338e+07;
	invP[17][2]=7.48445e+06;
	invP[17][3]=2.09051e+07;
	invP[17][4]=3.16216e+06;
	invP[17][5]=-1.20868e+07;
	invP[17][6]=5.94979e+06;
	invP[17][7]=-6.35095e+06;
	invP[17][8]=1.45734e+09;
	invP[17][9]=-3.73747e+06;
	invP[17][10]=1.8896e+07;
	invP[17][11]=-6.44995e+06;
	invP[17][12]=-1.32975e+07;
	invP[17][13]=3.07757e+08;
	invP[17][14]=4.84505e+06;
	invP[17][15]=1.17695e+07;
	invP[17][16]=1.93714e+06;
	invP[17][17]=-6.3958e+07;
	invP[17][18]=3.95748e+09;
	invP[17][19]=1.711e+06;
	invP[17][20]=-1.14924e+07;
	invP[17][21]=-6.77459e+06;
	invP[17][22]=3.61745e+07;
	invP[17][23]=-1.60969e+09;
	invP[17][24]=-3.02918e+06;
	invP[17][25]=3.18271e+06;
	invP[17][26]=4.64916e+06;
	invP[17][27]=-4.46926e+06;
	invP[17][28]=1.64601e+09;
	invP[18][0]=-1.10865e+08;
	invP[18][1]=-1.13814e+09;
	invP[18][2]=-2.43172e+08;
	invP[18][3]=-1.2391e+08;
	invP[18][4]=5.21422e+07;
	invP[18][5]=6.88269e+08;
	invP[18][6]=-3.17062e+08;
	invP[18][7]=6.57907e+08;
	invP[18][8]=2.46743e+09;
	invP[18][9]=1.11132e+08;
	invP[18][10]=-9.47768e+08;
	invP[18][11]=1.84684e+08;
	invP[18][12]=1.82455e+08;
	invP[18][13]=3.73993e+10;
	invP[18][14]=-1.76318e+08;
	invP[18][15]=-6.88411e+08;
	invP[18][16]=5.31605e+07;
	invP[18][17]=3.95748e+09;
	invP[18][18]=4.70023e+10;
	invP[18][19]=-5.92815e+07;
	invP[18][20]=-4.26005e+08;
	invP[18][21]=4.29791e+08;
	invP[18][22]=-1.29892e+09;
	invP[18][23]=1.05068e+11;
	invP[18][24]=-1.96903e+06;
	invP[18][25]=-1.22526e+08;
	invP[18][26]=-1.87668e+08;
	invP[18][27]=6.75878e+08;
	invP[18][28]=-8.88256e+10;
	invP[19][0]=-113567;
	invP[19][1]=142214;
	invP[19][2]=-1893.65;
	invP[19][3]=-564870;
	invP[19][4]=151211;
	invP[19][5]=382573;
	invP[19][6]=-137140;
	invP[19][7]=384785;
	invP[19][8]=9.87622e+06;
	invP[19][9]=-31739.3;
	invP[19][10]=-858284;
	invP[19][11]=145715;
	invP[19][12]=485529;
	invP[19][13]=-3.56228e+07;
	invP[19][14]=-67803.9;
	invP[19][15]=-364501;
	invP[19][16]=-128771;
	invP[19][17]=1.711e+06;
	invP[19][18]=-5.92815e+07;
	invP[19][19]=117916;
	invP[19][20]=-217889;
	invP[19][21]=172707;
	invP[19][22]=-1.58526e+06;
	invP[19][23]=9.27914e+07;
	invP[19][24]=-225016;
	invP[19][25]=188177;
	invP[19][26]=48396.5;
	invP[19][27]=346959;
	invP[19][28]=2.52428e+06;
	invP[20][0]=-321836;
	invP[20][1]=-1.12993e+07;
	invP[20][2]=866028;
	invP[20][3]=-2.18959e+06;
	invP[20][4]=-94504.8;
	invP[20][5]=-1.79674e+06;
	invP[20][6]=1.30596e+06;
	invP[20][7]=-4.1269e+06;
	invP[20][8]=-1.14514e+08;
	invP[20][9]=151752;
	invP[20][10]=1.83828e+06;
	invP[20][11]=53910;
	invP[20][12]=-1.54269e+06;
	invP[20][13]=8.53613e+07;
	invP[20][14]=433244;
	invP[20][15]=4.86818e+06;
	invP[20][16]=-242038;
	invP[20][17]=-1.14924e+07;
	invP[20][18]=-4.26005e+08;
	invP[20][19]=-217889;
	invP[20][20]=9.25187e+06;
	invP[20][21]=-1.8814e+06;
	invP[20][22]=9.13496e+06;
	invP[20][23]=-8.45216e+08;
	invP[20][24]=271429;
	invP[20][25]=2.29465e+06;
	invP[20][26]=796798;
	invP[20][27]=-2.31505e+06;
	invP[20][28]=4.15414e+08;
	invP[21][0]=561864;
	invP[21][1]=3.38798e+06;
	invP[21][2]=856819;
	invP[21][3]=2.58881e+06;
	invP[21][4]=336128;
	invP[21][5]=-1.63516e+06;
	invP[21][6]=767623;
	invP[21][7]=-829970;
	invP[21][8]=1.88189e+08;
	invP[21][9]=-487322;
	invP[21][10]=2.18423e+06;
	invP[21][11]=-755818;
	invP[21][12]=-1.27982e+06;
	invP[21][13]=-1.2233e+06;
	invP[21][14]=567610;
	invP[21][15]=1.75045e+06;
	invP[21][16]=-95525.1;
	invP[21][17]=-6.77459e+06;
	invP[21][18]=4.29791e+08;
	invP[21][19]=172707;
	invP[21][20]=-1.8814e+06;
	invP[21][21]=-1.06134e+06;
	invP[21][22]=3.71441e+06;
	invP[21][23]=-2.69007e+08;
	invP[21][24]=-336362;
	invP[21][25]=287410;
	invP[21][26]=525411;
	invP[21][27]=-535421;
	invP[21][28]=1.96415e+08;
	invP[22][0]=-2.36099e+06;
	invP[22][1]=-3.18892e+07;
	invP[22][2]=-4.0299e+06;
	invP[22][3]=-8.54856e+06;
	invP[22][4]=-1.40267e+06;
	invP[22][5]=5.87109e+06;
	invP[22][6]=-2.50827e+06;
	invP[22][7]=2.61428e+06;
	invP[22][8]=-6.35797e+08;
	invP[22][9]=2.01306e+06;
	invP[22][10]=-6.46608e+06;
	invP[22][11]=2.61047e+06;
	invP[22][12]=2.93232e+06;
	invP[22][13]=1.22594e+08;
	invP[22][14]=-2.37609e+06;
	invP[22][15]=-6.42007e+06;
	invP[22][16]=559362;
	invP[22][17]=3.61745e+07;
	invP[22][18]=-1.29892e+09;
	invP[22][19]=-1.58526e+06;
	invP[22][20]=9.13496e+06;
	invP[22][21]=3.71441e+06;
	invP[22][22]=-6.20765e+06;
	invP[22][23]=4.38939e+08;
	invP[22][24]=1.96676e+06;
	invP[22][25]=103998;
	invP[22][26]=-2.29331e+06;
	invP[22][27]=3.09096e+06;
	invP[22][28]=-7.41629e+08;
	invP[23][0]=1.18763e+08;
	invP[23][1]=1.6517e+09;
	invP[23][2]=1.505e+08;
	invP[23][3]=5.08475e+08;
	invP[23][4]=1.56448e+08;
	invP[23][5]=-3.00257e+08;
	invP[23][6]=1.37947e+08;
	invP[23][7]=5.82456e+07;
	invP[23][8]=5.31317e+10;
	invP[23][9]=-1.49939e+08;
	invP[23][10]=3.44372e+08;
	invP[23][11]=-1.80067e+08;
	invP[23][12]=-2.19132e+08;
	invP[23][13]=-4.28268e+09;
	invP[23][14]=1.45136e+08;
	invP[23][15]=3.47783e+08;
	invP[23][16]=-9.9523e+07;
	invP[23][17]=-1.60969e+09;
	invP[23][18]=1.05068e+11;
	invP[23][19]=9.27914e+07;
	invP[23][20]=-8.45216e+08;
	invP[23][21]=-2.69007e+08;
	invP[23][22]=4.38939e+08;
	invP[23][23]=-2.86085e+10;
	invP[23][24]=-1.76253e+08;
	invP[23][25]=1.01995e+08;
	invP[23][26]=1.15501e+08;
	invP[23][27]=-1.01498e+08;
	invP[23][28]=3.91061e+10;
	invP[24][0]=340199;
	invP[24][1]=505357;
	invP[24][2]=324711;
	invP[24][3]=896675;
	invP[24][4]=-146981;
	invP[24][5]=-908874;
	invP[24][6]=363189;
	invP[24][7]=-904096;
	invP[24][8]=4.93502e+07;
	invP[24][9]=-32629.4;
	invP[24][10]=1.25025e+06;
	invP[24][11]=-245921;
	invP[24][12]=-542812;
	invP[24][13]=3.12273e+07;
	invP[24][14]=124326;
	invP[24][15]=988564;
	invP[24][16]=153930;
	invP[24][17]=-3.02918e+06;
	invP[24][18]=-1.96904e+06;
	invP[24][19]=-225016;
	invP[24][20]=271429;
	invP[24][21]=-336362;
	invP[24][22]=1.96676e+06;
	invP[24][23]=-1.76253e+08;
	invP[24][24]=201717;
	invP[24][25]=-278743;
	invP[24][26]=46109.1;
	invP[24][27]=-398498;
	invP[24][28]=2.6234e+07;
	invP[25][0]=-484580;
	invP[25][1]=-4.96173e+06;
	invP[25][2]=-688934;
	invP[25][3]=-2.04622e+06;
	invP[25][4]=249421;
	invP[25][5]=1.22807e+06;
	invP[25][6]=-351790;
	invP[25][7]=672209;
	invP[25][8]=-9.01891e+07;
	invP[25][9]=5249.21;
	invP[25][10]=-1.46926e+06;
	invP[25][11]=290723;
	invP[25][12]=-14907.1;
	invP[25][13]=2.05921e+07;
	invP[25][14]=-62232.2;
	invP[25][15]=133242;
	invP[25][16]=-76836.9;
	invP[25][17]=3.18271e+06;
	invP[25][18]=-1.22526e+08;
	invP[25][19]=188177;
	invP[25][20]=2.29465e+06;
	invP[25][21]=287410;
	invP[25][22]=103998;
	invP[25][23]=1.01995e+08;
	invP[25][24]=-278743;
	invP[25][25]=411558;
	invP[25][26]=-506775;
	invP[25][27]=160027;
	invP[25][28]=-1.78859e+08;
	invP[26][0]=-379090;
	invP[26][1]=-2.26909e+06;
	invP[26][2]=-432062;
	invP[26][3]=-1.09056e+06;
	invP[26][4]=-97067.3;
	invP[26][5]=933680;
	invP[26][6]=-379479;
	invP[26][7]=280753;
	invP[26][8]=-8.46126e+07;
	invP[26][9]=206604;
	invP[26][10]=-1.28876e+06;
	invP[26][11]=393064;
	invP[26][12]=797687;
	invP[26][13]=-2.37786e+07;
	invP[26][14]=-244626;
	invP[26][15]=-449654;
	invP[26][16]=-81549.1;
	invP[26][17]=4.64916e+06;
	invP[26][18]=-1.87668e+08;
	invP[26][19]=48396.5;
	invP[26][20]=796798;
	invP[26][21]=525411;
	invP[26][22]=-2.29331e+06;
	invP[26][23]=1.15501e+08;
	invP[26][24]=46109.1;
	invP[26][25]=-506775;
	invP[26][26]=-470385;
	invP[26][27]=439341;
	invP[26][28]=-1.79309e+08;
	invP[27][0]=534932;
	invP[27][1]=2.74418e+06;
	invP[27][2]=770015;
	invP[27][3]=2.86643e+06;
	invP[27][4]=462500;
	invP[27][5]=-892441;
	invP[27][6]=492208;
	invP[27][7]=-203482;
	invP[27][8]=1.62332e+08;
	invP[27][9]=-706775;
	invP[27][10]=1.72215e+06;
	invP[27][11]=-816638;
	invP[27][12]=-1.29158e+06;
	invP[27][13]=-8.19409e+07;
	invP[27][14]=623760;
	invP[27][15]=494174;
	invP[27][16]=238708;
	invP[27][17]=-4.46926e+06;
	invP[27][18]=6.75878e+08;
	invP[27][19]=346959;
	invP[27][20]=-2.31505e+06;
	invP[27][21]=-535421;
	invP[27][22]=3.09096e+06;
	invP[27][23]=-1.01498e+08;
	invP[27][24]=-398498;
	invP[27][25]=160027;
	invP[27][26]=439341;
	invP[27][27]=-137728;
	invP[27][28]=1.46672e+08;
	invP[28][0]=-1.3155e+08;
	invP[28][1]=-9.90344e+08;
	invP[28][2]=-1.61648e+08;
	invP[28][3]=-4.55006e+08;
	invP[28][4]=-5.43982e+07;
	invP[28][5]=3.15968e+08;
	invP[28][6]=-1.30589e+08;
	invP[28][7]=6.51565e+07;
	invP[28][8]=-3.2701e+10;
	invP[28][9]=8.83191e+07;
	invP[28][10]=-4.40138e+08;
	invP[28][11]=1.4796e+08;
	invP[28][12]=2.75721e+08;
	invP[28][13]=-4.01139e+09;
	invP[28][14]=-9.69999e+07;
	invP[28][15]=-1.26921e+08;
	invP[28][16]=-1.67892e+07;
	invP[28][17]=1.64601e+09;
	invP[28][18]=-8.88256e+10;
	invP[28][19]=2.52428e+06;
	invP[28][20]=4.15414e+08;
	invP[28][21]=1.96415e+08;
	invP[28][22]=-7.41629e+08;
	invP[28][23]=3.91061e+10;
	invP[28][24]=2.6234e+07;
	invP[28][25]=-1.78859e+08;
	invP[28][26]=-1.79309e+08;
	invP[28][27]=1.46672e+08;
	invP[28][28]=-6.74485e+10;
}

}//: namespace KW_MAP
}//: namespace Processors
