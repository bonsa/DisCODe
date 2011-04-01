/*!
 * \file KW_PalmDetection.cpp
 * \brief Opisywanie cech dłoni
 * \author kwasak
 * \date 2011-03-31
 */

#include <memory>
#include <string>
#include <math.h> 

#include "KW_PalmDetection.hpp"
#include "Logger.hpp"
#include "Types/Ellipse.hpp"

namespace Processors {
namespace KW_Palm {

using namespace cv;

KW_PalmDetection::KW_PalmDetection(const std::string & name) : Base::Component(name)
{
	LOG(LTRACE) << "Hello KW_PalmDetection\n";
}

KW_PalmDetection::~KW_PalmDetection()
{
	LOG(LTRACE) << "Good bye KW_PalmDetection\n";
}

bool KW_PalmDetection::onInit()
{
	LOG(LTRACE) << "KW_PalmDetection::initialize\n";

	h_onNewImage.setup(this, &KW_PalmDetection::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	h_onNewBlobs.setup(this, &KW_PalmDetection::onNewBlobs);
	registerHandler("onNewBlobs", &h_onNewBlobs);

	registerStream("in_blobs", &in_blobs);
	registerStream("in_img", &in_img);

	newImage = registerEvent("newImage");

	registerStream("out_signs", &out_signs);
	registerStream("out_draw", &out_draw);

	return true;
}

bool KW_PalmDetection::onFinish()
{
	LOG(LTRACE) << "KW_PalmDetection::finish\n";

	return true;
}

bool KW_PalmDetection::onStep()
{
	LOG(LTRACE) << "KW_PalmDetection::step\n";

	blobs_ready = img_ready = false;

	try {

		int id = 0;
		int i,ii, numerElements ;
		std::ofstream plik("/home/kasia/Test.txt");
		IplImage h = IplImage(tsl_img);
		Types::Blobs::Blob *currentBlob;
		Types::Blobs::BlobResult result;
		CvSeq * contour;
		CvSeqReader reader;
		CvPoint actualPoint;
		vector<CvPoint> contourPoints;
		vector<float> dist;
		vector<float> meanDist;
		vector<float> derivative;
		vector<CvPoint> characteristicPoint;
		float TempDist;
		characteristic points

		Types::DrawableContainer signs; //kontener przechowujący elementy, które mozna narysować

		// iterate through all found blobs

		double m00, m10, m01, m11, m02, m20;
//		double M11, M02, M20, M7, M1, M2;
		double Area, Perimeter, Ratio, MaxArea, CenterOfGravity_x, CenterOfGravity_y, MaxY;


		Types::DrawableContainer drawcont;

		MaxArea = 0;
		MaxY = 0;

		for (i = 0; i < blobs.GetNumBlobs(); i++ )
		{
			currentBlob = blobs.GetBlob(i);

			double xtot = 0, ytot = 0;

			Area = currentBlob->Area();
			if (Area > MaxArea)
			{
				MaxArea = Area;
				id = i;
				contour = currentBlob->GetExternalContour()->GetContourPoints();
				cvStartReadSeq(contour, &reader);

				int cnt = 0;
				for (int j = 0; j < contour->total; j=j+1) {
					CV_READ_SEQ_ELEM( actualPoint, reader);


					if (j%10 == 1) {
						plik << actualPoint.x << " " << actualPoint.y << std::endl;
						contourPoints.push_back(cvPoint(actualPoint.x, actualPoint.y));
						if (actualPoint.y > MaxY)
						{
							MaxY = actualPoint.y;
						}
						cnt++;
					}


					//JAK TO NAMALOWAC!?
				}


			//	Perimeter = currentBlob->Perimeter();

			//	Ratio = Perimeter * Perimeter / (4*3.14*Area);

				//środek cięzkości
				// calculate moments
				m00 = currentBlob->Moment(0,0);
				m01 = currentBlob->Moment(0,1);
				m10 = currentBlob->Moment(1,0);



				CenterOfGravity_x = m10/m00;
				CenterOfGravity_y = m01/m00;

				//przesuniety punkt środka ciężkości
				CenterOfGravity_y += (MaxY-CenterOfGravity_y)*2/3;

				//obliczenie roznicy miedzy punktami z odwodu a środkiem ciężkosci
				for(ii=0; ii < contourPoints.size(); ii++)
				{
					TempDist = (contourPoints[ii].x - CenterOfGravity_x)*(contourPoints[ii].x - CenterOfGravity_x)+(contourPoints[ii].y - CenterOfGravity_y)*(contourPoints[ii].y - CenterOfGravity_y);
					if(TempDist > 12100)
						dist.push_back(TempDist);
					else
						dist.push_back(12100);
				}

				//uśredniam, usuwanim szumy z wektora odległości
				TempDist = (dist[1]+dist[2]+dist[0])/3;
				meanDist.push_back(TempDist);

				TempDist = (dist[1]+dist[2]+dist[3]+dist[0])/4;
				meanDist.push_back(TempDist);

				numerElements = dist.size();
				for(ii=2; ii < numerElements - 2; ii++)
				{
					TempDist = (dist[ii-2]+dist[ii-1]+dist[ii]+dist[ii+1]+dist[ii+2])/5;
					meanDist.push_back(TempDist);
				}

				TempDist = (dist[numerElements-1]+dist[numerElements-2]+dist[numerElements-3]+dist[numerElements-4])/4;
				meanDist.push_back(TempDist);

				TempDist = (dist[numerElements-1]+dist[numerElements-2]+dist[numerElements-3])/3;
				meanDist.push_back(TempDist);

				for(ii=0; ii < numerElements - 1; ii++)
				{
					derivative.push_back = meanDist[ii]- meanDist[ii+1];
					TempDist = (dist[ii-2]+dist[ii-1]+dist[ii]+dist[ii+1]+dist[ii+2])/5;
					meanDist.push_back(TempDist);
				}
				//obliczenie pochodnej, szukanie ekstremów


				derivative


				Types::Ellipse * el = new Types::Ellipse(Point2f(CenterOfGravity_x, CenterOfGravity_y), Size2f(20,20));
				drawcont.add(el);
				Types::Ellipse * el2 = new Types::Ellipse(Point2f(last_x, last_y), Size2f(7,7));
				drawcont.add(el2);

				last_x = CenterOfGravity_x;
				last_y = CenterOfGravity_y;


				plik <<"Punkt środka cieżkosci: "<< CenterOfGravity_x <<" "<< CenterOfGravity_y;


			//	m11 = currentBlob->Moment(1,1);
			//	m02 = currentBlob->Moment(0,2);
			//	m20 = currentBlob->Moment(2,0);

			//	M11 = m11 - (m10*m01)/m00;
			//	M02 = m02 - (m01*m01)/m00;
			//	M20 = m20 - (m10*m10)/m00;

			//	M1 = M20 + M02;
			//	M2 = (M20 + M02)*(M20 + M02)+4*M11*M11;
			//	M7 = (M20*M02-M11*M11) / (m00*m00*m00*m00);

			//	std::cout<<"\nArea ="<<Area<<"\n";
			//	std::cout<<"\nM1 ="<<M1<<"\n";
			//	std::cout<<"M2 ="<<M2<<"\n";
			//	std::cout<<"M7 ="<<M7<<"\n";
			//	std::cout<<"Stosunek ="<<Ratio<<"\n";
			//	plik << M7;

			}
		}
		result.AddBlob(blobs.GetBlob(id));

		out_signs.write(result);
		out_draw.write(drawcont);

		newImage->raise();

		plik.close();
		return true;
	} catch (...) {
		LOG(LERROR) << "KW_PalmDetection::onNewImage failed\n";
		return false;
	}
}

bool KW_PalmDetection::onStop()
{
	return true;
}

bool KW_PalmDetection::onStart()
{
	return true;
}

void KW_PalmDetection::onNewImage()
{
	LOG(LTRACE) << "KW_PalmDetection::onNewImage\n";

	img_ready = true;
	tsl_img = in_img.read();
	//co robi tak linijka?
	tsl_img = tsl_img.clone();
	if (blobs_ready && img_ready)
		onStep();
}

void KW_PalmDetection::onNewBlobs()
{
	LOG(LTRACE) << "KW_PalmDetection::onNewBlobs\n";

	blobs_ready = true;
	blobs = in_blobs.read();
	if (blobs_ready && img_ready)
		onStep();
}

}//: namespace KW_Palm
}//: namespace Processors
