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

namespace Processors {
namespace KW_Palm {


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
	registerStream("in_tsl", &in_tsl);

	newImage = registerEvent("newImage");

	registerStream("out_signs", &out_signs);

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

	blobs_ready = tsl_ready = false;

	try {
		int id = 0;
		int i;
		IplImage h = IplImage(tsl_img);
		Types::Blobs::Blob *currentBlob;
		Types::DrawableContainer signs; //kontener przechowujący elementy, które mozna narysować

		// iterate through all found blobs
		for (i = 0; i < blobs.GetNumBlobs(); i++ )
		{
			currentBlob = blobs.GetBlob(i);

			// get mean color from area coverd by blob (from hue component)
			double me = currentBlob->Mean(&h);
			double st = currentBlob->StdDev(&h);

			// get blob bounding rectangle and ellipse
			CvBox2D r2 = currentBlob->GetEllipse();

			// blob moments
			double m00, m10, m01, m11, m02, m20;
			double M11, M02, M20, M7;

			// probability that current blob is the one we need (roadsign) in range 0..255
			int prob;

			// ignore blobs that have only one color
			if (me < 130 || me > 155 || st < 20)
				continue;

			// calculate moments
			m00 = currentBlob->Moment(0,0);
			m01 = currentBlob->Moment(0,1);
			m10 = currentBlob->Moment(1,0);
			m11 = currentBlob->Moment(1,1);
			m02 = currentBlob->Moment(0,2);
			m20 = currentBlob->Moment(2,0);

			M11 = m11 - (m10*m01)/m00;
			M02 = m02 - (m01*m01)/m00;
			M20 = m20 - (m10*m10)/m00;

			// for circle it should be ~0.0063
			M7 = (M20*M02-M11*M11) / (m00*m00*m00*m00);

			// circle
			if (M7 < 0.007)
				prob = 255;
			// probably circle
			else if (M7 < 0.0085)
				prob = 128;
			// there is small chance that it's circular, but show them too
			else if (M7 < 0.011)
				prob = 10;
			else
			// not circle for sure
				continue;

			++id;

			signs.add(currentBlob);
		}

		out_signs.write(signs);

		newImage->raise();

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

	tsl_ready = true;
	tsl_img = in_tsl.read();
	//co robi tak linijka?
	tsl_img = tsl_img.clone();
	if (blobs_ready && tsl_ready)
		onStep();
}

void KW_PalmDetection::onNewBlobs()
{
	LOG(LTRACE) << "KW_PalmDetection::onNewBlobs\n";

	blobs_ready = true;
	blobs = in_blobs.read();
	if (blobs_ready && tsl_ready)
		onStep();
}

}//: namespace KW_Palm
}//: namespace Processors
