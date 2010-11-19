/*!
 * \file KW_Palm_Description.cpp
 * \brief
 * \author kwasak
 * \date 2010-11-19
 */

#include <memory>
#include <string>

#include "KW_Palm_Description.hpp"
#include "Logger.hpp"

#include "Types/Ellipse.hpp"

namespace Processors {
namespace KW_Palm {

// OpenCV writes hue in range 0..180 instead of 0..360
#define H(x) (x>>1)

KW_Palm_Description::KW_Palm_Description(const std::string & name) : Base::Component(name)
{
	LOG(LTRACE) << "Hello KW_Palm_Description\n";
}

KW_Palm_Description::~KW_Palm_Description()
{
	LOG(LTRACE) << "Good bye KW_Palm_Description\n";
}

bool KW_Palm_Description::onInit()
{
	LOG(LTRACE) << "KW_Palm_Description::initialize\n";

	h_onNewImage.setup(this, &KW_Palm_Description::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	h_onNewBlobs.setup(this, &KW_Palm_Description::onNewBlobs);
	registerHandler("onNewBlobs", &h_onNewBlobs);

	registerStream("in_blobs", &in_blobs);
	registerStream("in_binary", &in_binary);

	newImage = registerEvent("newImage");

	//opisana dłon
	registerStream("out_dpalm", &out_dpalm);

	return true;
}

bool KW_Palm_Description::onFinish()
{
	LOG(LTRACE) << "KW_Palm_Description::finish\n";

	return true;
}

bool KW_Palm_Description::onStep()
{
	return true;
}

bool KW_Palm_Description::onStop()
{
	return true;
}

bool KW_Palm_Description::onStart()
{
	return true;
}

void KW_Palm_Description::onNewImage()
{
	LOG(LTRACE) << "KW_Palm_Description::onNewImage\n";


	binary_img = in_binary.read();
	//****************************po co?***************//
	///hue_img = hue_img.clone();

	LOG(LTRACE) << "KW_Palm_Description::step\n";

		try {
			int i;
			IplImage h = IplImage(binary_img);
			Types::Blobs::Blob *currentBlob;
			Types::DrawableContainer signs;

			// iterate through all found blobs
			for (i = 0; i < blobs.GetNumBlobs(); i++ )
			{
				currentBlob = blobs.GetBlob(i);


				// blob moments
				double m00, m10, m01, m11, m02, m20;
				double M11, M02, M20, M7;




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


			}

			out_dpalm.write(signs);

			newImage->raise();


		} catch (...) {
			LOG(LERROR) << "KW_Palm_Description::onNewImage failed\n";

		}
}




}//: namespace KW_Palm
}//: namespace Processors
