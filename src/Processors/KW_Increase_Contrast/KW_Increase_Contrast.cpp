/*!
 * \file KW_Increase_Contrast.cpp
 * \brief Zwięszenie Kotrastu barw TS
 * \author kwasak
 * \date 2010-12-17
 */

#include <memory>
#include <string>

#include "KW_Increase_Contrast.hpp"
#include "Logger.hpp"

namespace Processors {
namespace KW_Contrast {

// OpenCV writes hue in range 0..180 instead of 0..360
#define H(x) (x>>1)

KW_Increase_Contrast::KW_Increase_Contrast(const std::string & name) : Base::Component(name)
{
	LOG(LTRACE) << "Hello KW_Increase_Contrast\n";
	k = 0;
}

KW_Increase_Contrast::~KW_Increase_Contrast()
{
	LOG(LTRACE) << "Good bye KW_Initial_Filter\n";
}

bool KW_Increase_Contrast::onInit()
{
	LOG(LTRACE) << "KW_Initial_Filter::initialize\n";

	h_onNewImage.setup(this, &KW_Increase_Contrast::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	registerStream("in_img", &in_img);

	newImage = registerEvent("newImage");

	registerStream("out_img", &out_img);



	return true;
}

bool KW_Increase_Contrast::onFinish()
{
	LOG(LTRACE) << "KW_Initial_Filter::finish\n";

	return true;
}

bool KW_Increase_Contrast::onStep()
{
	LOG(LTRACE) << "KW_Increase_Contrast::step\n";
	return true;
}

bool KW_Increase_Contrast::onStop()
{
	return true;
}

bool KW_Increase_Contrast::onStart()
{
	return true;
}

void KW_Increase_Contrast::onNewImage()
{
	LOG(LTRACE) << "KW_Increase_Contrast::onNewImage\n";
	try {
		cv::Mat TSL_img = in_img.read();	//czytam obrazem w zejścia
		Processors::KW_TSL::min_max MinMax = in_img2.read();
		
		cv::Size size = TSL_img.size();		//rozmiar obrazka
				

		// Check the arrays for continuity and, if this is the case,
		// treat the arrays as 1D vectors
		if (TSL_img.isContinuous()) {
			size.width *= size.height;
			size.height = 1;
		}
		size.width *= 3;

		for (int i = 0; i < size.height; i++) {
			// when the arrays are continuous,
			// the outer loop is executed only once
			// if not - it's executed for each row

			// get pointer to beggining of i-th row of input hsv image
			uchar* TSL_p = TSL_img.ptr <uchar> (i);
			// get pointer to beggining of i-th row of output hue image
			
			int j;
			for (j = 0; j < size.width; j += 3)
			{
				//T
				TSL_p[j] = (TSL_p[j] - MinMax.minS)/(MinMax.maxT- MinMax.minS) ;
				//S
				TSL_p[j + 1] = (TSL_p[j + 1] - MinMax.minS )/(MinMax.maxS- MinMax.minS );
			}
		}

		out_img.write(TSL_img);


		newImage->raise();
		

	}
	catch (Common::DisCODeException& ex) {
		LOG(LERROR) << ex.what() << "\n";
		ex.printStackTrace();
		exit(EXIT_FAILURE);
	}
	catch (const char * ex) {
		LOG(LERROR) << ex;
	}
	catch (...) {
		LOG(LERROR) << "KW_Increase_Contrast::onNewImage failed\n";
	}
}

}//: namespace KW_Filter
}//: namespace Processors
