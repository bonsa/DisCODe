/*!
 * \file KW_SkinDetectionAkademik.cpp
 * \brief Detekcja skory w przestrzni barw TSL w Akademiku
 * \author kwasak
 * \date 2011-03-05
 */

#include <memory>
#include <string>

#include "KW_SkinDetectionAkademik.hpp"
#include "Logger.hpp"

namespace Processors {
namespace KW_SkinAkademik {

// OpenCV writes hue in range 0..180 instead of 0..360
#define H(x) (x>>1)

KW_SkinDetectionAkademik::KW_SkinDetectionAkademik(const std::string & name) : Base::Component(name)
{
	LOG(LTRACE) << "Hello KW_SkinDetectionAkademik\n";
}

KW_SkinDetectionAkademik::~KW_SkinDetectionAkademik()
{
	LOG(LTRACE) << "Good bye KW_SkinDetectionAkademik\n";
}

bool KW_SkinDetectionAkademik::onInit()
{
	LOG(LTRACE) << "KW_SkinDetectionAkademik::initialize\n";

	h_onNewImage.setup(this, &KW_SkinDetectionAkademik::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	registerStream("in_img", &in_img);

	newImage = registerEvent("newImage");

	registerStream("out_img", &out_img);


	return true;
}

bool KW_SkinDetectionAkademik::onFinish()
{
	LOG(LTRACE) << "KW_SkinDetectionAkademik::finish\n";

	return true;
}

bool KW_SkinDetectionAkademik::onStep()
{
	LOG(LTRACE) << "KW_SkinDetectionAkademik::step\n";
	return true;
}

bool KW_SkinDetectionAkademik::onStop()
{
	return true;
}

bool KW_SkinDetectionAkademik::onStart()
{
	return true;
}

void KW_SkinDetectionAkademik::onNewImage()
{
	LOG(LTRACE) << "KW_SkinDetectionAkademik::onNewImage\n";
	try {
		cv::Mat TSL_img = in_img.read();	//czytam obrazu w wejścia

		cv::Size size = TSL_img.size();		//rozmiar obrazka

		skin_img.create(size, CV_8UC1);		//8bitów, 0-255, 1 kanał




		// Check the arrays for continuity and, if this is the case,
		// treat the arrays as 1D vectors
		if (TSL_img.isContinuous() && skin_img.isContinuous())  {
			size.width *= size.height;
			size.height = 1;
		}
		size.width *= 3;

		for (int i = 0; i < size.height; i++) {

			// when the arrays are continuous,
			// the outer loop is executed only once
			// if not - it's executed for each row

			// get pointer to beggining of i-th row of input image
			const uchar* c_p = TSL_img.ptr <uchar> (i);
			// get pointer to beggining of i-th row of output hue image
			uchar* skin_p = skin_img.ptr <uchar> (i);


			int j,k = 0;
			for (j = 0; j < size.width; j += 3) 
			{
				if((c_p[j]>200))
				{
					skin_p[k] = 0;
		
				}


				else if ((c_p[j]>40)&&(c_p[j]<120) && (c_p[j+1]<40) && (c_p[j+1]>5))
				{
					skin_p[k] = 255;
	
				}
				else if (c_p[j+2]<140)
				{
					skin_p[k] = 0;
	
				}

				else{
					skin_p[k] = 0;
	
				}


				++k;
			}
		}


		out_img.write(skin_img);


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
		LOG(LERROR) << "KW_SkinDetectionAkademik::onNewImage failed\n";
	}
}

}//: namespace KW_SkinAkademik
}//: namespace Processors
