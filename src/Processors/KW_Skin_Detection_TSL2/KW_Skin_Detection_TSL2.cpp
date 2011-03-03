/*!
 * \file KW_Skin_Detection2.cpp
 * \brief Detekcja skory w przestrzni barw TSL
 * \author kwasak
 * \date 2010-03-01
 */

#include <memory>
#include <string>

#include "KW_Skin_Detection_TSL2.hpp"
#include "Logger.hpp"

namespace Processors {
namespace KW_Skin_TSL2 {

// OpenCV writes hue in range 0..180 instead of 0..360
#define H(x) (x>>1)

KW_Skin_Detection_TSL2::KW_Skin_Detection_TSL2(const std::string & name) : Base::Component(name)
{
	LOG(LTRACE) << "Hello KW_Skin_Detection_TSL2\n";
}

KW_Skin_Detection_TSL2::~KW_Skin_Detection_TSL2()
{
	LOG(LTRACE) << "Good bye KW_Skin_Detection_TSL2\n";
}

bool KW_Skin_Detection_TSL2::onInit()
{
	LOG(LTRACE) << "KW_Skin_Detection_TSL2::initialize\n";

	h_onNewImage.setup(this, &KW_Skin_Detection_TSL2::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	registerStream("in_img", &in_img);

	newImage = registerEvent("newImage");

	registerStream("out_img", &out_img);


	return true;
}

bool KW_Skin_Detection_TSL2::onFinish()
{
	LOG(LTRACE) << "KW_Skin_Detection_TSL2::finish\n";

	return true;
}

bool KW_Skin_Detection_TSL2::onStep()
{
	LOG(LTRACE) << "KW_Skin_Detection_TSL2::step\n";
	return true;
}

bool KW_Skin_Detection_TSL2::onStop()
{
	return true;
}

bool KW_Skin_Detection_TSL2::onStart()
{
	return true;
}

void KW_Skin_Detection_TSL2::onNewImage()
{
	LOG(LTRACE) << "KW_Skin_Detection_TSL2::onNewImage\n";
	try {
		cv::Mat TSL_img = in_img.read();	//czytam obrazu w wejścia

		cv::Size size = TSL_img.size();		//rozmiar obrazka

		skin_img.create(size, CV_8UC3);		//8bitów, 0-255, 1 kanał




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
			for (j = 0; j < size.width; j += 3) {

				if (c_p[j] < c_p[j+1] - 10)
				{
						skin_p[j] = 255;
						skin_p[j+1] = 255;
						skin_p[j+2] = 255;

				}
				else{
					skin_p[j] = c_p[j];
					skin_p[j+1] = c_p[j+1];
					skin_p[j+2] = c_p[j+2];
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
		LOG(LERROR) << "KW_Skin_Detection_TSL2::onNewImage failed\n";
	}
}

}//: namespace KW_Skin_TSL2
}//: namespace Processors
