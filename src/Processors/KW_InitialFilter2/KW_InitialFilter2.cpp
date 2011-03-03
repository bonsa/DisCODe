/*!
 * \file KW_Initial_Filter.cpp
 * \brief Filtr, który odrzuci piksele, któ¶e na pewno nie sa skora
 * \author kwasak
 * \date 2010-03-01
 */

#include <memory>
#include <string>

#include "KW_InitialFilter2.hpp"
#include "Logger.hpp"

namespace Processors {
namespace KW_Filter2 {

// OpenCV writes hue in range 0..180 instead of 0..360
#define H(x) (x>>1)

KW_InitialFilter2::KW_InitialFilter2(const std::string & name) : Base::Component(name)
{
	LOG(LTRACE) << "Hello KW_InitialFilter2\n";
	k = 0;
}

KW_InitialFilter2::~KW_InitialFilter2()
{
	LOG(LTRACE) << "Good bye KW_InitialFilter2\n";
}

bool KW_InitialFilter2::onInit()
{
	LOG(LTRACE) << "KW_InitialFilter2::initialize\n";

	h_onNewImage.setup(this, &KW_InitialFilter2::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	registerStream("in_img", &in_img);

	newImage = registerEvent("newImage");

	registerStream("out_img", &out_img);



	return true;
}

bool KW_InitialFilter2::onFinish()
{
	LOG(LTRACE) << "KW_InitialFilter2::finish\n";

	return true;
}

bool KW_InitialFilter2::onStep()
{
	LOG(LTRACE) << "KW_InitialFilter2::step\n";
	return true;
}

bool KW_InitialFilter2::onStop()
{
	return true;
}

bool KW_InitialFilter2::onStart()
{
	return true;
}

void KW_InitialFilter2::onNewImage()
{
	LOG(LTRACE) << "KW_InitialFilter2::onNewImage\n";
	try {
		cv::Mat RGB_img = in_img.read();	//czytam obrazem w zejścia

		cv::Size size = RGB_img.size();		//rozmiar obrazka
		
		Filtered_img.create(size, CV_8UC3);

		// Check the arrays for continuity and, if this is the case,
		// treat the arrays as 1D vectors
		if (RGB_img.isContinuous()) {
			size.width *= size.height;
			size.height = 1;
		}
		size.width *= 3;

		for (int i = 0; i < size.height; i++) {
			// when the arrays are continuous,
			// the outer loop is executed only once
			// if not - it's executed for each row

			// get pointer to beggining of i-th row of input hsv image
			const uchar* RGB_p = RGB_img.ptr <uchar> (i);
			// get pointer to beggining of i-th row of output hue image
			uchar* Filtered_img_p = Filtered_img.ptr <uchar> (i);
			// get pointer to beggining of i-th row of output hue image

			uchar R, G, B;


			int j;
			for (j = 0; j < size.width; j += 3)
			{
				B = RGB_p[j];
				G = RGB_p[j + 1];
				R = RGB_p[j + 2];

				if((B > 160 && R < 180 && G < 180) || // too much blue
					(G > 160 && R < 180 && B < 180)|| // too much green
					(B < 30 && R < 30 && G < 30) || // too dark
					(G > 200) || //Green
					(R+G > 400) || // too much red and green (yellow like color)
					(G > 150 && B < 90)|| // too Yellow like also
					(1.0*B/(R+G+B) > 0.4)|| // too much blue in contrast to others
					(1.0*G/(R+G+B) > 0.4)|| // too much green in contrast to others
					(R < 102 && G > 100 && B > 110 && G < 140 && B <160))
					{
						Filtered_img_p[j] = 255;
						Filtered_img_p[j + 1] = 255;
						Filtered_img_p[j + 2] = 255;
					}
				else
				{
					Filtered_img_p[j] = RGB_p[j];
					Filtered_img_p[j + 1] = RGB_p[j + 1];
					Filtered_img_p[j + 2] = RGB_p[j + 2] ;
				}
				++k;
			}
		}

		out_img.write(Filtered_img);


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
		LOG(LERROR) << "KW_InitialFilter2::onNewImage failed\n";
	}
}

}//: namespace KW_Filter
}//: namespace Processors
