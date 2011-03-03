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

KW_InitialFilter2::KW_InitialFilter2(const std::string & name) : Base::Component(name),

		blue_B("down_blue_B", 160, "range"),
		blue_R("up_blue_R", 180, "range"),
		blue_G("up_blue_G", 180, "range"),

		green_G("down_green_G", 160, "range"),
		green_R("up_green_R", 180, "range"),
		green_B("up_green_B", 180, "range"),

		dark_R("up_dark_R", 40, "range"),
		dark_G("up_dark_G", 40, "range"),
		dark_B("up_dark_B", 40, "range"),

		Yellow_G("down_Yellow_G", 150, "range"),
		Yellow_B("up_Yellow_B", 90, "range")

{
	LOG(LTRACE) << "Hello KW_InitialFilter2\n";
	k = 0;

	blue_R.addConstraint("0");
	blue_R.addConstraint("255");

	blue_G.addConstraint("0");
	blue_G.addConstraint("255");

	blue_B.addConstraint("0");
	blue_B.addConstraint("255");

	registerProperty(blue_R);
	registerProperty(blue_G);
	registerProperty(blue_B);


	dark_R.addConstraint("0");
	dark_R.addConstraint("255");

	dark_G.addConstraint("0");
	dark_G.addConstraint("255");

	dark_B.addConstraint("0");
	dark_B.addConstraint("255");

	registerProperty(dark_R);
	registerProperty(dark_G);
	registerProperty(dark_B);

	Yellow_G.addConstraint("0");
	Yellow_G.addConstraint("255");

	Yellow_B.addConstraint("0");
	Yellow_B.addConstraint("255");

	registerProperty(Yellow_G);
	registerProperty(Yellow_B);
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

				if((B > blue_B && R < blue_R && G < blue_G) || // too much blue
					(G > green_G && R < green_R && B < green_B)|| // too much green
					(B < dark_B && R < dark_R && G < dark_G) || // too dark
					(G > 200) || //Green
					(R+G > 400) || // too much red and green (yellow like color)
					(G > Yellow_G && B < Yellow_B)|| // too Yellow like also
					(1.0*B/(R+G+B) > 0.4)|| // too much blue in contrast to others
					(1.0*G/(R+G+B) > 0.4)//|| // too much green in contrast to others
				//	(R < 102 && G > 100 && B > 110 && G < 140 && B <160)
					)
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
