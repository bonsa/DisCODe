/*!
 * \file KW_Palm_LUT.cpp
 * \brief
 * \author kwasak
 * \date 2010-11-05
 */

#include <memory>
#include <string>

#include "KW_Palm_LUT.hpp"
#include "Logger.hpp"

namespace Processors {
namespace KW_Palm {

// OpenCV writes hue in range 0..180 instead of 0..360
#define H(x) (x>>1)

KW_Palm_LUT::KW_Palm_LUT(const std::string & name) : Base::Component(name)
{
	LOG(LTRACE) << "Hello KW_Palm_LUT\n";
}

KW_Palm_LUT::~KW_Palm_LUT()
{
	LOG(LTRACE) << "Good bye KW_Palm_LUT\n";
}

bool KW_Palm_LUT::onInit()
{
	LOG(LTRACE) << "KW_Palm_LUT::initialize\n";

	h_onNewImage.setup(this, &KW_Palm_LUT::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	registerStream("in_img", &in_img);

	newImage = registerEvent("newImage");

	registerStream("out_img", &out_img);


	return true;
}

bool KW_Palm_LUT::onFinish()
{
	LOG(LTRACE) << "KW_Palm_LUT::finish\n";

	return true;
}

bool KW_Palm_LUT::onStep()
{
	LOG(LTRACE) << "KW_Palm_LUT::step\n";
	return true;
}

bool KW_Palm_LUT::onStop()
{
	return true;
}

bool KW_Palm_LUT::onStart()
{
	return true;
}

void KW_Palm_LUT::onNewImage()
{
	LOG(LTRACE) << "KW_Palm_LUT::onNewImage\n";
	try {
		cv::Mat hsv_img = in_img.read();	//czytam obrazu w wejścia

		cv::Size size = hsv_img.size();		//rozmiar obrazka

		skin_img.create(size, CV_8UC1);		//8bitów, 0-255, 1 kanał

		double lambda;
		float value;
		cv::Mat inv_cov(2, 2, CV_32FC1);	//odwrotna macierz kowariancji
		cv::Mat pixel(1, 2, CV_32FC1);

		// Check the arrays for continuity and, if this is the case,
		// treat the arrays as 1D vectors
		if (hsv_img.isContinuous())  {
			size.width *= size.height;
			size.height = 1;
		}
		size.width *= 3;


		for (int i = 0; i < size.height; i++) {

			// when the arrays are continuous,
			// the outer loop is executed only once
			// if not - it's executed for each row

			// get pointer to beggining of i-th row of input image
			const uchar* c_p = hsv_img.ptr <uchar> (i);
			// get pointer to beggining of i-th row of output hue image
			uchar* skin_p = skin_img.ptr <uchar> (i);


			int j = 0;
			for (j = 0; j < size.width; j += 3) {


				pixel.at<float>(0, 0) = c_p[j];
				pixel.at<float>(0, 1) = c_p[j+1];



				cv::invert(props.cov, inv_cov, CV_LU);



			/*	for (int i = 0; i < props.mean.size().width; ++i) {
					LOG(LERROR) << "Jestem w petli for do wyświetlania mean";
					LOG(LERROR) << "Mean[" << i << "] = " << props.mean.at<double>(0, i);
				}

				for (int i = 0; i < props.cov.size().height; ++i)
					for (int j = 0; j < props.cov.size().width; ++j)
						LOG(LERROR) << "Covar[" << i << "," << j << "] = " << props.cov.at<double>(i, j);
*/
	//			LOG(LERROR) << "KW_Palm_LUT:*******************PrzedMaha\n";
				lambda = cv::Mahalanobis(pixel, props.mean, props.cov);
		//		LOG(LERROR) << "KW_Palm_LUT:*******************poOdlMaha\n";
				value = c_p[j+2];

				if ((lambda <= props.lambda) && (value >= props.value)) {
					skin_p[j] = 255;
				}
				else {
					skin_p[j] = 0;
				}

//				hue_p[k] = hsv_p[j];
	//			sat_p[k] = hsv_p[j + 1];
				//seg_p[k] = hsv_p[j + 2];

//				seg_p[k] = 0;
/*				if ((hue_p[k] > H(0)) && (hue_p[k] < H(50)))
				{
					if((sat_p[k] > 23) && (sat_p[k] < 68))
					{
						seg_p[k] = 255;
					}
				}

				++k;
	*/		}
		}

		LOG(LERROR) << "KW_Palm_LUT:*******************888****8\n";
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
		LOG(LERROR) << "KW_Palm_LUT::onNewImage failed\n";
	}
}

}//: namespace KW_Palm
}//: namespace Processors
