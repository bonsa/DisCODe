/*!
 * \file CameraOpenCV.cpp
 * \brief Class responsible for retrieving images from camera.
 * \author mstefanc
 * \date 2010-06-10
 */

#include <iostream>

#include "CameraOpenCV_Source.hpp"
#include "Logger.hpp"

namespace Sources {
namespace CameraOpenCV {

CameraOpenCV_Source::CameraOpenCV_Source() {
	cout << "CameraOpenCV_Source::CameraOpenCV_Source()\n";
}

CameraOpenCV_Source::~CameraOpenCV_Source() {
	finish();

	cout << "CameraOpenCV_Source::~CameraOpenCV_Source()\n";
}

void CameraOpenCV_Source::initialize() {
	cout << "CameraOpenCV_Source::initialize()\n";
	newImage = registerEvent("newImage");

	registerStream("out_img", &out_img);

	cap.open(0);

	if (!cap.isOpened()) {
		throw Common::FraDIAException("Couldn't open camera device!");
	}
}


void CameraOpenCV_Source::finish() {
	cout << "CameraOpenCV_Source::finish()\n";
	cap.release();
}


int CameraOpenCV_Source::step() {

	cap >> frame;

	if (frame.empty()) {
		return 0;
	}

	out_img.write(frame);

	newImage->raise();
	return 0;
}


}//: namespace CameraOpenCV
}//: namespace Sources
