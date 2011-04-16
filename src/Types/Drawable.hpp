#ifndef DRAWABLE_HPP_
#define DRAWABLE_HPP_

#include <cv.h>
#include <iostream>

namespace Types {

/*!
 * \class Drawable
 *
 * \brief Interface used for representation of drawable objects.
 *
 * Each object that implements this interface must implement draw method.
 *
 * \author mstefanc
 */
class Drawable {
public:
	Drawable() {
		m_col = CV_RGB(0,255,0);
	}

	virtual ~Drawable() {}

	virtual void draw(cv::Mat& image, CvScalar color, int offsetX = 0, int offsetY = 0) = 0;

	virtual Drawable * clone() {
		return NULL;
	}

	void setCol(CvScalar col) {
		m_col = col;
	}

	CvScalar getCol() {
		return m_col;
	}

protected:
	CvScalar m_col;
};

} //: namespace Types

#endif /* DRAWABLE_HPP_ */
