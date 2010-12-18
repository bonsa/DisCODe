#include <cmath>
#include <cstdio>

using namespace std;

int main() {
	int R, G, B;
	float T, S, L;

	float r_, g_;


	for (R = 0; R < 255 ; R = R + 10)
	{
		for (G = 0; G < 255; G = G + 10)
		{
			for (B = 0; B < 255; B = B + 10)
			{
				r_ = 1.0*R/(R+G+B+1)-1./3;
				g_ = 1.0*G/(R+G+B+1)-1./3;

				if (g_!=0)
				{
					T = atan(r_/g_)/3.141592 + 0.5;
				}
				else
				{
					T = 0;
				}
				S = sqrt(9.0/5.0*(r_*r_+g_*g_));

				printf("%d\t%d\t%d\t%.5f\t%.5f\t%.5f\t%.5f\n", R,G,B, r_,g_, T,S);
			}
		}
	}




	return 0;
}
