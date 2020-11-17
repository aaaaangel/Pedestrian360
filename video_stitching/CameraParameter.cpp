#include<iostream>
#include <fstream>
#include "CameraParameter.h"

using namespace std;

pc::CameraParameter::CameraParameter(string ParameterFileName){
    ifstream is(ParameterFileName);
    for (int r = 0; r != 3; ++r)
    {
        for (int c = 0; c != 3; ++c)
        {
            is >> R(r, c);
        }
        is >> t(r);
    }
    for (int r = 0; r != 3; ++r)
        for (int c = 0; c != 3; ++c)
        {
            is >> K(r, c);
        }
    for (int j = 0; j != 4; ++j)
    {
        is >> D(j);
    }
}