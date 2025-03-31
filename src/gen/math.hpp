#pragma once

#include <cmath>
#include <cstddef>

namespace fin1
{
namespace math
{

// [L,R)
double linspace
( double L, double R, double i, size_t n = 1
)
{ return L+(R-L)*i/n;
}
double linspace
( const double LR[2], double i, size_t n = 1
)
{ return linspace(LR[0], LR[1], i, n);
}

// Unbiased mean.
double mean
( const double *x, size_t size_x
)
{ double ret = 0;
  for (size_t i = 0; i < size_x; i++)
    ret += x[i];
  return ret/size_x;
}

// Unbiased variance.
double var
( const double *x, size_t size_x, double mean_x
)
{ double ret = 0;
  for (size_t i = 0; i < size_x; i++)
    ret += pow(x[i]-mean_x, 2);
  return ret/(size_x-1);
}
double var
( const double *x, size_t size_x
)
{ return var(x, size_x, mean(x, size_x));
}

} // math
} // fin1
