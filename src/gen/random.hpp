#pragma once

#include <cmath>
#include <random>
#include <limits>

namespace fin1
{
namespace random
{

// PCG XSL-RR generator, 64 bit output, 128 bit state. Adapted from the PCG
// library. Copyright M.E. O'Neill (2014), Apache License 2.0.
struct Generator
{
  static constexpr int rd_bits = std::numeric_limits<std::random_device::result_type>::digits;
  static constexpr auto mult = __uint128_t {2549297995355413924}<<64|4865540595714422341;
  __uint128_t state, inc;
  double normal_spare;
  bool normal_has_spare = false;

  static Generator deterministic
  ( __uint128_t state_seed, __uint128_t inc_seed
  )
  { Generator gen;
    gen.state = 0;
    gen.inc = inc_seed<<1|1; // inc must be odd
    gen.state = gen.state*mult+gen.inc;
    gen.state += state_seed;
    gen.state = gen.state*mult+gen.inc;
    return gen;
  }

  static Generator nondeterministic
  ()
  { static_assert(rd_bits <= 64);
    constexpr size_t n = (127+rd_bits)/rd_bits;
    __uint128_t state_seed = 0,
                  inc_seed = 0;
    std::random_device rd;
    for (size_t i = 0; i < n; i++)
      state_seed |= __uint128_t {rd()}<<rd_bits*i;
    for (size_t i = 0; i < n; i++)
      inc_seed   |= __uint128_t {rd()}<<rd_bits*i;
    return deterministic(state_seed, inc_seed);
  }

  uint64_t next
  ()
  { state = state*mult+inc;
    const uint64_t x = static_cast<uint64_t>(state>>64)^state,
                 rot = state>>122;
    return x>>rot|x<<(-rot&63);
  }

  void warmup
  ( size_t n_steps
  )
  { for (size_t i = 0; i < n_steps; i++)
      next();
  }

  double uniform
  ()
  { constexpr auto r_max_uint64_p1 = 1./(__uint128_t {1}<<64);
    return next()*r_max_uint64_p1;
  }

// Marsaglia polar method.
  double normal
  ()
  { if (normal_has_spare)
    { normal_has_spare = false;
      return normal_spare;
    }
    double u1, u2, s;
    for (;;)
    { u1 = uniform()*2-1;
      u2 = uniform()*2-1;
      s = pow(u1, 2)+pow(u2, 2);
      if (s != 0 && s < 1)
        break;
    }
    s = sqrt(-2*log(s)/s);
    normal_spare = u1*s;
    normal_has_spare = true;
    return u2*s;
  }

// Marsaglia, Tsang, 'A Simple Method for Generating Gamma Variables', 2001.
// For alpha >= 1, theta = 1.
  double gamma_1
  ( double alp
  )
  { const double d = alp-1./3,
                 c = 1/sqrt(9*d);
    for (;;)
    { double v = 0, n;
      const double u = uniform();
      for (; v <= 0;)
      { n = normal();
        v = 1+c*n;
      }
      v = pow(v, 3);
      if (u < 1-.0331*pow(n, 4) ||
          log(u) < .5*pow(n, 2)+d*(1-v+log(v)))
        return d*v;
    }
  }
// For alpha >= 1, any theta.
  double gamma_1
  ( double alp, double th
  )
  { return th*gamma_1(alp);
  }
// For theta = 1, any alpha.
  double gamma
  ( double alp
  )
  { return alp >= 1 ? gamma_1(alp)
                    : gamma_1(1+alp)*pow(uniform(), 1/alp);
  }
// For any alpha and theta.
  double gamma
  ( double alp, double th
  )
  { return th*gamma(alp);
  }

// Knuth, slow for large lambda.
  size_t Poisson_1
  ( double lam
  )
  { size_t x = -1;
    double p = 1;
    for (;;)
    { p *= uniform();
      x++;
      if (p <= exp(-lam))
        break;
    }
    return x;
  }
// Atkinson, 'The Computer Generation of Poisson Random Variables', 1979.
  size_t Poisson_2
  ( double lam
  )
  { const double bet = M_PI/sqrt(3*lam),
                 alp = bet*lam,
                 k = log((.767-3.36/lam)/bet)-lam;
    for (;;)
    { const double u1 = uniform(),
                   x = (alp-log(u1/(1-u1)))/bet;
      if (x < -.5)
        continue;
      const size_t n = x+.5;
      const double y = alp-bet*x,
                   L = y+log(uniform()/pow(1+exp(y), 2)),
                   R = k+n*log(lam)-lgamma(n+1);
      if (L <= R)
        return n;
    }
  }
// In general.
  size_t Poisson
  ( double lam
  )
  { return lam <= 30 ? Poisson_1(lam)
                     : Poisson_2(lam);
  }

// Fisher-Yates.
  template <typename T>
  void permute
  ( T *x, size_t n
  )
  { for (size_t i = n-1; i > 0; i--)
      std::swap(x[i], x[next()%(i+1)]);
  }
};

} // random
} // fin1
