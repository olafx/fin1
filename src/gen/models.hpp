#pragma once

#include "random.hpp"
#include "math.hpp"

#include <omp.h>

#include <cassert>
#include <cfloat>
#include <vector>

// TODO: temp
#include <print>

namespace fin1
{
namespace models
{

// Black-Scholes-Merton
struct BSM
{ double S0,  // initial spot price
         r,   // risk-free interest rate
         q,   // dividend rate
         sig, // volatility
         T;   // time to expiry

  double discount_factor
  () const
  { return exp(-r*T);
  }

// Generate a path via Euler-Maruyama/Milstein.
  void gen_path
  ( random::Generator &gen,
    double *S, size_t size_S
  ) const
  { const double dt = T/(size_S-1), 
            sqrt_dt = sqrt(dt);
    S[0] = S0;
    for (size_t i = 1; i < size_S; i++)
    { const double x = gen.normal();
      S[i] = S[i-1]*(1+(r-q)*dt+sig*sqrt_dt*x);
    }
      
  }

// Basic Monte Carlo pricing of an option.
  double price
  ( random::Generator *gen_ths, size_t n_ths,
    double *const *S_ths, size_t size_S,
    const auto &option,
    size_t n_MC
  ) const
  { double V0 = 0;
    #pragma omp parallel for num_threads(n_ths) reduction(+:V0)
    for (size_t i = 0; i < n_MC; i++)
    { const int i_th = omp_get_thread_num();
      auto &gen = gen_ths[i_th];
      auto *S = S_ths[i_th];
      gen_path(gen, S, size_S);
      V0 += option.price(S, size_S);
    }
    return V0*discount_factor()/n_MC;
  }

// Generate a path via Euler-Maruyama/Milstein under a modified probability
// measure and return the Radon-Nikodym derivative. (See attached documentation
// for details.) Provides functionality for the method of antithetic variates.
  template <int i_av = +1>
  double gen_path_1
  ( random::Generator &gen,
    double *S, size_t size_S,
    double lam
  ) const
  requires (i_av == +1 || i_av == -1)
  { const double dt = T/(size_S-1), 
            sqrt_dt = sqrt(dt);
    S[0] = S0;
    double WT = 0;
    for (size_t i = 1; i < size_S; i++)
    { const double x = i_av*gen.normal();
      WT += x;
      S[i] = S[i-1]*(1+(r-q-sig*lam)*dt+sig*sqrt_dt*x);
    }
    WT *= sqrt_dt;
    return exp(lam*(WT-.5*lam*T));
  }

// Monte Carlo pricing an option with importance sampling and antithetic paths.
// (See attached documentation for details.)
  double price_1
  ( random::Generator *gen_ths, size_t n_ths,
    double *const *S_ths, size_t size_S,
    const auto &option,
    double target_ST,
    size_t n_MC
  ) const
  { const double lam = log(S0/target_ST)/(sig*T)+(r-q)/sig-.5*sig;
    double V0 = 0;
    #pragma omp parallel for num_threads(n_ths) reduction(+:V0)
    for (size_t i = 0; i < n_MC/2; i++)
    { const int i_th = omp_get_thread_num();
      auto &gen = gen_ths[i_th];
      auto *S = S_ths[i_th];
      auto gen_old = gen;
      double RN;
      RN = gen_path_1<+1>(gen,     S, size_S, lam);
      V0 += RN*option.price(S, size_S);
      RN = gen_path_1<-1>(gen_old, S, size_S, lam);
      V0 += RN*option.price(S, size_S);
    }
    return V0*discount_factor()/n_MC;
  }

// Monte Carlo pricing an option with adaptive importance sampling (robust
// stochastic gradient descent) and antithetic paths. (See attached
// documentation for details.) Steps of size D_lam are made to lambda starting
// from 0, following the stochastic gradient in variance if there is any,
// otherwise a random walk is taken. n_est_var (even) samples are used to
// estimate the variance, which should for performance be at least 2-3 orders of
// magnitude greater than the number of threads used. Integration stops when the
// desired relative error (estimate) or maximum number of iterations has been
// reached. A struct is returned with the price, as well as some numerical
// information.
  struct price_2_t
  { double V0, rel_err, lam;
    size_t n_MC;
  };
  price_2_t price_2
  ( random::Generator *gen_ths, size_t n_ths,
    double *const *S_ths, size_t size_S,
    const auto &option,
    size_t n_MC_max, double rel_err = 0,
    double D_lam = .05,
    size_t n_est_var = 1024
  ) const
  { assert(n_est_var % 2 == 0);
    bool done = false;
    double V0   = 0,
           V0_2 = 0;
    double rel_err_V0 = 0;
    int i_lam = 0;
    price_2_t ret { .n_MC = n_MC_max/(2*n_est_var)*(2*n_est_var) };
    std::vector<double> last_V0_L(n_est_var),
                        last_V0_R(n_est_var);
    #pragma omp parallel num_threads(n_ths) reduction(+:V0)
    { const int i_th = omp_get_thread_num();
      auto &gen = gen_ths[i_th];
      auto *S = S_ths[i_th];
      for (size_t i = 0; i < n_MC_max/(2*n_est_var); i++)
      {
        const double lam_L = (i_lam-1)*D_lam,
                     lam_R = (i_lam+1)*D_lam;
        #pragma omp for nowait
        for (size_t j = 0; j < n_est_var/2; j++)
        { auto gen_old = gen;
          double RN;
          RN = gen_path_1<+1>(gen,     S, size_S, lam_L);
          last_V0_L[2*j  ] = RN*option.price(S, size_S);
          RN = gen_path_1<-1>(gen_old, S, size_S, lam_L);
          last_V0_L[2*j+1] = RN*option.price(S, size_S);
          const double V0_inc = last_V0_L[2*j  ]+
                                last_V0_L[2*j+1];
          V0   +=     V0_inc;
          V0_2 += pow(V0_inc, 2);
        }
        #pragma omp for
        for (size_t j = 0; j < n_est_var/2; j++)
        { auto gen_old = gen;
          double RN;
          RN = gen_path_1<+1>(gen,     S, size_S, lam_R);
          last_V0_R[2*j  ] = RN*option.price(S, size_S);
          RN = gen_path_1<-1>(gen_old, S, size_S, lam_R);
          last_V0_R[2*j+1] = RN*option.price(S, size_S);
          const double V0_inc = last_V0_R[2*j  ]+
                                last_V0_R[2*j+1];
          V0   +=     V0_inc;
          V0_2 += pow(V0_inc, 2);
        }
        #pragma omp single
        { double var_V0_L = math::var(last_V0_L.data(), n_est_var),
                 var_V0_R = math::var(last_V0_R.data(), n_est_var);
          if (var_V0_L == 0) var_V0_L = DBL_MAX;
          if (var_V0_R == 0) var_V0_R = DBL_MAX;
          if (var_V0_L == var_V0_R)
            i_lam += gen.uniform() < .5 ? -1 : +1;
          else
          { i_lam += var_V0_L < var_V0_R ? -1 : +1;
            ret.n_MC = (i+1)*2*n_est_var;
            const double var_V0 = V0_2/ret.n_MC-pow(V0/ret.n_MC, 2);
            rel_err_V0 = V0 == 0 ? DBL_MAX : sqrt(var_V0*ret.n_MC)/V0;
            done = rel_err_V0 < rel_err;
          }
        }
        if (done) break;
      }
    }
    ret.V0 = V0*discount_factor()/ret.n_MC;
    ret.rel_err = rel_err_V0;
    ret.lam = i_lam*D_lam;
    return ret;
  }
};

struct Heston
{ double S0,   // initial spot price
         vol0, // initial spot volatility
         r,    // risk-free interest rate
         q,    // dividend rate
         eta,  // level of mean reversion
         kap,  // spread of mean reversion
         th,   // vol-of-vol
         rho,  // vol-stock correlation
         T;    // time to expiry

  double discount_factor
  () const
  { return exp(-r*T);
  }

  bool Feller_cond
  () const
  { return 2*kap*eta >= pow(th, 2);
  }

// Generate a path via Milstein with reflection.
  void gen_path
  ( random::Generator &gen,
    double *S, double *vol2, size_t size_S
  ) const
  { const double dt = T/(size_S-1),
            sqrt_dt = sqrt(dt);
    S[0] = S0;
    vol2[0] = pow(vol0, 2);
    for (size_t i = 1; i < size_S; i++)
    { const double x1 = gen.normal(),
                   x2 = rho*x1+sqrt(1-pow(rho, 2))*gen.normal();
      const double vol = sqrt(std::max(0., vol2[i-1]));
      S[i] = S[i-1]*(1+(r-q)*dt+vol*sqrt_dt*x1);
      vol2[i] = vol2[i-1]+kap*(eta-vol2[i-1])*dt
               +th*vol*sqrt_dt*x2
               +.25*pow(th, 2)*dt*(pow(x2, 2)-1);
    }
  }

// Basic Monte Carlo pricing of an option.
  double price
  ( random::Generator *gen_ths, size_t n_ths,
    double *const *S_ths, double *const *vol2_ths, size_t size_S,
    const auto &option,
    size_t n_MC
  ) const
  { double V0 = 0;
    #pragma omp parallel for num_threads(n_ths) reduction(+:V0)
    for (size_t i = 0; i < n_MC; i++)
    { const int i_th = omp_get_thread_num();
      auto &gen = gen_ths[i_th];
      auto *S = S_ths[i_th];
      auto *vol2 = vol2_ths[i_th];
      gen_path(gen, S, vol2, size_S);
      V0 += option.price(S, size_S);
    }
    return V0*discount_factor()/n_MC;
  }

// Generate a path via Milstein with reflection under a modified probability
// measure and return the Radon-Nikodym derivative. (See attached documentation
// for details.) Provides functionality for the method of antithetic variates.
  template <int i_av = +1>
  double gen_path_1
  ( random::Generator &gen,
    double *S, double *vol2, size_t size_S,
    double lam
  ) const
  requires (i_av == +1 || i_av == -1)
  { const double dt = T/(size_S-1),
            sqrt_dt = sqrt(dt);
    S[0] = S0;
    vol2[0] = pow(vol0, 2);
    double WT1 = 0,
           WT2 = 0;
    for (size_t i = 1; i < size_S; i++)
    { const double x1 = i_av*gen.normal(),
                   x2 = i_av*gen.normal(),
                   x3 = rho*x1+sqrt(1-pow(rho, 2))*x2;
      WT1 += x1;
      WT2 += x2;
      const double vol = sqrt(std::max(0., vol2[i-1]));
      S[i] = S[i-1]*(1+(r-q-lam*vol)*dt+vol*sqrt_dt*x1);
      vol2[i] = vol2[i-1]+kap*(eta-vol2[i-1])*dt
                +th*vol*sqrt_dt*x3
                +.25*pow(th, 2)*dt*(pow(x3, 2)-1);
    }
    WT1 *= sqrt_dt;
    WT2 *= sqrt_dt;
    return exp(lam*WT1
              -lam*rho/sqrt(1-pow(rho, 2))*WT2
              -.5*pow(lam, 2)*T/(1-pow(rho, 2)));
  }

// Monte Carlo pricing an option with importance sampling and antithetic paths.
// (See attached documentation for details.)
  double price_1
  ( random::Generator *gen_ths, size_t n_ths,
    double *const *S_ths, double *const *vol2_ths, size_t size_S,
    const auto &option,
    double lam,
    size_t n_MC
  ) const
  { double V0 = 0;
    #pragma omp parallel for num_threads(n_ths) reduction(+:V0)
    for (size_t i = 0; i < n_MC/2; i++)
    { const int i_th = omp_get_thread_num();
      auto &gen = gen_ths[i_th];
      auto *S = S_ths[i_th];
      auto *vol2 = vol2_ths[i_th];
      auto gen_old = gen;
      double RN;
      RN = gen_path_1<+1>(gen,     S, vol2, size_S, lam);
      V0 += RN*option.price(S, size_S);
      RN = gen_path_1<-1>(gen_old, S, vol2, size_S, lam);
      V0 += RN*option.price(S, size_S);
    }
    return V0*discount_factor()/n_MC;
  }

// Monte Carlo pricing an option with adaptive importance sampling (robust
// stochastic gradient descent) and antithetic paths. (See attached
// documentation for details.) Steps of size D_lam are made to lambda starting
// from 0, following the stochastic gradient in variance if there is any,
// otherwise a random walk is taken. n_est_var (even) samples are used to
// estimate the variance, which should for performance be at least 2-3 orders of
// magnitude greater than the number of threads used. Integration stops when the
// desired relative error (estimate) or maximum number of iterations has been
// reached. A struct is returned with the price, as well as some numerical
// information.
  struct price_2_t
  { double V0, rel_err, lam;
    size_t n_MC;
  };
  price_2_t price_2
  ( random::Generator *gen_ths, size_t n_ths,
    double *const *S_ths, double *const *vol2_ths, size_t size_S,
    const auto &option,
    size_t n_MC_max, double rel_err = 0,
    double D_lam = .05,
    size_t n_est_var = 1024
  ) const
  { assert(n_est_var % 2 == 0);
    bool done = false;
    double V0   = 0,
           V0_2 = 0;
    double rel_err_V0 = 0;
    int i_lam;
    price_2_t ret { .n_MC = n_MC_max/(2*n_est_var)*(2*n_est_var) };
    std::vector<double> last_V0_L(n_est_var),
                        last_V0_R(n_est_var);
    #pragma omp parallel num_threads(n_ths) reduction(+:V0)
    { const int i_th = omp_get_thread_num();
      auto &gen = gen_ths[i_th];
      auto *S = S_ths[i_th];
      auto *vol2 = vol2_ths[i_th];
      for (size_t i = 0; i < n_MC_max/(2*n_est_var); i++)
      { const double lam_L = (i_lam-1)*D_lam,
                     lam_R = (i_lam+1)*D_lam;
        #pragma omp for nowait
        for (size_t j = 0; j < n_est_var/2; j++)
        { auto gen_old = gen;
          double RN;
          RN = gen_path_1<+1>(gen,     S, vol2, size_S, lam_L);
          last_V0_L[2*j  ] = RN*option.price(S, size_S);
          RN = gen_path_1<-1>(gen_old, S, vol2, size_S, lam_L);
          last_V0_L[2*j+1] = RN*option.price(S, size_S);
          const double V0_inc = last_V0_L[2*j  ]+
                                last_V0_L[2*j+1];
          V0   +=     V0_inc;
          V0_2 += pow(V0_inc, 2);
        }
        #pragma omp for
        for (size_t j = 0; j < n_est_var/2; j++)
        { auto gen_old = gen;
          double RN;
          RN = gen_path_1<+1>(gen,     S, vol2, size_S, lam_R);
          last_V0_R[2*j  ] = RN*option.price(S, size_S);
          RN = gen_path_1<-1>(gen_old, S, vol2, size_S, lam_R);
          last_V0_R[2*j+1] = RN*option.price(S, size_S);
          const double V0_inc = last_V0_R[2*j  ]+
                                last_V0_R[2*j+1];
          V0   +=     V0_inc;
          V0_2 += pow(V0_inc, 2);
        }
        #pragma omp single
        { double var_V0_L = math::var(last_V0_L.data(), n_est_var),
                 var_V0_R = math::var(last_V0_R.data(), n_est_var);
          if (var_V0_L == 0) var_V0_L = DBL_MAX;
          if (var_V0_R == 0) var_V0_R = DBL_MAX;
          if (var_V0_L == var_V0_R)
            i_lam += gen.uniform() < .5 ? -1 : +1;
          else
          { i_lam += var_V0_L < var_V0_R ? -1 : +1;
            ret.n_MC = (i+1)*2*n_est_var;
            const double var_V0 = V0_2/ret.n_MC-pow(V0/ret.n_MC, 2);
            rel_err_V0 = V0 == 0 ? DBL_MAX : sqrt(var_V0*ret.n_MC)/V0;
            done = rel_err_V0 < rel_err;
          }
        }
        if (done) break;
      }
    }
    ret.V0 = V0*discount_factor()/ret.n_MC;
    ret.rel_err = rel_err_V0;
    ret.lam = i_lam*D_lam;
    return ret;
  }
};

struct BG
{ double S0,    // initial spot price
         r,     // risk-free interest rate
         q,     // dividend rate
         al_p,  // upward shape parameter
         al_m,  // downward shape parameter
         lam_p, // upward rate parameter
         lam_m, // downward rate parameter
         T;     // time to expiry

  double discount_factor
  () const
  { return exp(-r*T);
  }

  bool param_cond
  () const
  { return lam_p > 1 && lam_m > 0;
  }

// Generate a path, as the difference of gamma processes.
  void gen_path
  ( random::Generator &gen,
    double *S, size_t size_S
  ) const
  { [[assume(lam_p > 1)]];
    [[assume(lam_m > 0)]];
    const double dt = T/(size_S-1),
                 xi = -al_p*log(lam_p/(lam_p-1))-al_m*log(lam_m/(lam_m+1));
    S[0] = S0;
    double X = 0;
    for (size_t i = 1; i < size_S; i++)
    { const double x_p = gen.gamma(al_p*dt, 1/lam_p),
                   x_m = gen.gamma(al_m*dt, 1/lam_m);
      X += x_p-x_m;
      const double t = math::linspace(0, T, i, size_S);
      S[i] = S0*exp((r-q+xi)*t+X);
    }
  }

// Basic Monte Carlo pricing of an option.
  double price
  ( random::Generator *gen_ths, size_t n_ths,
    double *const *S_ths, size_t size_S,
    const auto &option,
    size_t n_MC
  ) const
  { double V0 = 0;
    #pragma omp parallel for num_threads(n_ths) reduction(+:V0)
    for (size_t i = 0; i < n_MC; i++)
    { const int i_th = omp_get_thread_num();
      auto &gen = gen_ths[i_th];
      auto *S = S_ths[i_th];
      gen_path(gen, S, size_S);
      V0 += option.price(S, size_S);
    }
    return V0*discount_factor()/n_MC;
  }

// Generate a path as the difference of gamma process under a modified
// probability measure and return the Radon-Nikodym derivative. (See attached
// documentation for details.)
  double gen_path_1
  ( random::Generator &gen,
    double *S, size_t size_S,
    double lam_p_Q, double lam_m_Q
  ) const
  { [[assume(lam_p > 1)]];
    [[assume(lam_m > 0)]];
    const double dt = T/(size_S-1),
                 xi = -al_p*log(lam_p/(lam_p-1))-al_m*log(lam_m/(lam_m+1));
    S[0] = S0;
    double X    = 0,
           XT_p = 0,
           XT_m = 0;
    for (size_t i = 1; i < size_S; i++)
    { const double x_p = gen.gamma(al_p*dt, 1/lam_p_Q),
                   x_m = gen.gamma(al_m*dt, 1/lam_m_Q);
      XT_p += x_p;
      XT_m += x_m;
      X += x_p-x_m;
      const double t = math::linspace(0, T, i, size_S);
      S[i] = S0*exp((r-q+xi)*t+X);
    }
    return pow(lam_p/lam_p_Q, al_p*T)
          *pow(lam_m/lam_m_Q, al_m*T)
          *exp(-(lam_p-lam_p_Q)*XT_p
               -(lam_m-lam_m_Q)*XT_m);
  }

// Monte Carlo pricing an option with importance sampling. (See attached
// documentation for details.)
  double price_1
  ( random::Generator *gen_ths, size_t n_ths,
    double *const *S_ths, size_t size_S,
    const auto &option,
    double lam_p_Q, double lam_m_Q,
    size_t n_MC
  ) const
  { double V0 = 0;
    #pragma omp parallel for num_threads(n_ths) reduction(+:V0)
    for (size_t i = 0; i < n_MC; i++)
    { const int i_th = omp_get_thread_num();
      auto &gen = gen_ths[i_th];
      auto *S = S_ths[i_th];
      const double RN = gen_path_1(gen, S, size_S, lam_p_Q, lam_m_Q);
      V0 += RN*option.price(S, size_S);
    }
    return V0*discount_factor()/n_MC;
  }

// Monte Carlo pricing an option with adaptive importance sampling (robust
// stochastic gradient descent). (See attached documentation for details.)
// Steps of size D_lam are made to the rates lam^+ and lam^-, following
// the stochastic gradient in variance if there is any, otherwise a random walk
// is taken. The rates are kept within their respective meaningful regimes,
// lam^+ > 1 and lam^- > 0. n_est_var samples are used to estimate the variance,
// which should for performance be at least 2-3 orders of magnitude greater than
// the number of threads used. Integrations stops when the desired relative
// error (estimate) or maximum number of iterations has been reached. A struct
// is returned with the price, as well as some numerical information.
  struct price_2_t
  { double V0, rel_err, lam_p_Q, lam_m_Q;
    size_t n_MC;
  };
  template <size_t n_lam_Q_switch = 1>
  price_2_t price_2
  ( random::Generator *gen_ths, size_t n_ths,
    double *const *S_ths, size_t size_S,
    const auto &option,
    size_t n_MC_max, double rel_err = 0,
    double D_lam_Q = .1,
    size_t n_est_var = 4096
  ) const
  { bool done = false;
    double V0   = 0,
           V0_2 = 0;
    double rel_err_V0 = 0;
    int i_lam_p_Q = 0,
        i_lam_m_Q = 0;
    double lam_p_Q_0 = std::max(1.05, lam_p),
           lam_m_Q_0 = std::max(0.05, lam_m);
    price_2_t ret { .n_MC = n_MC_max/(2*n_est_var)*(2*n_est_var) };
    std::vector<double> last_V0_L(n_est_var),
                        last_V0_R(n_est_var);
    #pragma omp parallel num_threads(n_ths) reduction(+:V0,V0_2)
    { const int i_th = omp_get_thread_num();
      auto &gen = gen_ths[i_th];
      auto *S = S_ths[i_th];
      for (size_t i = 0; i < n_MC_max/(2*n_est_var); i++)
      { const bool update_p = i % (2*n_lam_Q_switch) < n_lam_Q_switch;
        double lam_p_Q_L,
               lam_p_Q_R,
               lam_m_Q_L,
               lam_m_Q_R;
        if (update_p)
        { lam_p_Q_L = lam_p_Q_0+(i_lam_p_Q-1)*D_lam_Q;
          lam_p_Q_R = lam_p_Q_0+(i_lam_p_Q+1)*D_lam_Q;
          lam_m_Q_L = lam_m_Q_R = lam_m_Q_0+i_lam_m_Q*D_lam_Q;
        } else
        { lam_m_Q_L = lam_m_Q_0+(i_lam_m_Q-1)*D_lam_Q;
          lam_m_Q_R = lam_m_Q_0+(i_lam_m_Q+1)*D_lam_Q;
          lam_p_Q_L = lam_p_Q_R = lam_p_Q_0+i_lam_p_Q*D_lam_Q;
        }
        #pragma omp for nowait
        for (size_t j = 0; j < n_est_var; j++)
        { const double RN = gen_path_1(gen, S, size_S, lam_p_Q_L, lam_m_Q_L);
          last_V0_L[j] = RN*option.price(S, size_S);
          V0 += last_V0_L[j];
          V0_2 += pow(last_V0_L[j], 2);
        }
        #pragma omp for
        for (size_t j = 0; j < n_est_var; j++)
        { const double RN = gen_path_1(gen, S, size_S, lam_p_Q_R, lam_m_Q_R);
          last_V0_R[j] = RN*option.price(S, size_S);
          V0 += last_V0_R[j];
          V0_2 += pow(last_V0_R[j], 2);
        }
        #pragma omp single
        { int &i_lam_Q = update_p ? i_lam_p_Q : i_lam_m_Q;
          double var_V0_L = math::var(last_V0_L.data(), n_est_var),
                 var_V0_R = math::var(last_V0_R.data(), n_est_var);
          if (var_V0_L == 0) var_V0_L = DBL_MAX;
          if (var_V0_R == 0) var_V0_R = DBL_MAX;
          if (var_V0_L == var_V0_R)
            i_lam_Q += gen.uniform() < .5 ? -1 : +1;
          else
          { if (var_V0_L < var_V0_R)
            { if (( update_p && lam_p_Q_0+(i_lam_p_Q-2)*D_lam_Q >= 1.05) ||
                  (!update_p && lam_m_Q_0+(i_lam_m_Q-2)*D_lam_Q >= .05))
                i_lam_Q -= 1;
            }
            else
              i_lam_Q += 1;
            ret.n_MC = (i+1)*2*n_est_var;
            const double var_V0 = V0_2/ret.n_MC-pow(V0/ret.n_MC, 2);
            rel_err_V0 = V0 == 0 ? DBL_MAX : sqrt(var_V0*ret.n_MC)/V0;
            done = rel_err_V0 < rel_err;
          }
        }
        if (done) break;
      }
    }
    ret.V0 = V0*discount_factor()/ret.n_MC;
    ret.rel_err = rel_err_V0;
    ret.lam_p_Q = lam_p_Q_0+i_lam_p_Q*D_lam_Q;
    ret.lam_m_Q = lam_m_Q_0+i_lam_m_Q*D_lam_Q;
    return ret;
  }
};

} // models
} // fin1
