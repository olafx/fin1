/*
Test pricing of European barrier options under Black-Scholes-Merton with
importance sampling.
*/

#include "../../gen/models.hpp"
#include "../../gen/options.hpp"
#include "../../gen/math.hpp"

#include <vector>
#include <print>

int main
()
{
  constexpr auto model = fin1::models::BSM
  { .S0 = 90,
    .r = .05,
    .q = .02,
    .sig = .2,
    .T = 1
  };
  constexpr auto option = fin1::options::European::barrier::DownOutCall
  { .K = 100,
    .B = 70
  };
  constexpr double rel_err = 0;
  constexpr size_t size_S = 100,
                   n_MC_max = 1e6;

  const int n_ths = omp_get_max_threads();
  std::vector<fin1::random::Generator> gen_ths(n_ths);
  std::vector<std::vector<double>> S_ths(n_ths);
  std::vector<double *> S_ths_C(n_ths);
  for (size_t i = 0; i < n_ths; i++)
  { gen_ths[i] = fin1::random::Generator::deterministic(42+i, 69+i*i);
    gen_ths[i].warmup(10);
      S_ths[i] = std::vector<double>(size_S);
      S_ths_C[i] = S_ths[i].data();
  }

  // double V0 = model.price_1(gen_ths.data(), n_ths, S_ths_C.data(), size_S, option, 100, n_MC_max);
  // const fin1::models::BSM::price_2_t price = model.price_2(gen_ths.data(), n_ths, S_ths_C.data(), size_S, option, n_MC);
  // double V0 = model.price_1(gen_ths.data(), 1, S_ths_C.data(), size_S, option, 100, n_MC_max);
  const fin1::models::BSM::price_2_t price = model.price_2(gen_ths.data(), 1, S_ths_C.data(), size_S, option, n_MC_max, rel_err);
  std::println("V0 {:.2e}", price.V0);
}
