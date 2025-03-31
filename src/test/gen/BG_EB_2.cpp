#include "../../gen/models.hpp"
#include "../../gen/options.hpp"
#include "../../gen/math.hpp"

#include <vector>
#include <print>

int main
()
{ auto model = fin1::models::BG
  { .S0    = 100,
    .r     = .05,
    .q     = .02,
    .al_p  = 1.18,
    .al_m  = 1.44,
    .lam_p = 10.57,
    .lam_m = 5.57,
    .T     = 1
  };
  constexpr auto option = fin1::options::European::barrier::DownOutCall
  { .K = 80,
    .B = 70
  };
  constexpr double rel_err  = 0;
  constexpr size_t size_S   = 100,
                   n_MC_max = 1e7;

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

  const fin1::models::BG::price_2_t price = model.price_2(gen_ths.data(), n_ths, S_ths_C.data(), size_S, option, n_MC_max, rel_err);
  std::println("V0 {:.2e}", price.V0);
}
