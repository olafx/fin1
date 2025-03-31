/*
Test pricing of European barrier options under Heston, varying various
parameters, with importance sampling.
*/

#include "../../gen/models.hpp"
#include "../../gen/options.hpp"
#include "../../gen/math.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>
#include <arrow/buffer.h>
#include <omp.h>

#include <memory>
#include <filesystem>
#include <print>

int main
()
{ auto base_model = fin1::models::Heston
  { .S0   = 100,
    .vol0 = .2,
    .r    = .05,
    .q    = .02,
    .eta  = .04,
    .kap  = 1,
    .th   = .1,
    .rho  = -.3,
    .T    = 1
  };
  constexpr auto base_option = fin1::options::European::barrier::DownOutCall
  { .K = 80,
    .B = 70
  };
  constexpr double range_S0  [] {50, 120},
                   range_vol0[] {.1, .5},
                   range_r   [] {-.03, .1},
                   range_q   [] {0, .7},
                   range_eta [] {.01, .25},
                   range_kap [] {.1, 6},
                   range_th  [] {.01, .3},
                   range_rho [] {-.9, +.9},
                   range_T   [] {.1, 3},
                   range_K   [] {60, 120},
                   range_B   [] {50, 110};
  // constexpr double lam = -.5;
  constexpr double rel_err = 0;
  constexpr size_t n_datasets   = 11,
                   size_S       = 100,
                   n_MC_max     = 1e6,
                   size_dataset = 100;
  const std::string filename_base = "test_Heston_EB_1";

  const int strlen_n_datasets   = std::to_string(n_datasets)  .length(),
            strlen_size_dataset = std::to_string(size_dataset).length();
  const auto path_out = std::filesystem::current_path()/"out";
  std::filesystem::create_directories(path_out);

  auto gen_problem = [&](size_t i_dataset, size_t i_elem)
  { auto model  = base_model;
    auto option = base_option;
    switch (i_dataset)
    { case  0: model.S0   = fin1::math::linspace(range_S0,   i_elem, size_dataset); break;
      case  1: model.vol0 = fin1::math::linspace(range_vol0, i_elem, size_dataset); break;
      case  2: model.r    = fin1::math::linspace(range_r,    i_elem, size_dataset); break;
      case  3: model.q    = fin1::math::linspace(range_q,    i_elem, size_dataset); break;
      case  4: model.eta  = fin1::math::linspace(range_eta,  i_elem, size_dataset); break;
      case  5: model.kap  = fin1::math::linspace(range_kap,  i_elem, size_dataset); break;
      case  6: model.th   = fin1::math::linspace(range_th,   i_elem, size_dataset); break;
      case  7: model.rho  = fin1::math::linspace(range_rho,  i_elem, size_dataset); break;
      case  8: model.T    = fin1::math::linspace(range_T,    i_elem, size_dataset); break;
      case  9: option.K   = fin1::math::linspace(range_K,    i_elem, size_dataset); break;
      case 10: option.B   = fin1::math::linspace(range_B,    i_elem, size_dataset); break;
    }
    return std::tuple {model, option};
  };

  auto gen_dataset = [&](size_t i_dataset)
  { const int n_ths = omp_get_max_threads();
    std::vector<fin1::random::Generator> gen_ths(n_ths);
    std::vector<std::vector<double>> S_ths(n_ths),
                                  vol2_ths(n_ths);
    std::vector<double *> S_ths_C(n_ths),
                      vol2_ths_C(n_ths);
    for (size_t i = 0; i < n_ths; i++)
    {  gen_ths[i] = fin1::random::Generator::deterministic(42+i, 69+i*i);
       gen_ths[i].warmup(10);
         S_ths[i] = std::vector<double>(size_S);
      vol2_ths[i] = std::vector<double>(size_S);
         S_ths_C[i] =    S_ths[i].data();
      vol2_ths_C[i] = vol2_ths[i].data();
    }
    std::vector<float> data_S0  (size_dataset), 
                       data_vol0(size_dataset),
                       data_r   (size_dataset),
                       data_q   (size_dataset),
                       data_eta (size_dataset),
                       data_kap (size_dataset),
                       data_th  (size_dataset),
                       data_rho (size_dataset),
                       data_T   (size_dataset),
                       data_K   (size_dataset),
                       data_B   (size_dataset),
                       data_V0  (size_dataset);

    for (size_t i = 0; i < size_dataset; i++)
    { std::print("\r{:>{}}/{} {:>{}}/{}",
        i_dataset+1, strlen_n_datasets, n_datasets,
        i+1, strlen_size_dataset, size_dataset);
      fflush(stdout);
      auto [model, option] = gen_problem(i_dataset, i);
      // const double V0 =
      //   model.price(gen_ths.data(), n_ths, S_ths_C.data(), vol2_ths_C.data(), size_S, option, n_MC_max);
      //   model.price_1(gen_ths.data(), n_ths, S_ths_C.data(), vol2_ths_C.data(), size_S, option, lam, n_MC_max);
      const fin1::models::Heston::price_2_t price =
        model.price_2(gen_ths.data(), n_ths, S_ths_C.data(), vol2_ths_C.data(), size_S, option, n_MC_max, rel_err);
      data_S0  [i] = model.S0;
      data_vol0[i] = model.vol0;
      data_r   [i] = model.r;
      data_q   [i] = model.q;
      data_eta [i] = model.eta;
      data_kap [i] = model.kap;
      data_th  [i] = model.th;
      data_rho [i] = model.rho;
      data_T   [i] = model.T;
      data_K   [i] = option.K;
      data_B   [i] = option.B;
      // data_V0  [i] = V0;
      data_V0  [i] = price.V0;
    }

    const std::vector<std::string> md_k {"model", "option"},
                                   md_v {"Heston", "down-and-out European barrier call"};
    const auto md = std::make_shared<const arrow::KeyValueMetadata>(md_k, md_v);

    const auto schema = arrow::schema(
    { arrow::field("S0"  , arrow::float32()),
      arrow::field("vol0", arrow::float32()),
      arrow::field("r"   , arrow::float32()),
      arrow::field("q"   , arrow::float32()),
      arrow::field("eta" , arrow::float32()),
      arrow::field("kap" , arrow::float32()),
      arrow::field("th"  , arrow::float32()),
      arrow::field("rho" , arrow::float32()),
      arrow::field("T"   , arrow::float32()),
      arrow::field("K"   , arrow::float32()),
      arrow::field("B"   , arrow::float32()),
      arrow::field("V0"  , arrow::float32())
    });

    std::shared_ptr<arrow::io::FileOutputStream> fp;
    std::unique_ptr<parquet::arrow::FileWriter> writer;
    const std::shared_ptr<parquet::WriterProperties> writer_props = parquet::WriterProperties::Builder()
      .compression(parquet::Compression::ZSTD)->compression_level(1)->build();
    PARQUET_ASSIGN_OR_THROW(fp, arrow::io::FileOutputStream::Open(path_out/std::format("{}_{}.parquet", filename_base, i_dataset+1)));
    PARQUET_ASSIGN_OR_THROW(writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), fp, writer_props));
    PARQUET_THROW_NOT_OK(writer->AddKeyValueMetadata(md));

    const std::shared_ptr<arrow::Buffer>
      buf_S0   = arrow::Buffer::Wrap(data_S0  .data(), size_dataset),
      buf_vol0 = arrow::Buffer::Wrap(data_vol0.data(), size_dataset),
      buf_r    = arrow::Buffer::Wrap(data_r   .data(), size_dataset),
      buf_q    = arrow::Buffer::Wrap(data_q   .data(), size_dataset),
      buf_eta  = arrow::Buffer::Wrap(data_eta .data(), size_dataset),
      buf_kap  = arrow::Buffer::Wrap(data_kap .data(), size_dataset),
      buf_th   = arrow::Buffer::Wrap(data_th  .data(), size_dataset),
      buf_rho  = arrow::Buffer::Wrap(data_rho .data(), size_dataset),
      buf_T    = arrow::Buffer::Wrap(data_T   .data(), size_dataset),
      buf_K    = arrow::Buffer::Wrap(data_K   .data(), size_dataset),
      buf_B    = arrow::Buffer::Wrap(data_B   .data(), size_dataset),
      buf_V0   = arrow::Buffer::Wrap(data_V0  .data(), size_dataset);
    const auto arr_S0   = std::make_shared<arrow::FloatArray>(size_dataset, buf_S0  ),
              arr_vol0 = std::make_shared<arrow::FloatArray>(size_dataset, buf_vol0),
              arr_r    = std::make_shared<arrow::FloatArray>(size_dataset, buf_r   ),
              arr_q    = std::make_shared<arrow::FloatArray>(size_dataset, buf_q   ),
              arr_eta  = std::make_shared<arrow::FloatArray>(size_dataset, buf_eta ),
              arr_kap  = std::make_shared<arrow::FloatArray>(size_dataset, buf_kap ),
              arr_th   = std::make_shared<arrow::FloatArray>(size_dataset, buf_th  ),
              arr_rho  = std::make_shared<arrow::FloatArray>(size_dataset, buf_rho ),
              arr_T    = std::make_shared<arrow::FloatArray>(size_dataset, buf_T   ),
              arr_K    = std::make_shared<arrow::FloatArray>(size_dataset, buf_K   ),
              arr_B    = std::make_shared<arrow::FloatArray>(size_dataset, buf_B   ),
              arr_V0   = std::make_shared<arrow::FloatArray>(size_dataset, buf_V0  );
    const auto arrs = arrow::ArrayVector
    { arr_S0,
      arr_vol0,
      arr_r,
      arr_q,
      arr_eta,
      arr_kap,
      arr_th,
      arr_rho,
      arr_T,
      arr_K,
      arr_B,
      arr_V0
    };
    const std::shared_ptr<const arrow::Table> table = arrow::Table::Make(schema, arrs);

    PARQUET_THROW_NOT_OK(writer->WriteTable(*table));
  }; // gen_dataset

  for (size_t i = 0; i < n_datasets; i++)
    gen_dataset(i);
}
