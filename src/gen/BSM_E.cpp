/*
Generate random European option data under Black-Scholes-Merton.
*/

#include "models.hpp"
#include "options.hpp"
#include "math.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>
#include <arrow/buffer.h>
#include <omp.h>

#include <memory>
#include <filesystem>
#include <regex>
#include <print>

int main
()
{ namespace fs = std::filesystem;
  using fin1::math::linspace;

// All prices are divided by strike K for undimensionalization.
  constexpr double range_S0_o_K [] {.2, 5},
                   range_r      [] {-.03, .1},
                   range_q      [] {0., .07},
                   range_sig    [] {1e-3, 1},
                   range_T      [] {1e-6, 5};

  constexpr double D_lam     = .05,
                   rel_err   = 1e-2;
  constexpr size_t n_est_var = 1<<12;
  constexpr size_t size_S    = 1e2,
                   n_MC_max  = 1e8,
                   n_samples = 1e4;

  const int strlen_n_samples = std::to_string(n_samples).length();

  const auto path_out = fs::current_path()/"out";
  const std::string filename_base = "BSM_E";

  fs::create_directories(path_out);
  std::regex pattern_filepath {filename_base+R"(_(\d+)\.parquet)"};
  size_t i_file_next = 0;
  for (const auto &entry : fs::directory_iterator {path_out})
  { if (entry.is_regular_file())
    { std::smatch match;
      auto filename = entry.path().filename().string();
      if (std::regex_match(filename, match, pattern_filepath))
      { auto i_file = static_cast<size_t>(std::stoul(match[1]));
        if (i_file > i_file_next)
          i_file_next = i_file;
      }
    }
  }

  const int n_ths = omp_get_max_threads();
  std::vector<fin1::random::Generator> gen_ths(n_ths);
  auto &gen = gen_ths[0];
  std::vector<std::vector<double>> S_ths(n_ths);
  std::vector<double *> S_ths_C(n_ths);
  for (size_t i = 0; i < n_ths; i++)
  { gen_ths[i] = fin1::random::Generator::nondeterministic();
      S_ths[i] = std::vector<double>(size_S);
      S_ths_C[i] = S_ths[i].data();
  }
  std::vector<float> data_S0_o_K (n_samples),
                     data_r      (n_samples),
                     data_q      (n_samples),
                     data_sig    (n_samples),
                     data_T      (n_samples),
// Moneyness V0/K.
                     data_M      (n_samples),
                     data_rel_err(n_samples);

  for (size_t i = 0; i < n_samples; i++)
  { std::print("\r{:>{}}/{}", i+1, strlen_n_samples, n_samples);
    fflush(stdout);
    fin1::models::BSM model
    { .S0  = linspace(range_S0_o_K, gen.uniform()),
      .r   = linspace(range_r,      gen.uniform()),
      .q   = linspace(range_q,      gen.uniform()),
      .sig = linspace(range_sig,    gen.uniform()),
      .T   = linspace(range_T,      gen.uniform())
    };
    fin1::options::European::Call option
    { .K = 1
    };
    const fin1::models::BSM::price_2_t price =
      model.price_2(gen_ths.data(), n_ths, S_ths_C.data(), size_S, option, n_MC_max, rel_err, D_lam, n_est_var);
    data_S0_o_K [i] = model.S0;
    data_r      [i] = model.r;
    data_q      [i] = model.q;
    data_sig    [i] = model.sig;
    data_T      [i] = model.T;
    data_M      [i] = price.V0;
    data_rel_err[i] = price.rel_err;
  }
  std::println();

  const std::vector<std::string> md_k {"model", "option"},
                                 md_v {"Black-Scholes-Merton", "European call"};
  const auto md = std::make_shared<const arrow::KeyValueMetadata>(md_k, md_v);

  const std::shared_ptr<arrow::Schema> schema = arrow::schema
  ({arrow::field("S0/K"   , arrow::float32()),
    arrow::field("r"      , arrow::float32()),
    arrow::field("q"      , arrow::float32()),
    arrow::field("sig"    , arrow::float32()),
    arrow::field("T"      , arrow::float32()),
    arrow::field("M"      , arrow::float32()),
    arrow::field("rel err", arrow::float32())
  });

  std::shared_ptr<arrow::io::FileOutputStream> fp;
  std::unique_ptr<parquet::arrow::FileWriter> writer;
  const std::shared_ptr<parquet::WriterProperties> writer_props = parquet::WriterProperties::Builder()
    .compression(parquet::Compression::ZSTD)->compression_level(1)->build();
  PARQUET_ASSIGN_OR_THROW(fp, arrow::io::FileOutputStream::Open(path_out/std::format("{}_{}.parquet", filename_base, i_file_next+1)));
  PARQUET_ASSIGN_OR_THROW(writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), fp, writer_props));
  PARQUET_THROW_NOT_OK(writer->AddKeyValueMetadata(md));

  const std::shared_ptr<arrow::Buffer>
    buf_S0_o_K  = arrow::Buffer::Wrap(data_S0_o_K .data(), n_samples),
    buf_r       = arrow::Buffer::Wrap(data_r      .data(), n_samples),
    buf_q       = arrow::Buffer::Wrap(data_q      .data(), n_samples),
    buf_sig     = arrow::Buffer::Wrap(data_sig    .data(), n_samples),
    buf_T       = arrow::Buffer::Wrap(data_T      .data(), n_samples),
    buf_M       = arrow::Buffer::Wrap(data_M      .data(), n_samples),
    buf_rel_err = arrow::Buffer::Wrap(data_rel_err.data(), n_samples);
  const auto arr_S0_o_K  = std::make_shared<arrow::FloatArray>(n_samples, buf_S0_o_K ),
             arr_r       = std::make_shared<arrow::FloatArray>(n_samples, buf_r      ),
             arr_q       = std::make_shared<arrow::FloatArray>(n_samples, buf_q      ),
             arr_sig     = std::make_shared<arrow::FloatArray>(n_samples, buf_sig    ),
             arr_T       = std::make_shared<arrow::FloatArray>(n_samples, buf_T      ),
             arr_M       = std::make_shared<arrow::FloatArray>(n_samples, buf_M      ),
             arr_rel_err = std::make_shared<arrow::FloatArray>(n_samples, buf_rel_err);
  const auto arrs = arrow::ArrayVector
  { arr_S0_o_K,
    arr_r,
    arr_q,
    arr_sig,
    arr_T,
    arr_M,
    arr_rel_err
  };
  const std::shared_ptr<const arrow::Table> table = arrow::Table::Make(schema, arrs);

  PARQUET_THROW_NOT_OK(writer->WriteTable(*table));
}
