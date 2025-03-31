/*
Test pricing of European options under Heston, varying the initial spot price,
without importance sampling.
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
{ auto model = fin1::models::Heston
  { .vol0 = .2,
    .r    = .05,
    .q    = .02,
    .eta  = .04,
    .kap  = 1,
    .th   = .1,
    .rho  = -.3,
    .T    = 1
  };
  assert(model.Feller_cond());
  constexpr auto option = fin1::options::European::Call
  { .K = 100
  };
  constexpr double range_S0[]
  { 50, 150
  };
  constexpr size_t size_S = 100,
                   n_MC   = 1e6,
                   n_S0   = 100;
  const std::string filename = "test_Heston_E_1";

  const auto path_out = std::filesystem::current_path()/"out";
  std::filesystem::create_directories(path_out);
  const int strlen_n_S0 = std::to_string(n_S0).length();

  const int n_ths = omp_get_max_threads();
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
  std::vector<float> data_S0  (n_S0),
                     data_vol0(n_S0),
                     data_r   (n_S0),
                     data_q   (n_S0),
                     data_eta (n_S0),
                     data_kap (n_S0),
                     data_th  (n_S0),
                     data_rho (n_S0),
                     data_T   (n_S0),
                     data_K   (n_S0),
                     data_V0  (n_S0);

  for (size_t i_S0 = 0; i_S0 < n_S0; i_S0++)
  { std::print("\r{:>{}}/{}", i_S0+1, strlen_n_S0, n_S0);
    fflush(stdout);
    model.S0 = fin1::math::linspace(range_S0[0], range_S0[1], i_S0, n_S0);
    const double V0 = model.price(gen_ths.data(), n_ths, S_ths_C.data(), vol2_ths_C.data(), size_S, option, n_MC);
    data_S0  [i_S0] = model.S0;
    data_vol0[i_S0] = model.vol0;
    data_r   [i_S0] = model.r;
    data_q   [i_S0] = model.q;
    data_eta [i_S0] = model.eta;
    data_kap [i_S0] = model.kap;
    data_th  [i_S0] = model.th;
    data_rho [i_S0] = model.rho;
    data_T   [i_S0] = model.T;
    data_K   [i_S0] = option.K;
    data_V0  [i_S0] = V0;
  }

  const std::vector<std::string> md_k {"model", "option"},
                                 md_v {"Heston", "European call"};
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
    arrow::field("V0"  , arrow::float32())
  });

  std::shared_ptr<arrow::io::FileOutputStream> fp;
  std::unique_ptr<parquet::arrow::FileWriter> writer;
  const std::shared_ptr<parquet::WriterProperties> writer_props = parquet::WriterProperties::Builder()
    .compression(parquet::Compression::ZSTD)->compression_level(1)->build();
  PARQUET_ASSIGN_OR_THROW(fp, arrow::io::FileOutputStream::Open(path_out/std::format("{}.parquet", filename)));
  PARQUET_ASSIGN_OR_THROW(writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), fp, writer_props));
  PARQUET_THROW_NOT_OK(writer->AddKeyValueMetadata(md));

  const std::shared_ptr<arrow::Buffer>
    buf_S0   = arrow::Buffer::Wrap(data_S0  .data(), n_S0),
    buf_vol0 = arrow::Buffer::Wrap(data_vol0.data(), n_S0),
    buf_r    = arrow::Buffer::Wrap(data_r   .data(), n_S0),
    buf_q    = arrow::Buffer::Wrap(data_q   .data(), n_S0),
    buf_eta  = arrow::Buffer::Wrap(data_eta .data(), n_S0),
    buf_kap  = arrow::Buffer::Wrap(data_kap .data(), n_S0),
    buf_th   = arrow::Buffer::Wrap(data_th  .data(), n_S0),
    buf_rho  = arrow::Buffer::Wrap(data_rho .data(), n_S0),
    buf_T    = arrow::Buffer::Wrap(data_T   .data(), n_S0),
    buf_K    = arrow::Buffer::Wrap(data_K   .data(), n_S0),
    buf_V0   = arrow::Buffer::Wrap(data_V0  .data(), n_S0);
  const auto arr_S0   = std::make_shared<arrow::FloatArray>(n_S0, buf_S0  ),
             arr_vol0 = std::make_shared<arrow::FloatArray>(n_S0, buf_vol0),
             arr_r    = std::make_shared<arrow::FloatArray>(n_S0, buf_r   ),
             arr_q    = std::make_shared<arrow::FloatArray>(n_S0, buf_q   ),
             arr_eta  = std::make_shared<arrow::FloatArray>(n_S0, buf_eta ),
             arr_kap  = std::make_shared<arrow::FloatArray>(n_S0, buf_kap ),
             arr_th   = std::make_shared<arrow::FloatArray>(n_S0, buf_th  ),
             arr_rho  = std::make_shared<arrow::FloatArray>(n_S0, buf_rho ),
             arr_T    = std::make_shared<arrow::FloatArray>(n_S0, buf_T   ),
             arr_K    = std::make_shared<arrow::FloatArray>(n_S0, buf_K   ),
             arr_V0   = std::make_shared<arrow::FloatArray>(n_S0, buf_V0  );
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
    arr_V0
  };
  const std::shared_ptr<const arrow::Table> table = arrow::Table::Make(schema, arrs);

  PARQUET_THROW_NOT_OK(writer->WriteTable(*table));
}
