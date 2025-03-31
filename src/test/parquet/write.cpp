#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <arrow/buffer.h>
#include <vector>
#include <memory>
#include <iostream>

int main()
{
  constexpr int n = 1000, n_chunks = 2, chunk_size = n/n_chunks;

  std::vector<float> a_data(n);
  std::vector<uint64_t> b_data(n);
  for (size_t i = 0; i < n; i++)
  { a_data[i] = static_cast<float>(i)/n;
    b_data[i] = i;
  }

  auto schema = arrow::schema(
  { arrow::field("a", arrow::float32()),
    arrow::field("b", arrow::uint64())
  });

  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  PARQUET_ASSIGN_OR_THROW(outfile, arrow::io::FileOutputStream::Open("out/2.parquet"));
  std::unique_ptr<parquet::arrow::FileWriter> writer;
  PARQUET_ASSIGN_OR_THROW(writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), outfile));
  for (size_t i_chunk = 0; i_chunk < n_chunks; i_chunk++)
  { auto a_buffer = arrow::Buffer::Wrap(a_data.data()+(i_chunk*chunk_size), chunk_size);
    auto b_buffer = arrow::Buffer::Wrap(b_data.data()+(i_chunk*chunk_size), chunk_size);
    auto a_array = std::make_shared<arrow::FloatArray>(chunk_size, a_buffer);
    auto b_array = std::make_shared<arrow::UInt64Array>(chunk_size, b_buffer);
    std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {a_array, b_array}, chunk_size);
    PARQUET_THROW_NOT_OK(writer->WriteTable(*table, chunk_size));
  }
}
