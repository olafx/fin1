CXX = /opt/homebrew/Cellar/llvm/19.1.6/bin/clang++
CXXFLAGS = -std=c++23 -O3 -ffast-math -fopenmp

BUILD_DIR = out

ARROW_INC = /opt/homebrew/Cellar/apache-arrow/19.0.0_1/include
ARROW_LIB = /opt/homebrew/Cellar/apache-arrow/19.0.0_1/lib
ARROW_LIBS = -larrow -lparquet -larrow_dataset -larrow_acero -larrow_flight -lgandiva
BLAS_LAPACK_INC = /opt/homebrew/Cellar/openblas/0.3.28/include
BLAS_LAPACK_LIB = /opt/homebrew/Cellar/openblas/0.3.28/lib
BLAS_LAPACK_LIBS = -lblas -llapack # -lopenblas
GSL_INC = /opt/homebrew/Cellar/gsl/2.8/include
GSL_LIB = /opt/homebrew/Cellar/gsl/2.8/lib
GSL_LIBS = -lgsl -lgslcblas
OMP_INC = /opt/homebrew/Cellar/libomp/19.1.3/include
OMP_LIB = /opt/homebrew/Cellar/libomp/19.1.3/lib
OMP_LIBS = -lomp

BINS = test_parquet_write test_BSM_E_1 test_BSM_EB_1 test_BSM_EB_2 test_Heston_E_1 test_Heston_EB_1 test_BG_E_1 test_BG_EB_1 test_BG_EB_2 gen_BSM_E

# Make sure the output directory exists
$(shell mkdir -p $(BUILD_DIR))

.PHONY: all clean $(BINS)

all: $(BINS)

clean:
	rm -f $(BUILD_DIR)/*

test_parquet_write: src/test/parquet/write.cpp
	$(CXX) $(CXXFLAGS) -I$(ARROW_INC) -o $(BUILD_DIR)/$@ $< -L$(ARROW_LIB) $(ARROW_LIBS)
test_BSM_E_1: src/test/gen/BSM_E_1.cpp
	$(CXX) $(CXXFLAGS) -I$(OMP_INC) -I$(ARROW_INC) -o $(BUILD_DIR)/$@ $< -L$(ARROW_LIB) $(ARROW_LIBS)
test_BSM_EB_1: src/test/gen/BSM_EB_1.cpp
	$(CXX) $(CXXFLAGS) -I$(OMP_INC) -I$(ARROW_INC) -o $(BUILD_DIR)/$@ $< -L$(ARROW_LIB) $(ARROW_LIBS)
test_BSM_EB_2: src/test/gen/BSM_EB_2.cpp
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/$@ $<
test_Heston_E_1: src/test/gen/Heston_E_1.cpp
	$(CXX) $(CXXFLAGS) -I$(OMP_INC) -I$(ARROW_INC) -o $(BUILD_DIR)/$@ $< -L$(ARROW_LIB) $(ARROW_LIBS)
test_Heston_EB_1: src/test/gen/Heston_EB_1.cpp
	$(CXX) $(CXXFLAGS) -I$(OMP_INC) -I$(ARROW_INC) -o $(BUILD_DIR)/$@ $< -L$(ARROW_LIB) $(ARROW_LIBS)
test_BG_E_1: src/test/gen/BG_E_1.cpp
	$(CXX) $(CXXFLAGS) -I$(OMP_INC) -I$(ARROW_INC) -o $(BUILD_DIR)/$@ $< -L$(ARROW_LIB) $(ARROW_LIBS)
test_BG_EB_1: src/test/gen/BG_EB_1.cpp
	$(CXX) $(CXXFLAGS) -I$(OMP_INC) -I$(ARROW_INC) -o $(BUILD_DIR)/$@ $< -L$(ARROW_LIB) $(ARROW_LIBS)
test_BG_EB_2: src/test/gen/BG_EB_2.cpp
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/$@ $<
gen_BSM_E: src/gen/BSM_E.cpp
	$(CXX) $(CXXFLAGS) -I$(OMP_INC) -I$(ARROW_INC) -o $(BUILD_DIR)/$@ $< -L$(ARROW_LIB) $(ARROW_LIBS)
