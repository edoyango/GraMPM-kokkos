SRCDIR = .
INCLUDEDIR = include

SRCFILENAMES = main.cpp

SRCFILEPATHS = $(addprefix $(SRCDIR)/, $(SRCFILENAMES))

INCLUDE_FILE_NAMES = grampm/accelerated/core.hpp grampm/accelerated/kernels.hpp grampm/accelerated/functors.hpp grampm/accelerated/stressupdate.hpp

INCLUDE_FILE_PATHS = $(addprefix $(INCLUDEDIR)/, $(INCLUDE_FILE_NAMES))

CXXFLAGS := $(CXXFLAGS)

mpm.x: $(SRCFILEPATHS) $(INCLUDE_FILE_PATHS)
	$(CXX) -o $@ $< -IGraMPM/include -Iinclude -I$(INCLUDEDIR) $(CXXFLAGS) -I$(KOKKOS_ROOT)/include -L$(KOKKOS_ROOT)/lib -lkokkoscore

tests: tests/interfaces.x tests/kernels.x tests/grid_mapper.x tests/boundary_conditions.x tests/p2g.x tests/g2p.x tests/stress_update.x tests/updates.x

tests/%.x: tests/%.cpp $(INCLUDE_FILE_PATHS)
	$(CXX) -o $@ $< -IGraMPM/include -Iinclude -I$(INCLUDEDIR) $(CXXFLAGS) -I$(KOKKOS_ROOT)/include -L$(KOKKOS_ROOT)/lib -lkokkoscore -I/usr/local/gtest -I/usr/local/gmock -L/usr/local/lib -lgtest