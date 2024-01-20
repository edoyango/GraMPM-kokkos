SRCDIR = src
INCLUDEDIR = include

SRCFILENAMES = main.cpp

SRCFILEPATHS = $(addprefix $(SRCDIR)/, $(SRCFILENAMES))

CXXFLAGS := $(CXXFLAGS)

mpm.x: $(SRCFILEPATHS)
	$(CXX) -o $@ $^ -IGraMPM/include -Iinclude -I$(INCLUDEDIR) $(CXXFLAGS) -I$(KOKKOS_ROOT)/include -L$(KOKKOS_ROOT)/lib -lkokkoscore

tests: tests/interfaces.x tests/kernels.x tests/grid_mapper.x tests/boundary_conditions.x tests/p2g.x tests/g2p.x

tests/%.x: tests/%.cpp include/grampm-kokkos.hpp
	$(CXX) -o $@ $< -IGraMPM/include -Iinclude -I$(INCLUDEDIR) $(CXXFLAGS) -I$(KOKKOS_ROOT)/include -L$(KOKKOS_ROOT)/lib -lkokkoscore -I/usr/local/gtest -I/usr/local/gmock -L/usr/local/lib -lgtest