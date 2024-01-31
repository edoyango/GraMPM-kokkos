#ifndef GRAMPM_KOKKOS_MPI
#define GRAMPM_KOKKOS_MPI

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <limits>
#include <iostream>

static int choose_cut_axis(const Kokkos::View<const int***> &pincell_in, const int minidx[3], const int maxidx[3], 
    const int idx_start[3], const int idx_end[3]) {
    /* 0: cut plane orthogonal to x
       1: "                     " y
       2: "                     " z
    */

    int min_non_zero[3], max_non_zero[3];
    // checking if this process should participate
    if (minidx[0] >= idx_end[0] || 
        minidx[1] >= idx_end[1] || 
        minidx[2] >= idx_end[2] || 
        maxidx[0] <= idx_start[0] ||
        maxidx[1] <= idx_start[1] || 
        maxidx[2] <= idx_start[2]) {
        
        min_non_zero[0] = idx_end[0];
        min_non_zero[1] = idx_end[1];
        min_non_zero[2] = idx_end[2];
        max_non_zero[0] = idx_start[0];
        max_non_zero[1] = idx_start[1];
        max_non_zero[2] = idx_start[2];
        
    } else {

        const int start[3] {
            std::max(idx_start[0], minidx[0]) - minidx[0],
            std::max(idx_start[1], minidx[1]) - minidx[1],
            std::max(idx_start[2], minidx[2]) - minidx[2]
        }, end[3] {
            std::min(idx_end[0], maxidx[0]) - minidx[0],
            std::min(idx_end[1], maxidx[1]) - minidx[1],
            std::min(idx_end[2], maxidx[2]) - minidx[2]
        };

        const Kokkos::MDRangePolicy<Kokkos::Rank<2>> 
            policy0({start[1], start[2]}, {end[1], end[2]}),
            policy1({start[0], start[2]}, {end[0], end[2]}),
            policy2({start[0], start[1]}, {end[0], end[1]});
        

        // finding x min
        int i = start[0];
        int sum = 0;
        while (sum == 0 && i < end[0]) {
            Kokkos::parallel_reduce("sum in yz layer", policy0, KOKKOS_LAMBDA (const int j, const int k, int &lsum) {
                lsum += pincell_in(i, j, k);
            }, sum);
            i++;
        }
        min_non_zero[0] = minidx[0] + i - 1;

        // finding y min
        int j = start[1];
        sum = 0;
        while (sum == 0 && j < end[1]) {
            Kokkos::parallel_reduce("sum in xz layer", policy1, KOKKOS_LAMBDA (const int i, const int k, int &lsum) {
                lsum += pincell_in(i, j, k);
            }, sum);
            j++;
        }
        min_non_zero[1] = minidx[1] + j - 1;

        // finding z min
        int k = start[2];
        sum = 0;
        while (sum == 0 && k < end[2]) {
            Kokkos::parallel_reduce("sum in xy layer", policy2, KOKKOS_LAMBDA (const int i, const int j, int &lsum) {
                lsum += pincell_in(i, j, k);
            }, sum);
            k++;
        }
        min_non_zero[2] = minidx[2] + k - 1;

        // finding x min
        i = end[0];
        sum = 0;
        while (sum == 0 && i > start[0]) {
            i--;
            Kokkos::parallel_reduce("sum in yz layer", policy0, KOKKOS_LAMBDA (const int j, const int k, int &lsum) {
                lsum += pincell_in(i, j, k);
            }, sum);
        }
        max_non_zero[0] = minidx[0] + i + 1;

        // finding y min
        j = end[1];
        sum = 0;
        while (sum == 0 && j > start[1]) {
            j--;
            Kokkos::parallel_reduce("sum in xz layer", policy1, KOKKOS_LAMBDA (const int i, const int k, int &lsum) {
                lsum += pincell_in(i, j, k);
            }, sum);
        }
        max_non_zero[1] = minidx[1] + j + 1;

        // finding z min
        k = end[2];
        sum = 0;
        while (sum == 0 && k > start[2]) {
            k--;
            Kokkos::parallel_reduce("sum in xy layer", policy2, KOKKOS_LAMBDA (const int i, const int j, int &lsum) {
                lsum += pincell_in(i, j, k);
            }, sum);
        }
        max_non_zero[2] = minidx[2] + k + 1;

    }

    MPI_Request req[2];
    MPI_Status status[2];
    MPI_Iallreduce(MPI_IN_PLACE, min_non_zero, 3, MPI_INTEGER, MPI_MIN, MPI_COMM_WORLD, &req[0]);
    MPI_Iallreduce(MPI_IN_PLACE, max_non_zero, 3, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, &req[1]);

    MPI_Waitall(2, req, status);

    int L[3] {max_non_zero[0] - min_non_zero[0], max_non_zero[1] - min_non_zero[1], max_non_zero[2] - min_non_zero[2]};

    int cax = 0;
    if (L[1] > L[0] && L[1] >= L[2]) cax = 1;
    if (L[2] > L[0] && L[2] > L[1]) cax = 2;

    return cax;

}

struct ORB_tree_node {
    int node_id, proc_range[2], n, idx_start[3], idx_end[3];
};

struct ORB_find_local_coverage_func {
    const Kokkos::View<const int*> grid_idx;
    const int global_ngrid[3];
    ORB_find_local_coverage_func(const Kokkos::View<int*> grid_idx_, const int global_ngrid_[3])
        : grid_idx {grid_idx_}
        , global_ngrid {global_ngrid_[0], global_ngrid_[1], global_ngrid_[2]}
    {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, int& lminx, int& lminy, int& lminz, int& lmaxx, int& lmaxy, int& lmaxz) const {
        const int idx = grid_idx(i), 
            idxx = idx/(global_ngrid[1]*global_ngrid[2]), 
            idxy = (idx % (global_ngrid[1]*global_ngrid[2]))/global_ngrid[2], 
            idxz = idx-idxx*global_ngrid[1]*global_ngrid[2]-idxy*global_ngrid[2];
        if (idxx < lminx) lminx = idxx;
        if (idxy < lminy) lminy = idxy;
        if (idxz < lminz) lminz = idxz;
        if (idxx > lmaxx) lmaxx = idxx;
        if (idxy > lmaxy) lmaxy = idxy;
        if (idxz > lmaxz) lmaxz = idxz;
    }
};

struct ORB_populate_pincell_func {
    const Kokkos::View<const int*> grid_idx;
    const Kokkos::View<int***> p;
    const int global_ngrid[3], minidx[3];
    ORB_populate_pincell_func(const Kokkos::View<int*> grid_idx_, const Kokkos::View<int***> p_, const int global_ngrid_[3], const int minidx_[3])
        : grid_idx {grid_idx_}
        , p {p_}
        , global_ngrid {global_ngrid_[0], global_ngrid_[1], global_ngrid_[2]}
        , minidx {minidx_[0], minidx_[1], minidx_[2]}
    {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        const int idx = grid_idx(i), 
            idxx = idx/(global_ngrid[1]*global_ngrid[2]), 
            idxy = (idx % (global_ngrid[1]*global_ngrid[2]))/global_ngrid[2], 
            idxz = idx-idxx*global_ngrid[1]*global_ngrid[2]-idxy*global_ngrid[2];
        Kokkos::atomic_increment(&p(idxx-minidx[0], idxy-minidx[1], idxz-minidx[2]));
    }
};

struct ORB_sum_layers_by_x {
    const Kokkos::View<const int***> p;
    const Kokkos::View<int*> gridsums;
    const int minidx[3], idx_start[3], idx_end[3];
    ORB_sum_layers_by_x(Kokkos::View<const int***> p_, Kokkos::View<int*> gridsums_, const int minidx_[3], const int idx_start_[3], 
        const int idx_end_[3])
        : p {p_}
        , gridsums {gridsums_}
        , minidx {minidx_[0], minidx_[1], minidx_[2]}
        , idx_start {idx_start_[0], idx_start_[1], idx_start_[2]}
        , idx_end {idx_end_[0], idx_end_[1], idx_end_[2]}
    {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        const int globali = minidx[0] + i;
        for (int j = 0; j < p.extent(1); ++j) {
            for (int k = 0; k < p.extent(2); ++k) {
                int globalj = minidx[1] + j;
                int globalk = minidx[2] + k;
                if (globali >= idx_start[0] && globali < idx_end[0] && 
                    globalj >= idx_start[1] && globalj < idx_end[1] &&
                    globalk >= idx_start[2] && globalk < idx_end[2]) {
                    const int nodei = globali - idx_start[0];
                    gridsums(nodei) += p(i, j, k);
                }
            }
        }
    }
};

struct ORB_sum_layers_by_y {
    const Kokkos::View<const int***> p;
    const Kokkos::View<int*> gridsums;
    const int minidx[3], idx_start[3], idx_end[3];
    ORB_sum_layers_by_y(Kokkos::View<const int***> p_, Kokkos::View<int*> gridsums_, const int minidx_[3], const int idx_start_[3], 
        const int idx_end_[3])
        : p {p_}
        , gridsums {gridsums_}
        , minidx {minidx_[0], minidx_[1], minidx_[2]}
        , idx_start {idx_start_[0], idx_start_[1], idx_start_[2]}
        , idx_end {idx_end_[0], idx_end_[1], idx_end_[2]}
    {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int j) const {
        const int globalj = minidx[1] + j;
        for (int i = 0; i < p.extent(0); ++i) {
            for (int k = 0; k < p.extent(2); ++k) {
                int globali = minidx[0] + i;
                int globalk = minidx[2] + k;
                if (globali >= idx_start[0] && globali < idx_end[0] && 
                    globalj >= idx_start[1] && globalj < idx_end[1] &&
                    globalk >= idx_start[2] && globalk < idx_end[2]) {
                    const int nodej = globalj - idx_start[1];
                    gridsums(nodej) += p(i, j, k);
                }
            }
        }
    }
};

struct ORB_sum_layers_by_z {
    const Kokkos::View<const int***> p;
    const Kokkos::View<int*> gridsums;
    const int minidx[3], idx_start[3], idx_end[3];
    ORB_sum_layers_by_z(Kokkos::View<const int***> p_, Kokkos::View<int*> gridsums_, const int minidx_[3], const int idx_start_[3], 
        const int idx_end_[3])
        : p {p_}
        , gridsums {gridsums_}
        , minidx {minidx_[0], minidx_[1], minidx_[2]}
        , idx_start {idx_start_[0], idx_start_[1], idx_start_[2]}
        , idx_end {idx_end_[0], idx_end_[1], idx_end_[2]}
    {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int k) const {
        const int globalk = minidx[2] + k;
        for (int i = 0; i < p.extent(0); ++i) {
            for (int j = 0; j < p.extent(1); ++j) {
                int globali = minidx[0] + i;
                int globalj = minidx[1] + j;
                if (globali >= idx_start[0] && globali < idx_end[0] && 
                    globalj >= idx_start[1] && globalj < idx_end[1] &&
                    globalk >= idx_start[2] && globalk < idx_end[2]) {
                    const int nodek = globalk - idx_start[2];
                    gridsums(nodek) += p(i, j, k);
                }
            }
        }
    }
};

template<typename F>
static void ORB(const int procid, const ORB_tree_node &node_in, const int minidx[3], const int maxidx[3], 
    F ORB_mingrid[3], F ORB_maxgrid[3], const F mingrid_global[3], const F cell_size, const Kokkos::View<const int***> &pincell_in) {

    /*                  node_in
                       /       \
                      /         \
                     /           \
              node_lo             node_hi         */

    // trim down the dims
    const int cax = choose_cut_axis(pincell_in, minidx, maxidx, node_in.idx_start, node_in.idx_end);

    Kokkos::View<int*> gridsums("Grid sums by layer", node_in.idx_end[cax]-node_in.idx_start[cax]);
    typename Kokkos::View<int*>::HostMirror h_gridsums(create_mirror_view(gridsums));

    Kokkos::deep_copy(gridsums, 0);

    const int ngrid_in[3] {
        node_in.idx_end[0]-node_in.idx_start[0],
        node_in.idx_end[1]-node_in.idx_start[1],
        node_in.idx_end[2]-node_in.idx_start[2]
    };

    if (cax == 0) {
        Kokkos::parallel_for("sum layers by x", 
            pincell_in.extent(0), 
            ORB_sum_layers_by_x(pincell_in, gridsums, minidx, node_in.idx_start, node_in.idx_end)
        ); 
    } else if (cax == 1) {
        Kokkos::parallel_for("sum layers by y", 
            pincell_in.extent(1), 
            ORB_sum_layers_by_y(pincell_in, gridsums, minidx, node_in.idx_start, node_in.idx_end)
        ); 

    } else if (cax == 2) {
        Kokkos::parallel_for("sum layers by z", 
            pincell_in.extent(2), 
            ORB_sum_layers_by_y(pincell_in, gridsums, minidx, node_in.idx_start, node_in.idx_end)
        );
    }

    Kokkos::fence();

    MPI_Allreduce(MPI_IN_PLACE, gridsums.data(), int(gridsums.size()), MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);

    Kokkos::parallel_scan("scan1", gridsums.size(), KOKKOS_LAMBDA(const int i, int& partial_sum, bool is_final) {
        const int val_i = gridsums(i);
        if (is_final) gridsums(i) = partial_sum;
        partial_sum += val_i;
    });

    Kokkos::deep_copy(h_gridsums, gridsums);

    const int numprocs_in = node_in.proc_range[1]-node_in.proc_range[0];
    const F target_ratio = std::ceil(F(numprocs_in)*0.5)/F(numprocs_in);
    const int target_np_lower = target_ratio*node_in.n;

    int loc;
    Kokkos::parallel_reduce("find threshold", gridsums.size(), KOKKOS_LAMBDA(const int i, int &loc) {
        loc = (loc < gridsums.size()) ? loc : gridsums.size()-1;
        if (Kokkos::abs(gridsums(i)-target_np_lower) < Kokkos::abs(gridsums(loc)-target_np_lower)) loc = i;
    }, Kokkos::Min<int>(loc));
    int np_lower = h_gridsums(loc);

    // determine next step in the tree
    ORB_tree_node node_lo, node_hi;
    node_lo.proc_range[0] = node_in.proc_range[0];
    node_lo.proc_range[1] = node_in.proc_range[0] + int(std::ceil(F(numprocs_in)*0.5));
    node_hi.proc_range[0] = node_lo.proc_range[1];
    node_hi.proc_range[1] = node_in.proc_range[1];
    for (int d = 0; d < 3; ++d) {
        node_lo.idx_start[d] = node_in.idx_start[d];
        node_lo.idx_end[d] = node_in.idx_end[d];
        node_hi.idx_start[d] = node_in.idx_start[d];
        node_hi.idx_end[d] = node_in.idx_end[d];
    }
    node_lo.idx_end[cax] = loc;
    node_hi.idx_start[cax] = loc;

    node_lo.n = np_lower;
    node_hi.n = node_in.n-np_lower;
    
    node_lo.node_id = node_in.node_id*2;
    node_hi.node_id = node_in.node_id*2+1;

    // travel to lower node
    if (node_lo.proc_range[1]-node_lo.proc_range[0] == 1) {
        if (node_lo.proc_range[0]==procid) {
            ORB_mingrid[0] = mingrid_global[0] + node_lo.idx_start[0]*cell_size;
            ORB_mingrid[1] = mingrid_global[1] + node_lo.idx_start[1]*cell_size;
            ORB_mingrid[2] = mingrid_global[2] + node_lo.idx_start[2]*cell_size;
            ORB_maxgrid[0] = mingrid_global[0] + node_lo.idx_end[0]*cell_size;
            ORB_maxgrid[1] = mingrid_global[1] + node_lo.idx_end[1]*cell_size;
            ORB_maxgrid[2] = mingrid_global[2] + node_lo.idx_end[2]*cell_size;
        }
    } else {
        ORB(procid, node_lo, minidx, maxidx, ORB_mingrid, ORB_maxgrid, mingrid_global, cell_size, pincell_in);
    }

    // travel to high node
    if (node_hi.proc_range[1]-node_hi.proc_range[0] == 1) {
        if (node_hi.proc_range[0]==procid) {
        ORB_mingrid[0] = mingrid_global[0] + node_hi.idx_start[0]*cell_size;
        ORB_mingrid[1] = mingrid_global[1] + node_hi.idx_start[1]*cell_size;
        ORB_mingrid[2] = mingrid_global[2] + node_hi.idx_start[2]*cell_size;
        ORB_maxgrid[0] = mingrid_global[0] + node_hi.idx_end[0]*cell_size;
        ORB_maxgrid[1] = mingrid_global[1] + node_hi.idx_end[1]*cell_size;
        ORB_maxgrid[2] = mingrid_global[2] + node_hi.idx_end[2]*cell_size;
        }
    } else {
        ORB(procid, node_hi, minidx, maxidx, ORB_mingrid, ORB_maxgrid, mingrid_global, cell_size, pincell_in);
    }
}

namespace GraMPM {
    namespace accelerated {
        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::ORB_determine_boundaries() {
            
            MPI_Request req;
            MPI_Iallreduce(&m_p_size, &m_p_size_global, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, &req);

            int minidx[3], maxidx[3];
            Kokkos::parallel_reduce(
                "reduce",
                m_p_size,
                ORB_find_local_coverage_func(d_p_grid_idx, m_ngrid.data()),
                Kokkos::Min<int>(minidx[0]),
                Kokkos::Min<int>(minidx[1]),
                Kokkos::Min<int>(minidx[2]),
                Kokkos::Max<int>(maxidx[0]),
                Kokkos::Max<int>(maxidx[1]),
                Kokkos::Max<int>(maxidx[2])
            );

            maxidx[0]++;
            maxidx[1]++;
            maxidx[2]++;

            const int ngridx_loc[3] {
                maxidx[0]-minidx[0],
                maxidx[1]-minidx[1],
                maxidx[2]-minidx[2]
            };

            ORB_tree_node node1;
            node1.proc_range[0] = 0;
            node1.proc_range[1] = numprocs;
            for (int d = 0; d < 3; ++d) {
                node1.idx_start[d] = 0;
                node1.idx_end[d] = m_ngrid[d];
            }

            // declare pincell array and then zero
            Kokkos::View<int***> pincell("particles in local grid", ngridx_loc[0], ngridx_loc[1], ngridx_loc[2]);
            Kokkos::deep_copy(pincell, 0);

            Kokkos::parallel_for("populate pincell", 
                m_p_size, 
                ORB_populate_pincell_func(d_p_grid_idx, pincell, m_ngrid.data(), minidx)
            );

            MPI_Status stat;
            MPI_Wait(&req, &stat);
            node1.n = m_p_size_global;
            node1.node_id = 1;
            ORB(procid, node1, minidx, maxidx, m_ORB_mingrid.data(), m_ORB_maxgrid.data(), m_mingrid.data(), m_g_cell_size, pincell);

        }
    }
}


#endif