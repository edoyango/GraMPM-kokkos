#ifndef GRAMPM_KOKKOS_MPI
#define GRAMPM_KOKKOS_MPI

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <limits>
#include <iostream>
#include <grampm/extra.hpp>
#include <vector>

static int choose_cut_axis(const Kokkos::View<const int***> &pincell_in, const box<int> &proc_box, 
    const box<int> &node_box) {
    /* 0: cut plane orthogonal to x
       1: "                     " y
       2: "                     " z
    */

    box<int> non_zero_box;
    // checking if this process should participate
    if (no_overlap(proc_box, node_box)) {

        for (int d = 0; d < 3; ++d) {
            non_zero_box.min[d] = node_box.max[d];
            non_zero_box.max[d] = node_box.min[d];
        }
        
    } else {

        box<int> overlap_box {translate_origin(
           find_overlapping_box(node_box, proc_box), proc_box.min)
        };

        const Kokkos::MDRangePolicy<Kokkos::Rank<2>> 
            policy0({overlap_box.min[1], overlap_box.min[2]}, {overlap_box.max[1], overlap_box.max[2]}),
            policy1({overlap_box.min[0], overlap_box.min[2]}, {overlap_box.max[0], overlap_box.max[2]}),
            policy2({overlap_box.min[0], overlap_box.min[1]}, {overlap_box.max[0], overlap_box.max[1]});
        
        // finding x min
        int i = overlap_box.min[0];
        int sum = 0;
        while (sum == 0 && i < overlap_box.max[0]) {
            Kokkos::parallel_reduce("sum in yz layer", policy0, KOKKOS_LAMBDA (const int j, const int k, int &lsum) {
                lsum += pincell_in(i, j, k);
            }, sum);
            i++;
        }
        non_zero_box.min[0] = proc_box.min[0] + i - 1;

        // finding y min
        int j = overlap_box.min[1];
        sum = 0;
        while (sum == 0 && j < overlap_box.max[1]) {
            Kokkos::parallel_reduce("sum in xz layer", policy1, KOKKOS_LAMBDA (const int i, const int k, int &lsum) {
                lsum += pincell_in(i, j, k);
            }, sum);
            j++;
        }
        non_zero_box.min[1] = proc_box.min[1] + j - 1;

        // finding z min
        int k = overlap_box.min[2];
        sum = 0;
        while (sum == 0 && k < overlap_box.max[2]) {
            Kokkos::parallel_reduce("sum in xy layer", policy2, KOKKOS_LAMBDA (const int i, const int j, int &lsum) {
                lsum += pincell_in(i, j, k);
            }, sum);
            k++;
        }
        non_zero_box.min[2] = proc_box.min[2] + k - 1;

        // finding x min
        i = overlap_box.max[0];
        sum = 0;
        while (sum == 0 && i > overlap_box.min[0]) {
            i--;
            Kokkos::parallel_reduce("sum in yz layer", policy0, KOKKOS_LAMBDA (const int j, const int k, int &lsum) {
                lsum += pincell_in(i, j, k);
            }, sum);
        }
        non_zero_box.max[0] = proc_box.min[0] + i + 1;

        // finding y min
        j = overlap_box.max[1];
        sum = 0;
        while (sum == 0 && j > overlap_box.min[1]) {
            j--;
            Kokkos::parallel_reduce("sum in xz layer", policy1, KOKKOS_LAMBDA (const int i, const int k, int &lsum) {
                lsum += pincell_in(i, j, k);
            }, sum);
        }
        non_zero_box.max[1] = proc_box.min[1] + j + 1;

        // finding z min
        k = overlap_box.max[2];
        sum = 0;
        while (sum == 0 && k > overlap_box.min[2]) {
            k--;
            Kokkos::parallel_reduce("sum in xy layer", policy2, KOKKOS_LAMBDA (const int i, const int j, int &lsum) {
                lsum += pincell_in(i, j, k);
            }, sum);
        }
        non_zero_box.max[2] = proc_box.min[2] + k + 1;

    }

    MPI_Request req[2];
    MPI_Status status[2];
    MPI_Iallreduce(MPI_IN_PLACE, &non_zero_box.min, 3, MPI_INTEGER, MPI_MIN, MPI_COMM_WORLD, &req[0]);
    MPI_Iallreduce(MPI_IN_PLACE, &non_zero_box.max, 3, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, &req[1]);

    MPI_Waitall(2, req, status);

    int L[3] {range(non_zero_box, 0), range(non_zero_box, 1), range(non_zero_box, 2)};

    int cax = 0;
    if (L[1] > L[0] && L[1] >= L[2]) cax = 1;
    if (L[2] > L[0] && L[2] > L[1]) cax = 2;

    return cax;

}

struct ORB_tree_node {
    int node_id, proc_range[2], n;
    box<int> extents;
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
    ORB_populate_pincell_func(const Kokkos::View<int*> grid_idx_, const Kokkos::View<int***> p_, 
        const int global_ngrid_[3], const int minidx_[3])
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

struct ORB_sum_layers_by {
    const int cax;
    const Kokkos::View<const int***> p;
    const Kokkos::View<int*> gridsums;
    const int minidx[3];
    const box<int> node_box;
    ORB_sum_layers_by(const int cax_, Kokkos::View<const int***> p_, Kokkos::View<int*> gridsums_, 
        const int minidx_[3], const box<int> &node_box_)
        : cax {cax_}
        , p {p_}
        , gridsums {gridsums_}
        , minidx {minidx_[0], minidx_[1], minidx_[2]}
        , node_box {node_box_}
    {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int id) const {
        if (cax == 0) {
            const int i = id;
            const int globali = minidx[0] + i;
            for (int j = 0; j < p.extent(1); ++j) {
                for (int k = 0; k < p.extent(2); ++k) {
                    int globalj = minidx[1] + j;
                    int globalk = minidx[2] + k;
                    if (contains_point(node_box, globali, globalj, globalk)) {
                        const int nodei = globali - node_box.min[0];
                        gridsums(nodei) += p(i, j, k);
                    }
                }
            }
        } else if (cax == 1) {
            const int j = id;
            const int globalj = minidx[1] + j;
            for (int i = 0; i < p.extent(0); ++i) {
                for (int k = 0; k < p.extent(2); ++k) {
                    int globali = minidx[0] + i;
                    int globalk = minidx[2] + k;
                    if (contains_point(node_box, globali, globalj, globalk)) {
                        const int nodej = globalj - node_box.min[1];
                        gridsums(nodej) += p(i, j, k);
                    }
                }
            }
        } else if (cax == 2) {
            const int k = id;
            const int globalk = minidx[2] + k;
            for (int i = 0; i < p.extent(0); ++i) {
                for (int j = 0; j < p.extent(1); ++j) {
                    int globali = minidx[0] + i;
                    int globalj = minidx[1] + j;
                    if (contains_point(node_box, globali, globalj, globalk)) {
                        const int nodek = globalk - node_box.min[2];
                        gridsums(nodek) += p(i, j, k);
                    }
                }
            }
        }
    }
};

typedef Kokkos::MinLoc<int, int>::value_type minloc_type;

template<typename F>
static void ORB(const int procid, const ORB_tree_node &node_in, const box<int> proc_box, 
    const typename Kokkos::View<box<int>*>::HostMirror &ORB_extents, const box<F> &global_extents, 
    const F cell_size, const Kokkos::View<const int***> &pincell_in) {

    /*                  node_in
                       /       \
                      /         \
                     /           \
              node_lo             node_hi         */

    // find the 
    const int cax = choose_cut_axis(pincell_in, proc_box, node_in.extents);
    Kokkos::View<int*> gridsums;
    gridsums = Kokkos::View<int*>("Grid sums by layer", range(node_in.extents, cax));
    typename Kokkos::View<int*>::HostMirror h_gridsums(create_mirror_view(gridsums));

    Kokkos::deep_copy(gridsums, 0);

    Kokkos::parallel_for("sum layers by "+std::to_string(cax), 
        pincell_in.extent(cax), 
        ORB_sum_layers_by(cax, pincell_in, gridsums, proc_box.min, node_in.extents)
    );

    Kokkos::fence();

    MPI_Allreduce(MPI_IN_PLACE, gridsums.data(), int(gridsums.size()), MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);

    Kokkos::parallel_scan("scan1", gridsums.size(), KOKKOS_LAMBDA(const int i, int& partial_sum, bool is_final) {
        const int val_i = gridsums(i);
        if (is_final) gridsums(i) = partial_sum;
        partial_sum += val_i;
    });

    // pass default execution space for potential async
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_gridsums, gridsums);

    const int numprocs_in = node_in.proc_range[1]-node_in.proc_range[0];
    const F target_ratio = std::ceil(F(numprocs_in)*0.5)/F(numprocs_in);
    const int target_np_lower = target_ratio*node_in.n;

    minloc_type minloc;
    Kokkos::parallel_reduce("find cut location", gridsums.size(), KOKKOS_LAMBDA (const int &i, minloc_type &lminloc) {
        const int diffi = Kokkos::abs(gridsums(i)-target_np_lower);
        if (diffi < lminloc.val) {
            lminloc.val = diffi;
            lminloc.loc = i;
        }
    }, Kokkos::MinLoc<int, int>(minloc));

    Kokkos::fence();

    const int loc = minloc.loc, np_lower = h_gridsums(minloc.loc);

    // determine next step in the tree
    ORB_tree_node node_lo {node_in}, node_hi {node_in};
    node_lo.proc_range[1] = node_in.proc_range[0] + int(std::ceil(F(numprocs_in)*0.5));
    node_hi.proc_range[0] = node_lo.proc_range[1];
    node_lo.extents.max[cax] = loc;
    node_hi.extents.min[cax] = loc;

    node_lo.n = np_lower;
    node_hi.n = node_in.n-np_lower;
    
    node_lo.node_id = node_in.node_id*2;
    node_hi.node_id = node_in.node_id*2+1;

    // travel to lower node
    if (node_lo.proc_range[1]-node_lo.proc_range[0] == 1) {
        ORB_extents(node_lo.proc_range[0]) = node_lo.extents;
    } else {
        ORB(procid, node_lo, proc_box, ORB_extents, global_extents, cell_size, pincell_in);
    }

    // travel to high node
    if (node_hi.proc_range[1]-node_hi.proc_range[0] == 1) {
        ORB_extents(node_hi.proc_range[0]) = node_hi.extents;
    } else {
        ORB(procid, node_hi, proc_box, ORB_extents, global_extents, cell_size, pincell_in);
    }
}

struct locate_neighbours_func {
    const Kokkos::View<const box<int>*> all_extents;
    const Kokkos::View<int*> neighbours_location;
    const int buffer, procid;
    const box<int> my_extents;
    locate_neighbours_func(Kokkos::View<const box<int>*> all_extents_, Kokkos::View<int*> neighbours_location_, 
        int buffer_, int procid_)
        : all_extents {all_extents_}
        , neighbours_location {neighbours_location_}
        , buffer {buffer_}
        , procid {procid_}
        , my_extents {all_extents(procid)}
    {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int other_procid, int &partial_sum, const bool is_final) const {
        bool is_neighbour = !(
            my_extents.min[0] > all_extents(other_procid).max[0] + 2 ||
            my_extents.min[1] > all_extents(other_procid).max[1] + 2 ||
            my_extents.min[2] > all_extents(other_procid).max[2] + 2 ||
            my_extents.max[0] < all_extents(other_procid).min[0] - 2 ||
            my_extents.max[1] < all_extents(other_procid).min[1] - 2 ||
            my_extents.max[2] < all_extents(other_procid).min[2] - 2
        ) && procid != other_procid;
        if (is_final) neighbours_location(other_procid) = partial_sum;
        partial_sum += is_neighbour;
    }
};

struct pack_neighbour_list_func {
    const Kokkos::View<const int*> neighbours_location;
    const Kokkos::View<int*> neighbours;
    const int n_neighbours;
    pack_neighbour_list_func(const Kokkos::View<int*> neighbours_location_, const Kokkos::View<int*> neighbours_, const int n_)
        : neighbours_location {neighbours_location_}
        , neighbours {neighbours_}
        , n_neighbours {n_}
    {}
    void operator()(const int i) const {
        if (i < n_neighbours-1) { // not the last element
            if (neighbours_location(i) != neighbours_location(i+1)) 
                neighbours(neighbours_location(i)) = i;
        } else {
            if (neighbours_location(i) < n_neighbours) neighbours(neighbours_location(i)) = i;
        }
    }
};

static void ORB_find_neighbours(const Kokkos::View<box<int>*> &boundaries, const int procid, const int numprocs, const int buffer, const Kokkos::View<int*> neighbours, int &n_neighbours) {
    Kokkos::View<int*> neighbours_location("View storing neighbour status", numprocs);
    Kokkos::parallel_scan(
        "Flag processes as neighbour", 
        numprocs, 
        locate_neighbours_func(
            boundaries, 
            neighbours_location, 
            buffer, 
            procid
        ),
        n_neighbours
    );

    Kokkos::parallel_for(
        "Pack neighbour processes into list",
        numprocs,
        pack_neighbour_list_func(
            neighbours_location,
            neighbours,
            n_neighbours
        )
    );
}

namespace GraMPM {
    namespace accelerated {
        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::ORB_determine_boundaries() {
            
            // initiate sharing of number of points amongst all processes
            MPI_Request req;
            MPI_Iallreduce(&m_p_size, &m_p_size_global, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, &req);

            // find local minima/maxima of grid extents
            box<int> proc_box;
            Kokkos::parallel_reduce(
                "reduce",
                m_p_size,
                ORB_find_local_coverage_func(d_p_grid_idx, m_ngrid.data()),
                Kokkos::Min<int>(proc_box.min[0]),
                Kokkos::Min<int>(proc_box.min[1]),
                Kokkos::Min<int>(proc_box.min[2]),
                Kokkos::Max<int>(proc_box.max[0]),
                Kokkos::Max<int>(proc_box.max[1]),
                Kokkos::Max<int>(proc_box.max[2])
            );

            // adjust upper limit so it is an open interval
            for (int d = 0; d < 3; ++d) proc_box.max[d]++;

            // declare pincell array and then zero
            Kokkos::View<int***> pincell("particles in local grid", range(proc_box, 0), range(proc_box, 1), 
                range(proc_box, 2));
            Kokkos::deep_copy(pincell, 0);

            Kokkos::parallel_for("populate pincell", 
                m_p_size, 
                ORB_populate_pincell_func(d_p_grid_idx, pincell, m_ngrid.data(), proc_box.min)
            );

            // wait for Iallreduce to finish
            MPI_Status stat;
            MPI_Wait(&req, &stat);

            // define first ORB tree node info
            ORB_tree_node node1;
            node1.proc_range[0] = 0;
            node1.proc_range[1] = numprocs;
            for (int d = 0; d < 3; ++d) {node1.extents.min[d] = 0;}
            for (int d = 0; d < 3; ++d) {node1.extents.max[d] = m_ngrid[d];}
            node1.n = m_p_size_global;
            node1.node_id = 1;

            // start ORB
            ORB<F>(procid, node1, proc_box, h_ORB_extents, m_g_extents, m_g_cell_size, pincell);

            Kokkos::deep_copy(d_ORB_extents, h_ORB_extents);

            ORB_find_neighbours(d_ORB_extents, procid, numprocs, int(knl.radius), d_ORB_neighbours, n_ORB_neighbours);

        }
    }
}


#endif