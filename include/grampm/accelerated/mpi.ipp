#ifndef GRAMPM_KOKKOS_MPI
#define GRAMPM_KOKKOS_MPI

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <limits>
#include <iostream>
#include <grampm/extra.hpp>
#include <vector>

static int find_first_nonzero_layer(const Kokkos::View<const int***> &pincell_in, const int start, const int end, const box<int> &overlap_box, const int ax) { 
    int idx = start, sum = 0;
    const int offax[2] {(ax+1)%3, (ax+2)%3};
    while (sum == 0 && idx < end) {
        Kokkos::parallel_reduce(
            "sum layer", 
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {overlap_box.min[offax[0]], overlap_box.min[offax[1]]},
                {overlap_box.max[offax[0]], overlap_box.max[offax[1]]}
            ), 
            KOKKOS_LAMBDA (const int i, const int j, int &lsum) {
                if (ax==0) {
                    lsum += pincell_in(idx, i, j);
                } else if (ax==1) {
                    lsum += pincell_in(j, idx, i);
                } else if (ax==2) {
                    lsum += pincell_in(i, j, idx);
                }
        }, sum);
        idx += (start < end) ? 1 : -1;
    }
    idx += (start < end) ? -1 : 1; // accounting for overshoot
    return idx;
}

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
        
        // finding x min
        int i = find_first_nonzero_layer(pincell_in, overlap_box.min[0], overlap_box.max[0], overlap_box, 0);
        non_zero_box.min[0] = proc_box.min[0] + i;

        // finding y min
        int j = find_first_nonzero_layer(pincell_in, overlap_box.min[1], overlap_box.max[1], overlap_box, 1);
        non_zero_box.min[1] = proc_box.min[1] + j;

        // finding z min
        int k = find_first_nonzero_layer(pincell_in, overlap_box.min[2], overlap_box.max[2], overlap_box, 2);
        non_zero_box.min[2] = proc_box.min[2] + k;

        // finding x min
        i = find_first_nonzero_layer(pincell_in, overlap_box.max[0]-1, overlap_box.min[0]-1, overlap_box, 0);
        non_zero_box.max[0] = proc_box.min[0] + i;

        // finding y min
        j = find_first_nonzero_layer(pincell_in, overlap_box.max[1]-1, overlap_box.min[1]-1, overlap_box, 1);
        non_zero_box.max[1] = proc_box.min[1] + j;

        // finding z min
        k = find_first_nonzero_layer(pincell_in, overlap_box.max[2]-1, overlap_box.min[2]-1, overlap_box, 2);
        non_zero_box.max[2] = proc_box.min[2] + k;

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
        const int offax[2] {(cax+1)%3, (cax+2)%3}, i = id, globali = minidx[cax] + i, 
            nodei = globali - node_box.min[cax];
        for (int j = 0; j < p.extent(offax[0]); ++j) {
            for (int k = 0; k < p.extent(offax[1]); ++k) {
                const int globalj = minidx[offax[0]] + j, globalk = minidx[offax[1]] + k;
                if (cax == 0) {
                    if (contains_point(node_box, globali, globalj, globalk)) gridsums(nodei) += p(i, j, k);
                } else if (cax == 1) {
                    if (contains_point(node_box, globalk, globali, globalj)) gridsums(nodei) += p(k, i, j);
                } else if (cax == 2) {
                    if (contains_point(node_box, globalj, globalk, globali)) gridsums(nodei) += p(j, k, i);
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
    Kokkos::View<int*> gridsums("Grid sums by layer", range(node_in.extents, cax));
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
    const Kokkos::View<int*> neighbours;
    const int buffer, procid;
    locate_neighbours_func(Kokkos::View<const box<int>*> all_extents_, Kokkos::View<int*> neighbours_, 
        int buffer_, int procid_)
        : all_extents {all_extents_}
        , neighbours {neighbours_}
        , buffer {buffer_}
        , procid {procid_}
    {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int other_procid, int &partial_sum, const bool is_final) const {
        const bool is_neighbour = !no_overlap(
            all_extents(procid),
            extend(all_extents(other_procid), buffer)
        ) && procid != other_procid;
        if (is_final && is_neighbour) neighbours(partial_sum) = other_procid;
        partial_sum += is_neighbour;
    }
};

static void ORB_find_neighbours(const Kokkos::View<box<int>*> &boundaries, const int procid, const int numprocs, const int buffer, const Kokkos::View<int*> neighbours, int &n_neighbours) {
    Kokkos::parallel_scan(
        "Flag processes as neighbour", 
        numprocs, 
        locate_neighbours_func(
            boundaries, 
            neighbours, 
            buffer, 
            procid
        ),
        n_neighbours
    );
}

struct ORB_determine_neighbour_halos_func {
    const int procid, numprocs, buffer, n_neighbours;
    const Kokkos::View<const box<int>*> boundaries;
    const Kokkos::View<const int*> neighbours;
    const Kokkos::View<box<int>*> send_halo_boxes, recv_halo_boxes;
    ORB_determine_neighbour_halos_func(const int procid_, const int numprocs_, const int buffer_, 
        const int n_neighbours_, const Kokkos::View<const box<int>*> boundaries_, 
        const Kokkos::View<const int*> neighbours_, const Kokkos::View<box<int>*> send_halo_boxes_, 
        const Kokkos::View<box<int>*> recv_halo_boxes_)
        : procid {procid_}
        , numprocs {numprocs_}
        , buffer {buffer_}
        , n_neighbours {n_neighbours_}
        , boundaries {boundaries_}
        , neighbours {neighbours_}
        , send_halo_boxes {send_halo_boxes_}
        , recv_halo_boxes {recv_halo_boxes_}
    {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        const int neighbouri = neighbours(i);
        const box<int> mybox {boundaries(procid)}, mybox_w_buffer {extend(mybox, buffer)};
        const box<int> neighbourbox {boundaries(neighbouri)}, neighbourbox_w_buffer {extend(neighbourbox, buffer)};
        send_halo_boxes(i) = find_overlapping_box(mybox, neighbourbox_w_buffer);
        recv_halo_boxes(i) = find_overlapping_box(mybox_w_buffer, neighbourbox);
    }
};

static void ORB_determine_neighbour_halos(const Kokkos::View<const box<int>*> &boundaries, const int procid, const int numprocs, const int buffer, const Kokkos::View<const int*> neighbours,
    const int &n_neighbours, const Kokkos::View<box<int>*> &send_halo_boxes, const Kokkos::View<box<int>*>  &recv_halo_boxes) {
    
    Kokkos::parallel_for("finding halo regions to send/recv", 
        n_neighbours, 
        ORB_determine_neighbour_halos_func(
            procid, 
            numprocs, 
            buffer, 
            n_neighbours, 
            boundaries, 
            neighbours, 
            send_halo_boxes, 
            recv_halo_boxes
        )
    );
}

namespace GraMPM {
    namespace accelerated {
        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::ORB_determine_boundaries() {
            
            // initiate sharing of number of points amongst all processes
            MPI_Request req;
            MPI_Iallreduce(&m_p_size, &m_p_size_global, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, &req);

            // find local minima/maxima of grid extents
            box<int> proc_box;
            Kokkos::parallel_reduce(
                "reduce",
                m_p_size,
                ORB_find_local_coverage_func(m_p_grid_idx.d_view, m_ngrid.data()),
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
                ORB_populate_pincell_func(m_p_grid_idx.d_view, pincell, m_ngrid.data(), proc_box.min)
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
            ORB<F>(procid, node1, proc_box, m_ORB_extents.h_view, m_g_extents, m_g_cell_size, pincell);

            Kokkos::deep_copy(m_ORB_extents.d_view, m_ORB_extents.h_view);

            ORB_find_neighbours(m_ORB_extents.d_view, procid, numprocs, int(knl.radius), m_ORB_neighbours.d_view, 
                n_ORB_neighbours);

            ORB_determine_neighbour_halos(m_ORB_extents.d_view, procid, numprocs, int(knl.radius), 
                m_ORB_neighbours.d_view, n_ORB_neighbours, m_ORB_send_halo.d_view, m_ORB_recv_halo.d_view);

        }
    }
}


#endif
