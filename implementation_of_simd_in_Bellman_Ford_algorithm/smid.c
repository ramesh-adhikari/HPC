#include <arm_neon.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define V 4     // number of vertices
#define E 5     // number of edges

// edge struct
struct Edge {
    int src, dest;
    float weight;
};

// function to run Bellman-Ford algorithm
void bellmanFord(struct Edge edges[], int src)
{
    // allocate distance array and initialize it to infinity
    float dist[V];
    for (int i = 0; i < V; i++) {
        dist[i] = FLT_MAX;
    }
    dist[src] = 0;

    // use NEON SIMD to iterate over edges
    float32x4_t w_vec;
    float32x4_t dist_src_vec, dist_dest_vec;
    for (int i = 0; i < V - 1; i++) {
        for (int j = 0; j < E; j += 4) {
            // load edge weights and distance values into NEON registers
            w_vec = vld1q_f32(&edges[j].weight);
            dist_src_vec = vld1q_f32(&dist[edges[j].src]);
            dist_dest_vec = vld1q_f32(&dist[edges[j].dest]);

            // add edge weight to source vertex distance
            dist_src_vec = vaddq_f32(dist_src_vec, w_vec);

            // compare updated distance values to current values and store minimum
            float32x4_t cmp_vec = vcltq_f32(dist_src_vec, dist_dest_vec);
            dist_dest_vec = vbslq_f32(cmp_vec, dist_src_vec, dist_dest_vec);

            // store updated distance values back to distance array
            vst1q_f32(&dist[edges[j].dest], dist_dest_vec);
        }
    }

    // print final distance array
    printf("Vertex Distance from Source:\n");
    for (int i = 0; i < V; i++) {
        if (dist[i] == FLT_MAX) {
            printf("%d \t INF\n", i);
        } else {
            printf("%d \t %0.2f\n", i, dist[i]);
        }
    }
}

// main function
int main()
{
    // initialize edge array
    struct Edge edges[E];
    edges[0].src = 0;
    edges[0].dest = 1;
    edges[0].weight = -1.0;
    edges[1].src = 0;
    edges[1].dest = 2;
    edges[1].weight = 4.0;
    edges[2].src = 1;
    edges[2].dest = 2;
    edges[2].weight = 3.0;
    edges[3].src = 1;
    edges[3].dest = 3;
    edges[3].weight = 2.0;
    edges[4].src = 3;
    edges[4].dest = 2;
    edges[4].weight = 5.0;

    // run Bellman-Ford algorithm with source vertex 0
    bellmanFord(edges, 0);

    return 0;
}
