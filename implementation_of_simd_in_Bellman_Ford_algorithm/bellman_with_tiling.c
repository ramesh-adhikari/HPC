#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define INF INT_MAX // infinite distance
#define BLOCK_SIZE 32

struct Edge {
    int from, to, weight;
};

void bellmanFord(int start, int n, struct Edge* edges, int m) {
    int dist[n];
    for (int i = 0; i < n; i++) {
        dist[i] = INF;
    }
    dist[start] = 0;

    for (int k = 0; k < n; k += BLOCK_SIZE) {
        for (int i = 0; i < n; i += BLOCK_SIZE) {
            for (int j = k; j < k + BLOCK_SIZE && j < n; j++) {
                for (int l = i; l < i + BLOCK_SIZE && l < n; l++) {
                    for (int p = 0; p < m; p++) {
                        int u = edges[p].from, v = edges[p].to, w = edges[p].weight;
                        if (u == j && v == l && dist[u] != INF && dist[u] + w < dist[v]) {
                            dist[v] = dist[u] + w;
                        }
                    }
                }
            }
        }
    }

    // check for negative-weight cycles
    for (int i = 0; i < m; i++) {
        int u = edges[i].from, v = edges[i].to, w = edges[i].weight;
        if (dist[u] != INF && dist[u] + w < dist[v]) {
            printf("Graph contains negative-weight cycle\n");
            return;
        }
    }

    // print shortest distances
    printf("Shortest distances from node %d:\n", start);
    for (int i = 0; i < n; i++) {
        printf("%d: %d\n", i, dist[i]);
    }
}

int main() {
    // read data from csv file
    FILE* fin = fopen("data.csv", "r");
    int n, m;
    fscanf(fin, "%d,%d\n", &n, &m);
    struct Edge edges[m];
    for (int i = 0; i < m; i++) {
        int u, v, w;
        fscanf(fin, "%d,%d,%d\n", &u, &v, &w);
        edges[i].from = u;
        edges[i].to = v;
        edges[i].weight = w;
    }
    fclose(fin);

    int start = 0; // source node
    bellmanFord(start, n, edges, m);

    return 0;
}
