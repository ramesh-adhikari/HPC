#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define INF INT_MAX // infinite distance

struct Edge {
    int from, to, weight;
};

void bellmanFord(int start, int n, struct Edge* edges, int m) {
    int dist[n];
    for (int i = 1; i <= n; i++) {
        dist[i] = INF;
    }
    dist[start] = 0;

    // relax edges n-1 times
    for (int i = 1; i < n ; i++) {
        for (int j = 0; j < m; j++) {
            
            int u = edges[j].from, v = edges[j].to, w = edges[j].weight;
           
            if (dist[u] + w < dist[v]) {
               
                dist[v] = dist[u] + w;
                
            }
        }

    }

 

    // check for negative-weight cycles
    for (int i = 1; i < m; i++) {
        int u = edges[i].from, v = edges[i].to, w = edges[i].weight;
        if (dist[u] + w < dist[v]) {
            printf("Graph contains negative-weight cycle\n");
            return;
        }
    }

    // print shortest distances
    printf("Shortest distances from node %d:\n", start);
    for (int i = 1; i <= n; i++) {
        printf("%d: %d\n", i, dist[i]);
    }
}

int main() {
    // read data from csv file
    FILE* fin = fopen("data/data.csv", "r");
    if (!fin) {
        printf("Error opening file");
        exit(1);
    }
    int n, m;
    fscanf(fin, "%d,%d\n", &n, &m);
   
    struct Edge* edges = malloc(m * sizeof(struct Edge));
    for (int i = 0; i < m; i++) {
        int u, v, w;
        fscanf(fin, "%d,%d,%d\n", &u, &v, &w);
        edges[i].from = u;
        edges[i].to = v;
        edges[i].weight = w;
    }
    fclose(fin);

    int start = 1; // source node
    bellmanFord(start, n, edges, m);

    free(edges);
    return 0;
}
