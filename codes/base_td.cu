#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define SRC 0
#define nodes_per_thread 2

int vertex_count, edge_count, worklist_size;

struct csrGraph{
    int * offset;
    int * edges;
    int * weights;
    int * dist;
};

__global__ void copyKernel(struct csrGraph * d_graph, int * d_offset, int * d_edges, int * d_weights, int * d_dist){
    d_graph->offset = d_offset;
    d_graph->edges = d_edges;
    d_graph->weights = d_weights;
    d_graph->dist = d_dist;
}

__global__ void copyKernel2(struct csrGraph * d_graph, int ** dist){
    dist[0] = d_graph->dist;
}

__global__ void resetDist(struct csrGraph * d_graph, int vertex_count){
    int global_id = threadIdx.x + blockIdx.x*blockDim.x;
    if(global_id==SRC)d_graph->dist[global_id]=0;
    else if(global_id<vertex_count)d_graph->dist[global_id]=1e9;
}

struct csrGraph * initialize(int vertex_count, int edge_count){
    struct csrGraph *graph = (struct csrGraph*)malloc(sizeof(struct csrGraph));
    graph->offset = (int*)malloc((vertex_count+1)*sizeof(int));
    graph->edges = (int*)malloc((edge_count)*sizeof(int));
    graph->weights = (int*)malloc((edge_count)*sizeof(int));
    graph->dist = (int*)malloc((vertex_count)*sizeof(int));
    for(int i=0;i<vertex_count;i+=1)
        graph->dist[i] = 1e9;
    graph->dist[SRC]=0;
    return graph;
}

struct csrGraph * getInput(){
    scanf("%d %d", &vertex_count, &edge_count);

    struct csrGraph *graph = initialize(vertex_count, edge_count);

    for(int i=0;i<vertex_count+1;i+=1)
        scanf("%d", graph->offset+i);
    for(int i=0;i<edge_count;i+=1)
        scanf("%d", graph->edges+i);
    for(int i=0;i<edge_count;i+=1)
        scanf("%d", graph->weights+i);

    return graph;
}

struct csrGraph * copyGraphToGPU(struct csrGraph * graph){
    struct csrGraph * d_graph;
    cudaMalloc(&d_graph, sizeof(struct csrGraph));
    
    int * d_offset, *d_edges, *d_weights, *d_dist;
    cudaMalloc(&d_offset, (vertex_count+1)*sizeof(int));
    cudaMalloc(&d_edges, (edge_count)*sizeof(int));
    cudaMalloc(&d_weights, (edge_count)*sizeof(int));
    cudaMalloc(&d_dist, (vertex_count)*sizeof(int));

    cudaMemcpy(d_offset, graph->offset, (vertex_count+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, graph->edges, (edge_count)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, graph->weights, (edge_count)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, graph->dist, (vertex_count)*sizeof(int), cudaMemcpyHostToDevice);
    
    copyKernel<<<1, 1>>>(d_graph, d_offset, d_edges, d_weights, d_dist);
    cudaDeviceSynchronize();
    
    return d_graph;
}

void copyGraphToCPU(struct csrGraph * graph, struct csrGraph * d_graph){
    int ** dist;
    cudaHostAlloc(&dist, sizeof(int*), 0);
    copyKernel2<<<1, 1>>>(d_graph, dist);
    cudaDeviceSynchronize();
    cudaMemcpy(graph->dist, dist[0], vertex_count*sizeof(int), cudaMemcpyDeviceToHost);
}

__global__ void processKernel(struct csrGraph * graph, int * changed, int vertex_count){
    int global_id = threadIdx.x + blockDim.x*blockIdx.x;

    if(global_id<vertex_count){
        int start_vertex = global_id*nodes_per_thread;
        int end_vertex = (global_id+1)*nodes_per_thread;

        for(int v = start_vertex; v<end_vertex;v++){
            int start = graph->offset[v];
            int end = graph->offset[v+1];
            for(int i = start;i<end;i++){
                int end_vert = graph->edges[i];
                int weight = graph->weights[i];
                if(graph->dist[end_vert]>graph->dist[v]+weight){
                    graph->dist[end_vert] = graph->dist[v]+weight;
                    *changed=1;
                }
            }
        }
    }

}


int main(int argc, char* argv[]){
    FILE *output, *time;
    output = fopen(argv[1], "w");
    time = fopen(argv[2], "a");

    struct csrGraph * graph = getInput();

    struct csrGraph * d_graph = copyGraphToGPU(graph);


    int * changed;
    cudaHostAlloc(&changed, sizeof(int), 0);

    clock_t start, end;
    double tim = 0;
    for(int i=0;i<5;i++){
        start = clock();

        *changed = 1;

        while(*changed==1){
            *changed=0;

            processKernel<<<(1023+(vertex_count+nodes_per_thread-1)/nodes_per_thread)/1024, 1024>>>(d_graph, changed, vertex_count);
            cudaDeviceSynchronize();
        }

        end = clock();
        printf("\t%d : %f\n",i,((double)(end-start))/CLOCKS_PER_SEC);
        if(i!=0)tim += ((double)(end-start))/CLOCKS_PER_SEC;
        resetDist<<<(1023+vertex_count)/1024, 1024>>>(d_graph, vertex_count);
    }
    fprintf(time, "%12.6f", tim/4);

    copyGraphToCPU(graph, d_graph);
    cudaDeviceSynchronize();

/*
    for(int i=0;i<vertex_count;i++){
        fprintf(output, "%d ", graph->dist[i]);
    } 
*/    
    
}

