#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define SRC 0

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

void initializeWorklist(int * worklist){
    worklist_size = 5*edge_count;
    worklist = (int*)malloc(2*worklist_size*sizeof(int));
    worklist[0] = SRC;
}

int * copyWorklist(int * worklist){
    int * d_worklist;
    cudaMalloc(&d_worklist, 2*worklist_size*sizeof(int));
    cudaMemcpy(d_worklist, worklist, 2*worklist_size*sizeof(int), cudaMemcpyHostToDevice);
    return d_worklist;
}

__device__ void prefix(int * a){
    int id = threadIdx.x;
    int x=1;
    while(x<1024){
        int to_add =0;
        if(id-x>=0)to_add = a[id-x];
        __syncthreads();
        a[id]+=to_add;
        x*=2;
        __syncthreads();
        __threadfence();
    }
}

__global__ void processKernel(struct csrGraph *graph, int * worklist, int * count, int * count_, int * flag, int worklist_capacity){
    int * in_worklist = (*flag==0)?worklist:(worklist+worklist_capacity);    
    int * out_worklist = (*flag==1)?worklist:(worklist+worklist_capacity);    

    int global_id = threadIdx.x+blockIdx.x*blockDim.x;
    int block_id = threadIdx.x;

    int old_count = (*flag==0)?*count:*count_;

    int updated[9000];
    int updated_list_size=9000;
    int updated_count=0;
    __shared__ int add_count[1024];


    __shared__ int total_work;
    __shared__ int don_box[10*1024];
    __shared__ int idx;
    __shared__ int don_size;

    if(threadIdx.x==0)idx=0, total_work=0, don_size=0;
    __syncthreads();
     

    
    int src_vert, start, end;
    if(global_id<old_count){
        src_vert = in_worklist[global_id];
        start = graph->offset[src_vert];
        end = graph->offset[src_vert+1];
    }
    else{
        src_vert = -1;
        start = 0;
        end = 0;
    }

    atomicAdd(&total_work, end-start);
    __syncthreads();
    
    int ave_work = (1023+total_work)/1024;
    //ave_work-=1;
    //if(ave_work<0)ave_work=0;
    int extra_work=end-start-ave_work;
    if(end-start>ave_work) {

        int st = atomicAdd(&don_size, 2*(end-start-ave_work));
        for(int i=st;i<st+2*(extra_work) and i+1<10*1024;i+=2){
            don_box[i] = src_vert;
            don_box[i+1] = --end;
        }
    }

    __syncthreads();

    if(global_id >= old_count){
        updated_count=0;
    }
    else{
        //int end = graph->offset[src_vert+1];
        
        for(int i=start;i<end;i++){
            int end_vert = graph->edges[i];
            int weight = graph->weights[i];
            
            if(graph->dist[end_vert] > graph->dist[src_vert] + weight){
                graph->dist[end_vert] = graph->dist[src_vert] + weight;
                updated[updated_count++]=end_vert;
                if(updated_count==updated_list_size-1){updated[updated_count++]=src_vert;break;}
            }
        }
        /*for(int i=end;i<graph->offset[src_vert+1];i++){
            int end_vert = graph->edges[i];
            int weight = graph->weights[i];
            
            if(graph->dist[end_vert] > graph->dist[src_vert] + weight){
                graph->dist[end_vert] = graph->dist[src_vert] + weight;
                updated[updated_count++]=end_vert;
            }
        }*/
//        if(updated_count!=0){
//            updated[updated_count++]=in_worklist[global_id];
//        }
//        add_count[block_id] = updated_count;
    }
    /*int old;
    int x;
    do{
        if(updated_count+2>updated_list_size)break;
        old = idx;
        x = atomicCAS(&idx, old, old+2);
        if(x==old and old<don_size and old+1<10*1024){
            int s_v = don_box[old];
            int e_v = graph->edges[don_box[old+1]];
            int w = graph->weights[don_box[old+1]];
            if(graph->dist[e_v]>graph->dist[s_v]+w){
                graph->dist[e_v] = graph->dist[s_v]+w;
                updated[updated_count++]=e_v;
                updated[updated_count++]=s_v;
            }
            x = -1;
        }
    }while(x!=old and old < don_size and old+1<10*1024);*/
    if(idx<don_size and idx<10*1024 and updated_count<updated_list_size-2){
        int don_ind = atomicAdd((int *)&idx, 2);
        while(don_ind<don_size and don_ind+1 < 10*1024 and updated_count<updated_list_size-2){
            int s_v = don_box[don_ind];
            int e_v = graph->edges[don_box[don_ind+1]];
            int w = graph->weights[don_box[don_ind+1]];
            if(graph->dist[e_v]>graph->dist[s_v]+w){
                graph->dist[e_v] = graph->dist[s_v]+w;
                updated[updated_count++]=e_v;
                updated[updated_count++]=s_v;
            }
            if(updated_count>=updated_list_size-2)break;
            don_ind = atomicAdd((int*)&idx, 2);
        }
    }


    /*for(int i=0;i<don_size and i+1 < 10*1024;i+=2){
        if(updated_count>updated_list_size-2)break;
        int s_v = don_box[i];
        if(s_v==src_vert){
            int e_v = graph->edges[don_box[i+1]];
            int w = graph->weights[don_box[i+1]];
            if(graph->dist[e_v]>graph->dist[s_v]+w){
                graph->dist[e_v]=graph->dist[s_v]+w;
                updated[updated_count++]=e_v;
                updated[updated_count++]=s_v;
            }
        }
    }*/



    if(updated_count!=0){
        if(updated_count!=updated_list_size)updated[updated_count++]=in_worklist[global_id];
    }
    add_count[block_id] = updated_count;

    __syncthreads();
    prefix(add_count);

    __shared__ int block_start;
    if(block_id==0)
        block_start= atomicAdd(*flag==0?count_:count, add_count[1023]);
    __syncthreads();
    __threadfence();


    int thread_start = block_start + (block_id==0?0:add_count[block_id-1]);
    for(int i=0;i<updated_count;i+=1){
        out_worklist[thread_start+i] = updated[i];
    }
}

int main(int argc, char* argv[]){
    FILE *output, *time;
    output = fopen(argv[1], "w");
    time = fopen(argv[2], "a");

    struct csrGraph * graph = getInput();

    struct csrGraph * d_graph = copyGraphToGPU(graph);

    int * worklist;
    initializeWorklist(worklist);
    int * d_worklist = copyWorklist(worklist);
    
    int *flag, *count, *count_;
    cudaHostAlloc(&flag, sizeof(int), 0);
    cudaHostAlloc(&count, sizeof(int), 0);
    cudaHostAlloc(&count_, sizeof(int), 0);

    clock_t start, end;
    double tim = 0;
    for(int i=0;i<5;i++){
        start = clock();

        *flag = 0;
        *count = 1;
        *count_ = 0;

        while((*flag==0 && *count!=0) || (*flag==1 && *count_!=0)){
            int c = *flag==0?*count:*count_;
            processKernel<<<(1023+c)/1024, 1024>>>(d_graph, d_worklist, count, count_, flag, worklist_size);
            cudaDeviceSynchronize();
            *flag = 1-*flag;
            if(*flag==1)*count=0;
            else *count_=0;
        }

        end = clock();
        printf("\t%d : %f\n",i,((double)(end-start))/CLOCKS_PER_SEC);
        if(i!=0)tim += ((double)(end-start))/CLOCKS_PER_SEC;
        resetDist<<<(1023+vertex_count)/1024, 1024>>>(d_graph, vertex_count);
        d_worklist = copyWorklist(worklist);
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

