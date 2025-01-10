#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#define SRC 0
struct csrGraph{
    int * offset;
    int * edges;
    int * weights;
    int * dist;
};

int vertex_count, edge_count, max_degree;

struct csrGraph * initialize(int vertex_count, int edge_count);

struct csrGraph * getInput(){
    scanf("%d %d", &vertex_count, &edge_count);
    struct csrGraph *graph = initialize(vertex_count, edge_count);
    for(int i=0;i<vertex_count+1;i+=1)
        scanf("%d", graph->offset+i);
    for(int i=0;i<edge_count;i+=1)
        scanf("%d", graph->edges+i);
    for(int i=0;i<edge_count;i+=1)
        scanf("%d", graph->weights+i);
    max_degree = 0;
    for(int i=1;i<vertex_count+1;i+=1)
        max_degree = max_degree>(graph->offset[i]-graph->offset[i-1])?max_degree:(graph->offset[i]-graph->offset[i-1]); 
    return graph;
}

struct csrGraph * initialize(int vertex_count, int edge_count){
    struct csrGraph *graph = malloc(sizeof(struct csrGraph));
    graph->offset = malloc((vertex_count+1)*sizeof(int));
    graph->edges = malloc((edge_count)*sizeof(int));
    graph->weights = malloc((edge_count)*sizeof(int));
    graph->dist = malloc((vertex_count)*sizeof(int));
    for(int i=0;i<vertex_count;i+=1)
        graph->dist[i] = 1e9;
    return graph;
}

void resetDist(struct csrGraph * graph){
    for(int i=0;i<vertex_count;i+=1){
        if(i==SRC)graph->dist[i] = 0;
        else graph->dist[i] = 1e9;
    }
}

int main(int argc, char* argv[]){
    struct csrGraph * graph = getInput();

    FILE *output, *time;
    output = fopen(argv[1], "w");
    time = fopen(argv[2], "a");
    
    int flag = 0;

    int * worklist;
    int buff = 2*edge_count;
    worklist = malloc(2*2*edge_count*sizeof(int));

    clock_t start, end;
    double tim = 0;
    for(int i=0;i<5;i++){
        start = clock();

        int size = 1;
        worklist[0] = SRC;
        graph->dist[SRC] = 0;

        while(size!=0){
            int new_size = 0;
            int *in_ptr, *out_ptr;
            if(flag==0){
                in_ptr = worklist;
                out_ptr = worklist + buff;
            }
            else{
                in_ptr = worklist + buff;
                out_ptr = worklist;
            }
            for(int i=0;i<size;i+=1){
                int vert = in_ptr[i];
                for(int j=graph->offset[vert]; j<graph->offset[vert+1];j++){
                    int new_vert = graph->edges[j];
                    int weight = graph->weights[j];
                    int old_d = graph->dist[new_vert];
                    if(old_d > graph->dist[vert] + weight){
                        graph->dist[new_vert] = graph->dist[vert] + weight;
                        out_ptr[new_size] = new_vert;
                        new_size += 1;
                    }
                }
            }
            size = new_size;
            flag = 1-flag;
        }

        end = clock();
        printf("\t%d : %f\n",i,((double)(end-start))/CLOCKS_PER_SEC);
        if(i!=0)tim += ((double)(end-start))/CLOCKS_PER_SEC;
        resetDist(graph);
    }
    fprintf(time, "%12.6f", tim/4);

    //for(int i=0;i<vertex_count;i++)
    //    fprintf(output, "%d ", graph->dist[i]);

}


