// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include "assert.h"


const int LEN = 64;
const int BASELEN = LEN/2;
const int BLOCK_SIZE = LEN * LEN;
int n_block;


namespace {

__global__ void kernel_ref(int n, int k, int *graph) {
    // 每个线程(j,i)每次对graph的(j,i)进行计算: (j,i) <- (j,k) + (k,i)
    //... i,j
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
    }
}

__global__ void step1(int n, int *graph, int p)
{
    auto in_i = threadIdx.y;
    auto in_j = threadIdx.x;

    auto i = p * LEN + in_i;
    auto j = p * LEN + in_j;

    // step1
    // __shared__ int graph_in_block[LEN * LEN];
    __shared__ int graph_in_block[LEN ][ LEN+1];
    // TODO: 用二维数组会不会快一些?

    int id_in_block = threadIdx.y * LEN + threadIdx.x;
    if (i < n && j < n) 
        graph_in_block[in_i][in_j] = graph[i * n + j];
    if(i + BASELEN < n && j < n)
        graph_in_block[in_i+BASELEN][in_j] = graph[(i+BASELEN) * n + j];
    if(i< n && j + BASELEN < n)
        graph_in_block[in_i][in_j+BASELEN] = graph[(i) * n + j+BASELEN];
    if(i + BASELEN < n && j + BASELEN < n)
        graph_in_block[in_i+BASELEN][in_j + BASELEN] = graph[(i+BASELEN) * n + j + BASELEN];
        // graph_in_block[id_in_block] = graph[i * n + j];

    __syncthreads();

    if (i < n && j < n) { 
        // graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
        for (int k = 0; k < LEN && k < n - p * LEN; ++k)
        {
            // graph_in_block[id_in_block] = min(graph_in_block[id_in_block], 
            //     graph_in_block[ in_i * LEN + k ] + 
            //     graph_in_block[ k * LEN + in_j ] );
            graph_in_block[in_i][in_j] = min(graph_in_block[in_i][in_j],
                graph_in_block[in_i][k] +
                graph_in_block[ k ][ in_j ]);
            
            graph_in_block[in_i][in_j + BASELEN] = min(graph_in_block[in_i][in_j + BASELEN],
                graph_in_block[in_i][k] +
                graph_in_block[ k ][ in_j + BASELEN ]);
                
            graph_in_block[in_i + BASELEN][in_j] = min(graph_in_block[in_i + BASELEN][in_j],
                graph_in_block[in_i + BASELEN][k] +
                graph_in_block[ k ][ in_j ]);
            
            graph_in_block[in_i + BASELEN][in_j + BASELEN] = 
            min(graph_in_block[in_i + BASELEN][in_j + BASELEN],
                graph_in_block[in_i + BASELEN][k] +
                graph_in_block[ k ][ in_j + BASELEN ]);
             __syncthreads();
        }
    }
    
    if (i < n && j < n) 
        graph[i * n + j] = graph_in_block[in_i][in_j];
    if(i + BASELEN < n && j < n)
        graph[(i+BASELEN) * n + j] = graph_in_block[in_i+BASELEN][in_j];
    if(i< n && j + BASELEN < n)
        graph[(i) * n + j+BASELEN] = graph_in_block[in_i][in_j+BASELEN];
    if(i + BASELEN < n && j + BASELEN < n)
        graph[(i+BASELEN) * n + j + BASELEN] = graph_in_block[in_i+BASELEN][in_j + BASELEN];
        // graph[i * n + j] = graph_in_block[id_in_block];
    // in block : (y, x): (i, j) -> (i - blockIdx.y * blockDim.y, j - blockIdx.x * blockDim.x)
}

__global__ void step2(int n, int *graph, int p)
{
    // 线程块的位置
    int block_i, block_j;
    if(blockIdx.y == 0)  //竖
    {
        if(blockIdx.x < p) block_i = blockIdx.x;
        else block_i = blockIdx.x + 1;
        block_j = p;
    }
    else //blockIdx.y == 1  横
    {
        if(blockIdx.x < p) block_j = blockIdx.x;
        else block_j = blockIdx.x + 1;
        block_i = p;
    }
    //dangerous

    auto in_i = threadIdx.y;
    auto in_j = threadIdx.x;

    auto i = block_i * LEN + in_i;
    auto j = block_j * LEN + in_j;
    
    auto ci = p * LEN + in_i;
    auto cj = p * LEN + in_j;
    
    // __shared__ int graph_in_block[BLOCK_SIZE * 2];
    __shared__ int graph_in_block[LEN*2 ][ LEN+1];

    int id_in_block = in_i * LEN + in_j;
    if (i < n && j < n) 
    {
        // graph_in_block[id_in_block] = graph[i * n + j];
        graph_in_block[in_i][in_j] = graph[i * n + j];
    }
    if(i + BASELEN < n && j < n)
        graph_in_block[in_i+BASELEN][in_j] = graph[(i+BASELEN) * n + j];
    if(i< n && j + BASELEN < n)
        graph_in_block[in_i][in_j+BASELEN] = graph[(i) * n + j+BASELEN];
    if(i + BASELEN < n && j + BASELEN < n)
        graph_in_block[in_i+BASELEN][in_j + BASELEN] = graph[(i+BASELEN) * n + j + BASELEN];
    
    if( ci < n && cj < n)
    {
        // graph_in_block[id_in_block + BLOCK_SIZE] = graph[ci * n + cj];  //center block
        graph_in_block[in_i + LEN][in_j] = graph[ci * n + cj];
    }
    if( ci + BASELEN < n && cj < n)
        graph_in_block[in_i+ BASELEN + LEN][in_j] = graph[(ci+ BASELEN) * n + cj];
    if( ci < n && cj + BASELEN < n)
        graph_in_block[in_i + LEN][in_j+ BASELEN] = graph[(ci) * n + cj+ BASELEN];
    if( ci + BASELEN < n && cj+ BASELEN < n)
        graph_in_block[in_i+ BASELEN + LEN][in_j+ BASELEN] = graph[(ci+ BASELEN) * n + cj+ BASELEN];

    __syncthreads();

    // step2
    // (i,k) or (k,j) is in the (i,j) block
    if (i < n && j < n) 
    {
        int kMax = n - p * LEN;
        kMax = min( kMax, min(n - block_i * LEN, n - block_j * LEN) );
        if(blockIdx.y == 1)
        for (int k = 0; k < LEN && k < kMax; ++k)
        {
            // (i,j) and (k,j) is in the same block
            // (i,k) is in the center block
            
            graph_in_block[in_i][in_j] = min(graph_in_block[in_i][in_j],
            graph_in_block[in_i+LEN][k] +
            graph_in_block[ k ][ in_j ]);
            
            graph_in_block[in_i][in_j + BASELEN] = min(graph_in_block[in_i][in_j + BASELEN],
                graph_in_block[in_i+LEN][k] +
                graph_in_block[ k ][ in_j + BASELEN ]);
            
            graph_in_block[in_i + BASELEN][in_j] = min(graph_in_block[in_i + BASELEN][in_j],
                graph_in_block[in_i + BASELEN+LEN][k] +
                graph_in_block[ k ][ in_j ]);
            
            graph_in_block[in_i + BASELEN][in_j + BASELEN] = 
            min(graph_in_block[in_i + BASELEN][in_j + BASELEN],
                graph_in_block[in_i + BASELEN + LEN][k] +
                graph_in_block[ k ][ in_j + BASELEN ]);

            __syncthreads(); // 好像去掉快一点？
        }
        else
        for (int k = 0; k < LEN && k < kMax; ++k)
        {
            graph_in_block[in_i][in_j] = min(graph_in_block[in_i][in_j],
            graph_in_block[in_i][k] +
            graph_in_block[ k + LEN ][ in_j ]);
            
            graph_in_block[in_i][in_j + BASELEN] = min(graph_in_block[in_i][in_j + BASELEN],
                graph_in_block[in_i][k] +
                graph_in_block[ k  + LEN ][ in_j + BASELEN ]);

            graph_in_block[in_i + BASELEN][in_j] = min(graph_in_block[in_i + BASELEN][in_j],
                graph_in_block[in_i + BASELEN][k] +
                graph_in_block[ k + LEN  ][ in_j ]);
            
            graph_in_block[in_i + BASELEN][in_j + BASELEN] = 
            min(graph_in_block[in_i + BASELEN][in_j + BASELEN],
                graph_in_block[in_i + BASELEN][k] +
                graph_in_block[ k + LEN  ][ in_j + BASELEN ]);
            __syncthreads();
        }
    }
    
    if (i < n && j < n) 
        graph[i * n + j] = graph_in_block[in_i][in_j];
    if(i + BASELEN < n && j < n)
        graph[(i+BASELEN) * n + j] = graph_in_block[in_i+BASELEN][in_j];
    if(i< n && j + BASELEN < n)
        graph[(i) * n + j+BASELEN] = graph_in_block[in_i][in_j+BASELEN];
    if(i + BASELEN < n && j + BASELEN < n)
        graph[(i+BASELEN) * n + j + BASELEN] = graph_in_block[in_i+BASELEN][in_j + BASELEN];

}

__global__ void step3(int n, int *graph, int p)  //nblock * nblock  blocks
{
    
    auto i = blockIdx.y * LEN + threadIdx.y;
    auto j = blockIdx.x * LEN + threadIdx.x;

    auto in_i = threadIdx.y;
    auto in_j = threadIdx.x;

    int pLEN = p * LEN;

    if(!(pLEN <= i && i < (p + 1) * LEN && pLEN <= j && j < (p + 1) * LEN)) 
    {
        // __shared__ int graph_in_block[BLOCK_SIZE * 3];
        __shared__ int graph_in_block[LEN * 2][LEN+1];

        // if(i < n && j < n)
        //     graph_in_block[in_i][in_j] = graph[i * n + j];
        // if(i + BASELEN < n && j < n)
        //     graph_in_block[in_i+BASELEN][in_j] = graph[(i+BASELEN) * n + j];
        // if(i< n && j + BASELEN < n)
        //     graph_in_block[in_i][in_j+BASELEN] = graph[(i) * n + j+BASELEN];
        // if(i + BASELEN < n && j + BASELEN < n)
        //     graph_in_block[in_i+BASELEN][in_j + BASELEN] = graph[(i+BASELEN) * n + j + BASELEN];
        
        // if(i < n && pLEN + in_j < n)
            graph_in_block[in_i ][in_j] = (i < n && pLEN + in_j < n)? graph[i * n + pLEN + in_j] : __INT_MAX__;
        // if(i < n && pLEN + in_j + BASELEN < n)
            graph_in_block[in_i ][in_j+ BASELEN] = (i < n && pLEN + in_j + BASELEN < n)? graph[i * n + pLEN + in_j+ BASELEN]: __INT_MAX__;
        // if(i + BASELEN < n && pLEN + in_j < n)
            graph_in_block[in_i + BASELEN ][in_j] = (i + BASELEN < n && pLEN + in_j < n)? graph[(i+ BASELEN) * n + pLEN + in_j]:__INT_MAX__;
        // if(i + BASELEN < n && pLEN + in_j+ BASELEN < n)
            graph_in_block[in_i + BASELEN ][in_j+ BASELEN] = (i + BASELEN < n && pLEN + in_j+ BASELEN < n)? graph[(i+ BASELEN) * n + pLEN + in_j + BASELEN]:__INT_MAX__;
        
        
        // if(pLEN + in_i < n && j < n)
            graph_in_block[in_i + LEN][in_j] = (pLEN + in_i < n && j < n)?graph[(pLEN + in_i) * n + j]:__INT_MAX__;
        // if(pLEN + in_i < n && j+ BASELEN < n)
            graph_in_block[in_i + LEN][in_j+ BASELEN] = (pLEN + in_i < n && j+ BASELEN < n)?graph[(pLEN + in_i) * n + j+ BASELEN]:__INT_MAX__;
        // if(pLEN + in_i + BASELEN < n && j < n)
            graph_in_block[in_i+ BASELEN + LEN][in_j] = (pLEN + in_i + BASELEN < n && j < n)?graph[(pLEN + in_i+ BASELEN) * n + j]:__INT_MAX__;
        // if(pLEN + in_i+ BASELEN < n && j+ BASELEN < n)
            graph_in_block[in_i+ BASELEN + LEN][in_j+ BASELEN] = (pLEN + in_i+ BASELEN < n && j+ BASELEN < n)?graph[(pLEN + in_i+ BASELEN) * n + j+ BASELEN]:__INT_MAX__;
        //即使i >= n or j >= n，也要进行copy，给同块元素使用

        __syncthreads();

        int m1=(i < n && j < n) ?graph[i * n + j]:__INT_MAX__,
            m3=(i< n && j + BASELEN < n)?graph[(i) * n + j+BASELEN]:__INT_MAX__,
            m2=(i + BASELEN < n && j < n)?graph[(i+BASELEN) * n + j]:__INT_MAX__,
            m4=(i + BASELEN < n && j + BASELEN < n)?graph[(i+BASELEN) * n + j + BASELEN]:__INT_MAX__;
        // if (i < n && j < n) 
            // m1=(i < n && j < n) ?graph[i * n + j]:__INT_MAX__;
        // if(i + BASELEN < n && j < n)
            // m2=(i + BASELEN < n && j < n)?graph[(i+BASELEN) * n + j]:__INT_MAX__;
        // if(i< n && j + BASELEN < n)
            // m3=(i< n && j + BASELEN < n)?graph[(i) * n + j+BASELEN]:__INT_MAX__;
        // if(i + BASELEN < n && j + BASELEN < n)
            // m4=(i + BASELEN < n && j + BASELEN < n)?graph[(i+BASELEN) * n + j + BASELEN]:__INT_MAX__;
            
        if (i < n && j < n) 
        {
            // for (int k = pLEN; k < (p + 1) * LEN && k < n; ++k)
            // {  
            //     graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
            // }
            int kMax = min(n - pLEN, LEN);

            #pragma unroll 32
            for (int k = 0; k < kMax; ++k)
            {
                // graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
                // graph_in_block[id_in_block] = min(graph_in_block[id_in_block], 
                //     graph_in_block[in_i * LEN + k + BLOCK_SIZE] + 
                //     graph_in_block[k * LEN + in_j + 2 * BLOCK_SIZE] ) ;

                // graph_in_block[in_i][in_j] = min(graph_in_block[in_i][in_j],
                m1=min(m1, 
                    graph_in_block[in_i ][k] +
                    graph_in_block[ k + LEN ][ in_j ]);
                    
                // graph_in_block[in_i][in_j + BASELEN] = min(graph_in_block[in_i][in_j + BASELEN],
                m3=min(m3, 
                    graph_in_block[in_i ][k] +
                    graph_in_block[ k  + LEN ][ in_j + BASELEN ]);
            
                // graph_in_block[in_i + BASELEN][in_j] = min(graph_in_block[in_i + BASELEN][in_j],
                m2=min(m2, 
                    graph_in_block[in_i + BASELEN ][k] +
                    graph_in_block[ k + LEN  ][ in_j ]);
                
                // graph_in_block[in_i + BASELEN][in_j + BASELEN] = 
                // min(graph_in_block[in_i + BASELEN][in_j + BASELEN],
                m4=min(m4,
                    graph_in_block[in_i + BASELEN ][k] +
                    graph_in_block[ k + LEN  ][ in_j + BASELEN ]);
                
                // __syncthreads();
            }

            // graph_in_block[in_i][in_j] = min(graph_in_block[in_i][in_j], m1);
            // graph_in_block[in_i + BASELEN][in_j] = min(graph_in_block[in_i + BASELEN][in_j], m2);
            // graph_in_block[in_i][in_j + BASELEN] = min(graph_in_block[in_i][in_j + BASELEN], m3);
            // graph_in_block[in_i + BASELEN][in_j + BASELEN] = 
            //     min(graph_in_block[in_i + BASELEN][in_j + BASELEN], m4);
        }

        if (i < n && j < n) 
            // graph[i * n + j] = graph_in_block[in_i][in_j];
            graph[i * n + j] = m1;
        if(i< n && j + BASELEN < n)
            // graph[(i) * n + j+BASELEN] = graph_in_block[in_i][in_j+BASELEN];
            graph[(i) * n + j+BASELEN] = m3;
        if(i + BASELEN < n && j < n)
            // graph[(i+BASELEN) * n + j] = graph_in_block[in_i+BASELEN][in_j];
            graph[(i+BASELEN) * n + j]  = m2;
        if(i + BASELEN < n && j + BASELEN < n)
            // graph[(i+BASELEN) * n + j + BASELEN] = graph_in_block[in_i+BASELEN][in_j + BASELEN];
            graph[(i+BASELEN) * n + j + BASELEN] = m4;
            // graph[i * n + j] = graph_in_block[id_in_block];
    }
}

/*__global__ void step3(int n, int *graph, int p)  //(nblock-1) * (nblock-1)  blocks
{
    // 线程块的位置
    int block_i, block_j;
    if(blockIdx.y < p) block_i = blockIdx.y;
    else block_i = blockIdx.y + 1;
    if(blockIdx.x < p) block_j = blockIdx.x;
    else block_j = blockIdx.x + 1;
    
    auto in_i = threadIdx.y;
    auto in_j = threadIdx.x;

    auto i = block_i * LEN + in_i;
    auto j = block_j * LEN + in_j;

    assert(!(p * LEN <= i && i < (p + 1) * LEN && p * LEN <= j && j < (p + 1) * LEN));
    // __shared__ int graph_in_block[BLOCK_SIZE * 3];

    // int id_in_block = in_i * LEN + in_j;

    // if(i < n && j < n)
    //     graph_in_block[id_in_block] = graph[i * n + j];
    // if(i < n && p * LEN + in_j < n)
    //     graph_in_block[id_in_block + BLOCK_SIZE] = graph[i * n + p * LEN + in_j];
    // if(p * LEN + in_i < n && j < n)
    //     graph_in_block[id_in_block + 2 * BLOCK_SIZE] = graph[(p * LEN + in_i) * n + j];

    // __syncthreads();
    
    if (i < n && j < n) 
    {
        // int kMax = n - p * LEN;
        // for (int k = 0; k < LEN && k < kMax; ++k)
        for (int k = p * LEN; k < (p + 1) * LEN && k < n; ++k)
        {
            // assert( p * LEN <= k && k < (p + 1) * LEN && k < n);
            graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
            // graph_in_block[id_in_block] = min(graph_in_block[id_in_block], 
            //     graph_in_block[in_i * LEN + k + BLOCK_SIZE] + 
            //     graph_in_block[k * LEN + in_j + 2 * BLOCK_SIZE] ) ;
            
            __syncthreads();
        }
    }
    // if (i < n && j < n) 
    //     graph[i * n + j] = graph_in_block[id_in_block];

    //边界处理有问题，我1024可以过
}*/


} //this is for namespace

void apsp(int n, /* device */ int *graph) {
    // for (int k = 0; k < n; k++) {
    //     dim3 thr(LEN, LEN);
    //     dim3 blk((n - 1) / LEN + 1, (n - 1) / LEN + 1);
    //     kernel_ref<<<blk, thr>>>(n, k, graph);
    // }

    dim3 thr(BASELEN, BASELEN);
    dim3 bigthr(LEN, LEN);
    n_block = (n - 1) / LEN + 1;
    
    dim3 blk(n_block, n_block);

    dim3 blk_step1(1);
    dim3 blk_step2(n_block - 1, 2);
    dim3 blk_step3(n_block - 1, n_block - 1);
    int pMax = (n - 1) / LEN + 1;
    
    for(int p = 0; p < pMax; ++p)
    {
        // step1
        step1<<<blk_step1, thr>>>(n, graph, p);
        
        //step2
        step2<<<blk_step2, thr>>>(n, graph, p);

        //step3
        // for (int k = p * LEN; k < (p + 1) * LEN && k < n; ++k)
        // {
            // step3<<<blk, thr>>>(n, k, graph, p);
        // }
        // step3<<<blk_step3, thr>>>(n, graph, p);
        step3<<<blk, thr>>>(n, graph, p);
    }
}

