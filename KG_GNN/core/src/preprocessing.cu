#include "../include/KG_GNN.h"
#include <cuda.h>
#include <stdio.h>
#include <vector>

//using namespace std;

// Only partite the neighbours of high-degree nodes
void kg_csr_balance(int m, int nnz, int *RowPtr, int wsize, warp_info **winfo, int *winfo_n)
{
    std::vector<warp_info> host_winfo;
    for (int row = 0; row < m; row++)
    {
        int row_st = RowPtr[row];
        int row_ed = RowPtr[row + 1];

        int wi;
        for (wi = row_st; wi < row_ed - wsize; wi += wsize)
        {
            warp_info tmp = warp_info(row, row + 1, wi, wi + wsize);
            host_winfo.push_back(tmp);
        }
        warp_info tmp = warp_info(row, row + 1, wi, row_ed);
        host_winfo.push_back(tmp);
    }
    //printf("warp num %d size %d\n", host_winfo.size(), host_winfo.size() * sizeof(warp_info));

    *winfo_n = host_winfo.size();
    cudaMalloc(winfo, host_winfo.size() * sizeof(warp_info));
    cudaMemcpy(*winfo, &host_winfo[0], host_winfo.size() * sizeof(warp_info), cudaMemcpyHostToDevice);
}

// Partite nodes into equally sized groups
void kg_csr_balance2(int m, int nnz, int *RowPtr, int wsize, int alpha, warp_info **winfo, int *winfo_n)
{
    std::vector<warp_info> host_winfo;
    std::vector<int> warp_load;

    int group_n = 0;
    int last_start_row = 0;
    int last_end_row = -1;
    int last_start_col, last_end_col;

    for (int row = 0; row < m; row++)
    {
        int row_st = RowPtr[row];
        int row_ed = RowPtr[row + 1];

        // if (row < 50) printf("%d %d %d %d\n", row_st, row_ed, wsize, group_n);

        // Approximation of a write back
        // int alpha = 15;

        if (row_ed - row_st + alpha > wsize - group_n || last_end_row == -1)
        {
            //printf("?\n");
            //printf("%d\n", last_end_row);
            
            if (last_end_row != -1)
            {
                if (wsize - group_n <= alpha)
                //if (true)
                {
                    warp_info tmp = warp_info(last_start_row, last_end_row, last_start_col, last_end_col);
                    host_winfo.push_back(tmp);
                    warp_load.push_back(group_n);
                    group_n = 0;
                }
                else
                {
                    last_end_col += wsize - group_n - alpha;
                    row_st += wsize - group_n - alpha;
                    last_end_row = row + 1;
                    group_n += wsize - group_n;
                    warp_info tmp = warp_info(last_start_row, last_end_row, last_start_col, last_end_col);
                    host_winfo.push_back(tmp);
                    warp_load.push_back(group_n);
                    group_n = 0;
                }
            }

            // if (host_winfo.size() < 50)
            //     printf("%d row %d row_ed %d col %d col_ed %d\n", host_winfo.size(), 
            //     last_start_row, last_end_row, last_start_col, last_end_col);

            int wi;
            for (wi = row_st; wi < row_ed - wsize + alpha; wi += wsize - alpha)
            {
                warp_info tmp = warp_info(row, row + 1, wi, wi + wsize - alpha);
                host_winfo.push_back(tmp);
                warp_load.push_back(wsize);
                group_n = 0;
            }
            last_start_row = row;
            last_start_col = wi;
            group_n += row_ed - wi + alpha;
        }
        else
            group_n += row_ed - row_st + alpha;

        last_end_row = row + 1;
        last_end_col = row_ed;
    }
    // if (last_start_row < last_end_row && last_end_row < m)
    // {
    warp_info tmp = warp_info(last_start_row, last_end_row, last_start_col, last_end_col);
    host_winfo.push_back(tmp);
    warp_load.push_back(group_n);
    //}

    // for (int i = 0; i < host_winfo.size(); i++)
    // {
    //     if (host_winfo.size() < 50)
    //         printf("%d row %d row_ed %d col %d col_ed %d\n", i,
    //         host_winfo[i].row_st, host_winfo[i].row_ed, host_winfo[i].col_st, host_winfo[i].col_ed);
    // }

    // printf("total count %d\n", host_winfo.size());

    *winfo_n = host_winfo.size();
    cudaMalloc(winfo, host_winfo.size() * sizeof(warp_info));
    cudaMemcpy(*winfo, &host_winfo[0], host_winfo.size() * sizeof(warp_info), cudaMemcpyHostToDevice);

    int warp_min_load = -1;
    int warp_max_load = 0;

    for (int i = 0; i < warp_load.size(); i++)
    {
        int tmp = warp_load[i];

        if (warp_min_load == -1 || tmp < warp_min_load) warp_min_load = tmp;
        if (warp_max_load < tmp) warp_max_load = tmp;

        //printf("%d ", tmp);
    }

    //printf("warp load %d %d\n", warp_min_load, warp_max_load);
}

void sinfo2device(std::vector<warp_info> *host_sinfo, warp_info ***sinfo, int **sinfo_n)
{
    int host_sinfo_n[WARP_NUM];
    for (int i = 0; i < WARP_NUM; i++)
        host_sinfo_n[i] = host_sinfo[i].size();
    
    cudaMalloc(sinfo_n, WARP_NUM * sizeof(int));
    cudaMemcpy(*sinfo_n, host_sinfo_n, WARP_NUM * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(sinfo, WARP_NUM * sizeof(warp_info*));

    warp_info *info_h[WARP_NUM];
    for (int i = 0; i < WARP_NUM; i++)
    {
        int level = host_sinfo_n[i];
        if (level)
        {
            // printf("host wid %d level %d\n", i, level);
            // printf("host info st %d ed %d\n", host_sinfo[i][0].row_st, host_sinfo[i][0].row_ed);
            cudaMalloc(&info_h[i], level * sizeof(warp_info));
            cudaMemcpy(info_h[i], &host_sinfo[i][0], level * sizeof(warp_info), cudaMemcpyHostToDevice);
        }
        else
            info_h[i] = NULL;
    }

    cudaMemcpy(*sinfo, info_h, WARP_NUM * sizeof(warp_info*), cudaMemcpyHostToDevice);
}

// Persistent scheduling
void kg_csr_balance3(int m, int nnz, int *RowPtr, int wsize, int alpha, warp_info ***sinfo, int **sinfo_n)
{
    std::vector<warp_info> host_sinfo[WARP_NUM];
    std::vector<int> warp_load;

    int group_n = 0;
    int last_start_row = 0;
    int last_end_row = -1;
    int last_start_col, last_end_col;

    int current_warp = 0;
    int current_block = 0;
    int warp_round = 0;
    int switch_round = 1;

    for (int row = 0; row < m; row++)
    {
        int row_st = RowPtr[row];
        int row_ed = RowPtr[row + 1];

        // Approximation of a write back
        // int alpha = 2;

        if (row_ed - row_st + alpha > wsize - group_n || last_end_row == -1)
        {
            if (last_end_row != -1)
            {
                warp_info tmp = warp_info(last_start_row, last_end_row, last_start_col, last_end_col);
                //host_sinfo[current_warp].push_back(tmp);
                //current_warp = (current_warp + 1) % WARP_NUM;
                host_sinfo[current_block * WARP_PER_BLOCK + current_warp].push_back(tmp);
                current_warp++;
                if (current_warp == WARP_PER_BLOCK)
                {
                    warp_round++;
                    current_warp = 0;
                    if (warp_round == switch_round)
                    {
                        warp_round = 0;
                        current_block = (current_block + 1) % BLOCK_NUM;
                    }
                }
                group_n = 0;
            }

            int wi;
            for (wi = row_st; wi < row_ed - wsize + alpha; wi += wsize - alpha)
            {
                warp_info tmp = warp_info(row, row + 1, wi, wi + wsize - alpha);
                // host_sinfo[current_warp].push_back(tmp);
                // current_warp = (current_warp + 1) % WARP_NUM;
                host_sinfo[current_block * WARP_PER_BLOCK + current_warp].push_back(tmp);
                current_warp++;
                if (current_warp == WARP_PER_BLOCK)
                {
                    warp_round++;
                    current_warp = 0;
                    if (warp_round == switch_round)
                    {
                        warp_round = 0;
                        current_block = (current_block + 1) % BLOCK_NUM;
                    }
                }
            }
            last_start_row = row;
            last_start_col = wi;
            group_n += row_ed - wi + alpha;
        }
        else
            group_n += row_ed - row_st + alpha;

        last_end_row = row + 1;
        last_end_col = row_ed;
    }

    warp_info tmp = warp_info(last_start_row, last_end_row, last_start_col, last_end_col);
    //host_sinfo[current_warp].push_back(tmp);
    host_sinfo[current_block * WARP_PER_BLOCK + current_warp].push_back(tmp);

    sinfo2device(host_sinfo, sinfo, sinfo_n);
}

// Persistent scheduling (To maximize L1 hit rate)
void kg_csr_balance4(int m, int nnz, int *RowPtr, int wsize, warp_info ***sinfo, int **sinfo_n)
{
    std::vector<warp_info> host_sinfo[WARP_NUM];
    std::vector<warp_info> task_list;

    int group_n = 0;
    int last_start_row = 0;
    int last_end_row = -1;
    int last_start_col, last_end_col;

    int total_workload = 0;

    // Approximation of a write back
    int alpha = 2;

    for (int row = 0; row < m; row++)
    {
        int row_st = RowPtr[row];
        int row_ed = RowPtr[row + 1];

        if (row_ed - row_st + alpha > wsize - group_n || last_end_row == -1)
        {
            if (last_end_row != -1)
            {
                warp_info tmp = warp_info(last_start_row, last_end_row, last_start_col, last_end_col);
                task_list.push_back(tmp);
                total_workload += last_end_col - last_start_col + (last_end_row - last_start_row) * alpha;
                group_n = 0;
            }

            int wi;
            for (wi = row_st; wi < row_ed - wsize + alpha; wi += wsize - alpha)
            {
                warp_info tmp = warp_info(row, row + 1, wi, wi + wsize - alpha);
                task_list.push_back(tmp);
                total_workload += wsize - alpha + alpha;
            }
            last_start_row = row;
            last_start_col = wi;
            group_n += row_ed - wi + alpha;
        }
        else
        {
            group_n += row_ed - row_st + alpha;
        }

        last_end_row = row + 1;
        last_end_col = row_ed;
    }

    warp_info tmp = warp_info(last_start_row, last_end_row, last_start_col, last_end_col);
    task_list.push_back(tmp);
    total_workload += last_end_col - last_start_col + (last_end_row - last_start_row) * alpha;

    // float workload_per_block = 1.0 * total_workload / BLOCK_NUM;
    // int list_idx = 0;
    // int list_size = task_list.size();

    // //printf("total_workload %d size %d\n", total_workload, list_size);
    // for (int block = 0; block < BLOCK_NUM; block++)
    // {
    //     int current_warp = 0;
    //     int current_load = 0;
    //     int start_index = block * WARP_PER_BLOCK;
    //     while (list_idx < list_size && current_load < workload_per_block)
    //     {
    //         warp_info tmp = task_list[list_idx];
    //         current_load += tmp.col_ed - tmp.col_st + (tmp.row_ed - tmp.row_st) * alpha;
    //         host_sinfo[start_index + current_warp].push_back(tmp);
    //         current_warp = (current_warp + 1) % WARP_PER_BLOCK;
    //         list_idx++;
    //     }
    // }

    //printf("final idx %d\n", list_idx);

    int current_warp[BLOCK_NUM];
    for (int i = 0; i < BLOCK_NUM; i++) current_warp[i] = 0;

    int list_size_total = task_list.size();
    int L2_size = 7680;

    for (int sub_list = 0; sub_list < list_size_total; sub_list+=L2_size)
    {
        int list_size = kg_min(list_size_total - sub_list, L2_size);
        
        float workload_per_block = 0.0;
        for (int i = sub_list; i < sub_list + list_size; i++)
        {
            warp_info tmp = task_list[i];
            workload_per_block += tmp.col_ed - tmp.col_st + (tmp.row_ed - tmp.row_st) * alpha;
        }
        workload_per_block /= BLOCK_NUM;

        int list_idx = 0;
        for (int block = 0; block < BLOCK_NUM; block++)
        {
            //int current_warp = 0;
            int current_load = 0;
            int start_index = block * WARP_PER_BLOCK;
            while (list_idx < list_size && current_load < workload_per_block)
            {
                warp_info tmp = task_list[sub_list + list_idx];
                current_load += tmp.col_ed - tmp.col_st + (tmp.row_ed - tmp.row_st) * alpha;
                host_sinfo[start_index + current_warp[block]].push_back(tmp);
                current_warp[block] = (current_warp[block] + 1) % WARP_PER_BLOCK;
                list_idx++;
            }
        }
    }

    sinfo2device(host_sinfo, sinfo, sinfo_n);
}

void kg_csr_schedule_locality(int m, int nnz, int *RowPtr, int *ColIdx,
int bin_size, warp_info ***sinfo, int **sinfo_n)
{
    std::vector<warp_info> host_sinfo[WARP_NUM];

    int bins = (m + bin_size - 1) / bin_size;
    int bin_load[bins];
    std::vector<warp_info> bin_list[bins];

    int subrow_num = 0;

    for (int i = 0; i < bins; i++)
    {
        bin_load[i] = 0;
    }
    int bin_nnz = 0;
    int bin_load_total = 0;

    int alpha = 10;

    for (int row = 0; row < m; row++)
    {
        int row_st = RowPtr[row];
        int row_ed = RowPtr[row + 1];

        int bin_id = 0;
        // int bin_st = 0;
        // int bin_ed = bin_size;

        int ptr_st = row_st;
        int ptr = ptr_st;
        while (ptr < row_ed)
        {
            //int bin_st = bin_id * bin_size;
            int bin_ed = kg_min((bin_id + 1) * bin_size, m);
            while (ptr < row_ed && ColIdx[ptr] < bin_ed)
            {
                ptr++;
            }

            if (ptr > ptr_st + 5)
            {
                warp_info tmp = warp_info(row, row + 1, ptr_st, ptr);
                bin_list[bin_id].push_back(tmp);
                bin_load_total += ptr - ptr_st + alpha;
                bin_nnz += ptr - ptr_st;
                bin_load[bin_id] += ptr - ptr_st + alpha;
                subrow_num++;
            }

            bin_id++;

            ptr_st = ptr;
        }
    }

    printf("bin %d total %d subrow_num %d\n", bin_nnz, nnz, subrow_num);

    int min_bin = -1;
    int max_bin = 0;
    for (int i = 0; i < bins; i++)
    {
        if (bin_load[i] > max_bin) max_bin = bin_load[i];
        if (min_bin == -1 || bin_load[i] < min_bin) min_bin = bin_load[i];
    }
    float workload_per_block = 1.0 * bin_load_total / bins;
    printf("bin workload: bins %d max %d min %d\n", bins, max_bin, min_bin);

    int current_block = 0;
    int warp_pos[BLOCK_NUM];
    int total_count = 0;
    int block_load[BLOCK_NUM];
    //int warp_load[BLOCK_NUM][WARP_PER_BLOCK];
    
    for (int i = 0; i < BLOCK_NUM; i++)
    {
        warp_pos[i] = 0;
        block_load[i] = 0;
        // for (int j = 0; j < WARP_PER_BLOCK; j++)
        //     warp_load[i][j] = 0;
    }

    int current_warp = 0;
    int tasks = 0;
    int current_load = 0;
    int current_warp_load = 0;

    int bin_block = bins;
    for (int i = 0; i < bins; i += bin_block)
    {
        int bin_end = kg_min(bins, i + bin_block);
        workload_per_block = 0;
        for (int j = i; j < bin_end; j++)
        {
            workload_per_block += bin_load[j];
            total_count += bin_list[j].size();
        }
        workload_per_block = 1.0 * workload_per_block / BLOCK_NUM;

        for (int ii = i; ii < bin_end; ii++)
        {
            for (int j = 0; j < bin_list[ii].size(); j++)
            {
                host_sinfo[WARP_PER_BLOCK * current_block + current_warp].push_back(bin_list[ii][j]);
                block_load[current_block] += bin_list[ii][j].col_ed - bin_list[ii][j].col_st + alpha;

                current_warp_load += bin_list[ii][j].col_ed - bin_list[ii][j].col_st + alpha;
                if (current_warp_load >= 1024)
                {
                    current_warp_load = 0;
                    current_warp = (current_warp + 1) % WARP_PER_BLOCK;
                }

                tasks++;
                current_load += bin_list[ii][j].col_ed - bin_list[ii][j].col_st + alpha;
                //if (tasks == 1024)
                if (current_load >= workload_per_block)
                {
                    warp_pos[current_block] = current_warp;
                    current_block = (current_block + 1) % BLOCK_NUM;
                    current_warp = warp_pos[current_block];
                    tasks = 0;
                    current_load = 0;
                }
            }
        }
    }

    // for (int i = 0; i < bins; i++)
    // {
    //     total_count += bin_list[i].size();

    //     //if (i <= 20) printf("bin %d %d\n", i, bin_list[i].size());

    //     for (int j = 0; j < bin_list[i].size(); j++)
    //     {
    //         host_sinfo[WARP_PER_BLOCK * current_block + current_warp].push_back(bin_list[i][j]);
    //         block_load[current_block] += bin_list[i][j].col_ed - bin_list[i][j].col_st + alpha;

    //         current_warp_load += bin_list[i][j].col_ed - bin_list[i][j].col_st + alpha;
    //         if (current_warp_load >= 1024)
    //         {
    //             current_warp_load = 0;
    //             current_warp = (current_warp + 1) % WARP_PER_BLOCK;
    //         }

    //         tasks++;
    //         current_load += bin_list[i][j].col_ed - bin_list[i][j].col_st + alpha;
    //         //if (tasks == 1024)
    //         if (current_load >= workload_per_block)
    //         {
    //             warp_pos[current_block] = current_warp;
    //             current_block = (current_block + 1) % BLOCK_NUM;
    //             current_warp = warp_pos[current_block];
    //             tasks = 0;
    //             current_load = 0;
    //         }
    //     }
    // }

    // for (int i = 0; i < bins; i++)
    // {
    //     total_count += bin_list[i].size();

    //     current_block = 0;
    //     for (int j = 1; j < BLOCK_NUM; j++)
    //         if (block_load[j] < block_load[current_block])
    //             current_block = j;

    //     int current_warp = warp_pos[current_block];
    //     for (int j = 0; j < bin_list[i].size(); j++)
    //     {
    //         host_sinfo[WARP_PER_BLOCK * current_block + current_warp].push_back(bin_list[i][j]);
    //         current_warp = (current_warp + 1) % WARP_PER_BLOCK;

    //         block_load[current_block] += bin_list[i][j].col_ed - bin_list[i][j].col_st;
    //     }
    //     warp_pos[current_block] = current_warp;
    // }

    int warp_min_load = -1;
    int warp_max_load = 0;

    for (int i = 0; i < BLOCK_NUM; i++)
    {
        int tmp = 0;
        for (int j = 0; j < host_sinfo[i].size(); j++)
        {
            tmp += host_sinfo[i][j].col_ed - host_sinfo[i][j].col_st + alpha;
        }
        if (warp_min_load == -1 || tmp < warp_min_load) warp_min_load = tmp;
        if (warp_max_load < tmp) warp_max_load = tmp;
    }

    printf("warp load %d %d\n", warp_min_load, warp_max_load);

    // for (int i = 0; i < BLOCK_NUM; i++)
    // {
    //     printf("i %d load %d\n", i, block_load[i]);
    // }

    // printf("total count %d\n", total_count);

    sinfo2device(host_sinfo, sinfo, sinfo_n);
}

void kg_csr_schedule_locality2(int m, int nnz, int *RowPtr, int *ColIdx,
int wsize, warp_info ***sinfo, int **sinfo_n)
{
    std::vector<warp_info> host_sinfo[WARP_NUM];

    int bin_size = m / 1024;

    int bins = (m + bin_size - 1) / bin_size;
    //int bin_load[bins];
    std::vector<warp_info> bin_list[bins];

    int group_n = 0;
    int last_start_row = 0;
    int last_end_row = -1;
    int last_start_col, last_end_col;

    int bin_id;

// #define FIND_BIN_ID(a, b) ((ColIdx[(a)] + ColIdx[(b) - 1]) / 2 / bin_size)
#define FIND_BIN_ID(a, b) (ColIdx[(a)] / bin_size)

    for (int row = 0; row < m; row++)
    {
        int row_st = RowPtr[row];
        int row_ed = RowPtr[row + 1];

        if (row_ed - row_st > wsize - group_n || last_end_row == -1)
        {
            if (last_end_row != -1)
            {
                warp_info tmp = warp_info(last_start_row, last_end_row, last_start_col, last_end_col);
                bin_id = FIND_BIN_ID(last_start_col, last_end_col);
                bin_list[bin_id].push_back(tmp);
                group_n = 0;
            }

            int wi;
            for (wi = row_st; wi < row_ed - wsize; wi += wsize)
            {
                warp_info tmp = warp_info(row, row + 1, wi, wi + wsize);
                bin_id = FIND_BIN_ID(wi, wi + wsize);
                bin_list[bin_id].push_back(tmp);
            }
            last_start_row = row;
            last_start_col = wi;
            group_n += row_ed - wi;
        }
        else
            group_n += row_ed - row_st;

        last_end_row = row + 1;
        last_end_col = row_ed;
    }
    warp_info tmp = warp_info(last_start_row, last_end_row, last_start_col, last_end_col);
    bin_id = FIND_BIN_ID(last_start_col, last_end_col);
    bin_list[bin_id].push_back(tmp);

    int total_count = 0;

    int current_block = 0;
    int warp_pos[BLOCK_NUM];
    
    for (int i = 0; i < BLOCK_NUM; i++)
    {
        warp_pos[i] = 0;
    }

    for (int i = 0; i < bins; i++)
    {
        total_count += bin_list[i].size();

        int current_warp = warp_pos[current_block];
        for (int j = 0; j < bin_list[i].size(); j++)
        {
            host_sinfo[WARP_PER_BLOCK * current_block + current_warp].push_back(bin_list[i][j]);
            current_warp = (current_warp + 1) % WARP_PER_BLOCK;
        }
        warp_pos[current_block] = current_warp;
        current_block = (current_block + 1) % BLOCK_NUM;
    }

    printf("total count %d\n", total_count);

    sinfo2device(host_sinfo, sinfo, sinfo_n);

#undef FIND_BIN_ID

}

void kg_csr_block(int m, int nnz, int *RowPtr, block_info **binfo, int *binfo_n)
{
    std::vector<block_info> binfo_vec[BLOCK_NUM];

    int block_size = (m + BLOCK_NUM - 1) / BLOCK_NUM;
    int current_block = 0;
    int max_len = 0;
    for (int i = 0; i < m; i+=block_size)
    {
        block_info tmp = block_info(i, kg_min(i + block_size, m));

        binfo_vec[current_block].push_back(tmp);
        if (binfo_vec[current_block].size() > max_len)
            max_len = binfo_vec[current_block].size();

        current_block = (current_block + 1) % BLOCK_NUM;
    }

    *binfo_n = max_len;

    block_info binfo_h[BLOCK_NUM * max_len];

    for (int i = 0; i < BLOCK_NUM; i++)
    {
        int st_idx = i * max_len;
        int j;
        for (j = 0; j < binfo_vec[i].size(); j++)
        {
            binfo_h[st_idx + j] = binfo_vec[i][j];
            //printf("block_num %d[%d] row_st %d row_ed %d\n", i, j, binfo_h[st_idx + j].row_st, binfo_h[st_idx + j].row_ed);
        }
        for (; j < max_len; j++)
            binfo_h[st_idx + j] = block_info(-1, -1);
    }

    cudaMalloc(binfo, BLOCK_NUM * max_len * sizeof(block_info));
    //memcpy(*binfo, binfo_h, BLOCK_NUM * max_len * sizeof(block_info));
    cudaMemcpy(*binfo, binfo_h, BLOCK_NUM * max_len * sizeof(block_info), cudaMemcpyHostToDevice);
}

void kg_finalize_cu(ana_info* ana)
{

    if (ana->winfo)
    {
        CUDA_CHECK_ERROR(cudaFree(ana->winfo));
    }
    

    if (ana->sinfo)
    {
        warp_info* host_sinfo_n[WARP_NUM];
        CUDA_CHECK_ERROR(cudaMemcpy(host_sinfo_n, ana->sinfo_n, WARP_NUM * sizeof(warp_info*), cudaMemcpyDeviceToHost));

        for (int i = 0; i < WARP_NUM; i++)
            CUDA_CHECK_ERROR(cudaFree(host_sinfo_n[i]));

        CUDA_CHECK_ERROR(cudaFree(ana->sinfo));
        CUDA_CHECK_ERROR(cudaFree(ana->sinfo_n));
    }

    if (ana->binfo)
    {
        CUDA_CHECK_ERROR(cudaFree(ana->binfo));
    }

    if (ana->bp)
    {
        CUDA_CHECK_ERROR(cudaFree(ana->bp->PckPtr));
        CUDA_CHECK_ERROR(cudaFree(ana->bp->PckCont));
        CUDA_CHECK_ERROR(cudaFree(ana->bp->RowPtr_sp));
        CUDA_CHECK_ERROR(cudaFree(ana->bp->ColIdx_sp));
        
        if (ana->bpinfo)
        {
            bin_pack_info *host_sinfo_n[WARP_NUM];
            CUDA_CHECK_ERROR(cudaMemcpy(host_sinfo_n, ana->bpinfo, WARP_NUM * sizeof(bin_pack_info*), cudaMemcpyDeviceToHost));

            for (int i = 0; i < WARP_NUM; i++)
            {
                //if (i >= 10) continue;
                if (host_sinfo_n[i])
                    CUDA_CHECK_ERROR(cudaFree(host_sinfo_n[i]));
            }

            CUDA_CHECK_ERROR(cudaFree(ana->bpinfo));
            CUDA_CHECK_ERROR(cudaFree(ana->bpinfo_n));
        }

        if (ana->bpinfo2)
        {
            CUDA_CHECK_ERROR(cudaFree(ana->bpinfo2));
        }

        if (ana->spinfo)
            CUDA_CHECK_ERROR(cudaFree(ana->spinfo));
    
        delete ana->bp;
    }
}