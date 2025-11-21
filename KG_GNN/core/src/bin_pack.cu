#include "../include/KG_GNN.h"
#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <iostream>

void sinfo2device(std::vector<bin_pack_info> *host_sinfo, bin_pack_info ***sinfo, int **sinfo_n)
{
    int host_sinfo_n[WARP_NUM];
    for (int i = 0; i < WARP_NUM; i++)
        host_sinfo_n[i] = host_sinfo[i].size();
    
    cudaMalloc(sinfo_n, WARP_NUM * sizeof(int));
    cudaMemcpy(*sinfo_n, host_sinfo_n, WARP_NUM * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(sinfo, WARP_NUM * sizeof(bin_pack_info*));

    bin_pack_info *info_h[WARP_NUM];
    for (int i = 0; i < WARP_NUM; i++)
    {
        int level = host_sinfo_n[i];
        if (level)
        {
            // if (i <= 10)
            // {
            //     printf("host wid %d level %d\n", i, level);
            //     printf("host info st %d ed %d\n", host_sinfo[i][0].bp_st, host_sinfo[i][0].bp_ed);
            // }
            cudaMalloc(&info_h[i], level * sizeof(bin_pack_info));
            cudaMemcpy(info_h[i], &host_sinfo[i][0], level * sizeof(bin_pack_info), cudaMemcpyHostToDevice);
        }
        else
            info_h[i] = NULL;
    }

    cudaMemcpy(*sinfo, info_h, WARP_NUM * sizeof(bin_pack_info*), cudaMemcpyHostToDevice);
}

void bin_pack_construct(int m, int nnz, int *RowPtr, int *ColIdx, 
int bin_size, int pack_size, int bin_thresh, bin_pack **bp, int alpha)
{
    *bp = new bin_pack();

    (*bp)->PckPtr_h.push_back(0);
    (*bp)->RowPtr_sp_h.push_back(0);
    (*bp)->BinPtr_h.push_back(0);

    int bins = (m + bin_size - 1) / bin_size;

    std::vector<warp_info> bin_list[bins];

    int bin_nnz = 0;

    int max_pck_size = 128;

    for (int row = 0; row < m; row++)
    {
        int row_st = RowPtr[row];
        int row_ed = RowPtr[row + 1];

        int bin_id = 0;

        int ptr_st = row_st;
        int ptr = ptr_st;
        while (ptr < row_ed)
        {
            //int bin_st = bin_id * bin_size;
            int bin_ed = kg_min((bin_id + 1) * bin_size, m);

            //printf("m %d bin_st %d bin_ed %d\n", row, bin_st, bin_ed);
            
            while (ptr < row_ed && ColIdx[ptr] < bin_ed)
            {
                ptr++;

                // handling extreme long rows
                if (ptr > ptr_st + max_pck_size)
                {
                    warp_info tmp = warp_info(row, row + 1, ptr_st, ptr);
                    bin_list[bin_id].push_back(tmp);

                    bin_nnz += ptr - ptr_st;

                    ptr_st = ptr;
                }
            }

            //printf("ptr_st %d ptr_ed %d ColIdx[ptr] = %d\n", ptr_st, ptr, ColIdx[ptr]);

            if (ptr > ptr_st + bin_thresh)
            {
                warp_info tmp = warp_info(row, row + 1, ptr_st, ptr);
                bin_list[bin_id].push_back(tmp);

                bin_nnz += ptr - ptr_st;
                // (*bp)->PckCont_h.push_back(- row - 1);
                // for (int col = ptr_st; col < ptr; col++)
                // {
                //     (*bp)->PckCont_h.push_back(ColIdx[col]);
                // }
                // (*bp)->PckPtr_h.push_back(PckCont_h.size());
            }
            else
            {
                for (int col = ptr_st; col < ptr; col++)
                {
                    (*bp)->ColIdx_sp_h.push_back(ColIdx[col]);
                }
            }

            bin_id++;

            ptr_st = ptr;
        }

        (*bp)->RowPtr_sp_h.push_back((*bp)->ColIdx_sp_h.size());
    }

    int current_pck_size = 0;

    for (int bin_id = 0; bin_id < bins; bin_id++)
    {
        //printf("bin_id %d size %d\n", bin_id, bin_list[bin_id].size());
        int current_bin_load = 0;

        //int bin_st = bin_id * bin_size;
        //int bin_ed = kg_min((bin_id + 1) * bin_size - 1, m);

        for (int i = 0; i < bin_list[bin_id].size(); i++)
        {
            warp_info tmp = bin_list[bin_id][i];

            (*bp)->PckCont_h.push_back(tmp.row_st);
            //(*bp)->PckCont_h.push_back(tmp.col_ed - tmp.col_st);
            for (int col = tmp.col_st; col < tmp.col_ed; col++)
            {
                (*bp)->PckCont_h.push_back(ColIdx[col]);
            }

            // package strategy
            current_pck_size += tmp.col_ed - tmp.col_st + alpha;
            current_bin_load += tmp.col_ed - tmp.col_st + alpha;

            (*bp)->PckPtr_h.push_back((*bp)->PckCont_h.size());
            //(*bp)->PckLoad.push_back(current_pck_size);

            // if (current_pck_size > pack_size)
            // {
            //     (*bp)->PckPtr_h.push_back((*bp)->PckCont_h.size());
            //     (*bp)->PckLoad.push_back(current_pck_size);
            //     current_pck_size = 0;
            // }
        }

        // if (current_pck_size > 0)
        // {
        //     (*bp)->PckPtr_h.push_back((*bp)->PckCont_h.size());
        //     (*bp)->PckLoad.push_back(current_pck_size);
        //     current_pck_size = 0;
        // }

        (*bp)->BinPtr_h.push_back((*bp)->PckPtr_h.size() - 1);
        (*bp)->BinLoad.push_back(current_bin_load);
    }

    if ((*bp)->PckPtr_h.back() != (*bp)->PckCont_h.size())
        (*bp)->PckPtr_h.push_back((*bp)->PckCont_h.size());

    //(*bp)->BinPtr_h.push_back((*bp)->PckCont_h.size());

    (*bp)->Pckn = (*bp)->PckPtr_h.size() - 1;
    (*bp)->spn = (*bp)->RowPtr_sp_h.size() - 1;

    //printf("bin nnz %d total nnz %d rest nnz %d\n", bin_nnz, nnz, (*bp)->RowPtr_sp_h[(*bp)->spn]);

    int *PckPtr = NULL;
    int *PckCont = NULL;
    int *RowPtr_sp = NULL;
    int *ColIdx_sp = NULL;
    
    if ((*bp)->PckPtr_h.size() > 1)
    {
        cudaMalloc(&PckPtr, (*bp)->PckPtr_h.size() * sizeof(int));
        cudaMalloc(&PckCont, (*bp)->PckCont_h.size() * sizeof(int));

        cudaMemcpy(PckPtr, &((*bp)->PckPtr_h[0]), (*bp)->PckPtr_h.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(PckCont, &((*bp)->PckCont_h[0]), (*bp)->PckCont_h.size() * sizeof(int), cudaMemcpyHostToDevice);
    }

    if ((*bp)->spn > 0)
    {
        cudaMalloc(&RowPtr_sp, (*bp)->RowPtr_sp_h.size() * sizeof(int));
        cudaMalloc(&ColIdx_sp, (*bp)->ColIdx_sp_h.size() * sizeof(int));

        cudaMemcpy(RowPtr_sp, &((*bp)->RowPtr_sp_h[0]), (*bp)->RowPtr_sp_h.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(ColIdx_sp, &((*bp)->ColIdx_sp_h[0]), (*bp)->ColIdx_sp_h.size() * sizeof(int), cudaMemcpyHostToDevice);
    }

    (*bp)->bin_pack_cpy(PckPtr, PckCont, RowPtr_sp, ColIdx_sp);

    // printf("BinPtr\n");
    // for(int i = 0; i < (*bp)->BinPtr_h.size(); i++) {
    //     std::cout << (*bp)->BinPtr_h[i] << " ";
    // }
    // std::cout << endl;
    // printf("PckPtr\n");
    // for(int i = 0; i < (*bp)->PckPtr_h.size(); i++) {
    //     std::cout << (*bp)->PckPtr_h[i] << " ";
    // }
    // std::cout << endl;
    // printf("PckCont\n");
    // for(int i = 0; i < (*bp)->PckCont_h.size(); i++) {
    //     std::cout << (*bp)->PckCont_h[i] << " ";
    // }
    // std::cout << endl;
    // printf("BinLoad\n");
    // for(int i = 0; i < (*bp)->BinLoad.size(); i++) {
    //     std::cout << (*bp)->BinLoad[i] << " ";
    // }
    // std::cout << endl;
}

void bin_pack_schedule(bin_pack *bp, bin_pack_info ***bpinfo, int **bpinfo_n, int bin_block, int alpha)
{
    std::vector<bin_pack_info> host_bpinfo[WARP_NUM];

    int current_warp = 0;
    // int current_block = 0;
    // int warp_pos[BLOCK_NUM];
    // int block_load[BLOCK_NUM];

    // for (int i = 0; i < BLOCK_NUM; i++)
    // {
    //     warp_pos[i] = 0;
    //     block_load[i] = 0;
    // }

    int warp_load[WARP_NUM];
    for (int i = 0; i < WARP_NUM; i++)
        warp_load[i] = 0;

    int bins = bp->BinPtr_h.size() - 1;

    // printf("bin_pack bins %d pcks %d\n", bp->BinPtr_h.size() - 1, bp->PckPtr_h.size() - 1);

    // for (int i = 0; i < 1; i++)
    // {
    //     printf("BinPtr_st ed %d %d\n", bp->BinPtr_h[i], bp->BinPtr_h[i + 1]);
    //     int tmp_load = 0;
    //     for (int j = bp->BinPtr_h[i]; j < bp->BinPtr_h[i+1]; j++)
    //         tmp_load += bp->PckLoad[j];
    //     printf("checkout: %d %d %d\n", i, tmp_load, bp->BinLoad[i]);
    // }

    float workload_per_warp = 0;

    for (int i = 0; i < bins; i += bin_block)
    {
        int bin_end = kg_min(bins, i + bin_block);

        // based on workload
        float workload_per_block = 0;
        for (int j = i; j < bin_end; j++)
        {
            workload_per_block += bp->BinLoad[j];
        }
        workload_per_block = 1.0 * workload_per_block / BLOCK_NUM;
        workload_per_warp += workload_per_block / WARP_PER_BLOCK;

        int bp_st = bp->BinPtr_h[i];
        int bp_ed = bp_st;

        //int current_warp_load = 0;
        //int current_block_load = 0;

        //current_block = 0;

        while (bp_ed < bp->BinPtr_h[bin_end])
        {
            int tmp_load = bp->PckPtr_h[bp_ed + 1] - bp->PckPtr_h[bp_ed] + alpha;
            //current_warp_load += tmp_load;
            warp_load[current_warp] += tmp_load;
            //if (current_warp_load > workload_per_warp)
            if (warp_load[current_warp] > workload_per_warp)
            {
                bin_pack_info tmp = bin_pack_info(bp_st, bp_ed + 1);
                host_bpinfo[current_warp].push_back(tmp);

                current_warp = (current_warp + 1) % WARP_NUM;
                //if (current_warp == 0) current_block = (current_block + 1) % BLOCK_NUM;
                //current_warp_load = 0;
                bp_st = bp_ed + 1;
            }

            bp_ed++;
        }

        if (bp_st != bp->BinPtr_h[bin_end])
        {
            bin_pack_info tmp = bin_pack_info(bp_st, bp->BinPtr_h[bin_end]);
            host_bpinfo[current_warp].push_back(tmp);
            // printf("bin_end %d, bp->BinPtr_h[bin_end] = %d\n", bin_end, bp->BinPtr_h[bin_end]);
        }

    }

    int warp_min_load = -1;
    int warp_max_load = 0;

    // printf("workload: ");

    for (int i = 0; i < WARP_NUM; i++)
    {
        int tmp = 0;
        //tmp = warp_load[i];
        for (int j = 0; j < host_bpinfo[i].size(); j++)
        {
            for (int jj = host_bpinfo[i][j].bp_st; jj < host_bpinfo[i][j].bp_ed; jj++)
                tmp += bp->PckPtr_h[jj + 1] - bp->PckPtr_h[jj] + alpha;
            //bp->PckLoad[jj];//bp->PckPtr_h[jj + 1] - bp->PckPtr_h[jj] + alpha;
            //tmp += host_bpinfo[i][j].bp_ed - host_bpinfo[i][j].bp_st + alpha;
        }
        if (warp_min_load == -1 || tmp < warp_min_load) warp_min_load = tmp;
        if (warp_max_load < tmp) warp_max_load = tmp;
        //printf("%d ", tmp);
    }

    printf("min/max warp load %d %d\n", warp_min_load, warp_max_load);

    sinfo2device(host_bpinfo, bpinfo, bpinfo_n);
    
}

void bin_pack_schedule2(bin_pack *bp, bin_pack_info **bpinfo, int *bpinfo_n, int bin_block, int alpha)
{
    std::vector<bin_pack_info> host_bpinfo[WARP_NUM];

    int current_warp = 0;
    int current_block = 0;
    int warp_pos[BLOCK_NUM];
    int block_load[BLOCK_NUM];

    for (int i = 0; i < BLOCK_NUM; i++)
    {
        warp_pos[i] = 0;
        block_load[i] = 0;
    }

    int warp_load[WARP_NUM];
    for (int i = 0; i < WARP_NUM; i++)
        warp_load[i] = 0;

    int bins = bp->BinPtr_h.size() - 1;

    float workload_per_warp = 0;

    int current_warp_load;
    int current_block_load;

    for (int i = 0; i < bins; i += bin_block)
    {
        int bin_end = kg_min(bins, i + bin_block);

        // based on workload
        float workload_per_block = 0;
        int workload_per_block_test = 0;
        for (int j = i; j < bin_end; j++)
        {
            workload_per_block += bp->BinLoad[j];
        }

        //printf("workload %.3f\n", workload_per_block);

        workload_per_block = 1.0 * workload_per_block / BLOCK_NUM;
        workload_per_warp += workload_per_block / WARP_PER_BLOCK;

        int bp_st = bp->BinPtr_h[i];
        int bp_ed = bp_st;

        // if (i < 64)
        // {
        //     printf("bin %d~%d packages %d\n", i, bin_end, bp->BinPtr_h[bin_end] - bp->BinPtr_h[i]);
        //     printf("workload_per_warp %.3f\n", workload_per_warp);
        // }

        //current_warp_load = 0;
        //current_block_load = 0;

        current_block = 0;

        while (bp_ed < bp->BinPtr_h[bin_end])
        {
            int tmp_load = bp->PckPtr_h[bp_ed + 1] - bp->PckPtr_h[bp_ed] - 1 + alpha;
            //current_warp_load += tmp_load;
            //printf("debug %d\n", tmp_load);
            warp_load[current_warp] += tmp_load;
            workload_per_block_test += tmp_load;
            //if (current_warp_load > workload_per_warp)
            if (warp_load[current_warp] > workload_per_warp)
            {
                bin_pack_info tmp = bin_pack_info(bp_st, bp_ed + 1);
                host_bpinfo[current_warp].push_back(tmp);

                current_warp = (current_warp + 1) % WARP_NUM;
                //if (current_warp == 0) current_block = (current_block + 1) % BLOCK_NUM;
                //current_warp_load = 0;
                bp_st = bp_ed + 1;
            }

            bp_ed++;
        }

        if (bp_st < bp->BinPtr_h[bin_end])
        {
            bin_pack_info tmp = bin_pack_info(bp_st, bp->BinPtr_h[bin_end]);
            host_bpinfo[current_warp].push_back(tmp);
            workload_per_block_test += bp->PckPtr_h[bp->BinPtr_h[bin_end]] - bp->PckPtr_h[bp_st] + 
            (bp->BinPtr_h[bin_end] - bp_st) * (alpha - 1);
            // printf("bin_end %d, bp->BinPtr_h[bin_end] = %d\n", bin_end, bp->BinPtr_h[bin_end]);
        }

        // if (i < 64)
        // {
        //     for (int ii = 0; ii < 20; ii++)
        //         printf("warp %d load %d ", ii, int(warp_load[ii]));
        //     printf("\n");
        //     printf("test workload_per_block_test %d\n", workload_per_block_test);
        //     printf("level %d %d %d %d %d\n", host_bpinfo[0].size(),
        //     host_bpinfo[1].size(), host_bpinfo[2].size(),
        //     host_bpinfo[3].size(), host_bpinfo[4].size());
        // }

    }

    int max_level = 0;
    for (int i = 0; i < WARP_NUM; i++)
        if (host_bpinfo[i].size() > max_level) max_level = host_bpinfo[i].size();
    
    //printf("maxlevel %d\n", max_level);
    
    bin_pack_info bpinfo_tmp[max_level * WARP_NUM];

    for (int i = 0; i < WARP_NUM; i++)
    {
        for (int wave = 0; wave < host_bpinfo[i].size(); wave++)
        {
            bpinfo_tmp[i + WARP_NUM * wave] = host_bpinfo[i][wave];
            // if (i <= 20 && wave <= 1) printf("warp_id %d st %d ed %d\n", i + WARP_NUM * wave,
            // bpinfo_tmp[i + WARP_NUM * wave].bp_st, bpinfo_tmp[i + WARP_NUM * wave].bp_ed);
        }
        for (int wave = host_bpinfo[i].size(); wave < max_level; wave++)
        {
            bpinfo_tmp[i + WARP_NUM * wave] = bin_pack_info(-1, -1);
        }
    }

    cudaMalloc(bpinfo, sizeof(bin_pack_info) * WARP_NUM * max_level);
    cudaMemcpy(*bpinfo, bpinfo_tmp, sizeof(bin_pack_info) * WARP_NUM * max_level, cudaMemcpyHostToDevice);

    *bpinfo_n = max_level * WARP_NUM;
}