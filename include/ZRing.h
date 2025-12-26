#include <ctime>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include <unordered_map>

#include "MurmurHash3.h"
#include "PackedVector.hpp"
#include "DatasetManager.h"



class ZRing
{
private:
    
    
    
public:

    ZRing(int sketch_size, int register_size,int g_size);

    double hash(int data_id, uint32_t seed);
    int hash_G(int data_id, int row, uint32_t seed, int G);
    void Update(const DatasetManager& dataset, std::vector<PackedVector> &g, int round, float ratio=1.0, bool flag_insert=true);
    

    
    double EstimateColumnCardinality(std::vector<PackedVector> &g);
    double EstimateColumnCardinality_IVW(std::vector<PackedVector> &g);
    double EstimateCardinality_MLE(std::vector<PackedVector> &g);
    void EstimateCard(std::vector<PackedVector> &g);


    double Count();
    double Actual();
    double GetAARE();
    double Estimated_card();

    double actual;
    double estimated_card;
    std::unordered_map<double, std::pair<int, double>> item_stats;
    void UpdateItemStats(int value, bool flag_insert, double weight);
    

    int r_max;
    int r_min;
    int offset;
    int sketch_size;
    int range;
    int G_size;


    uint32_t* seed;
    std::mt19937 gen;
    std::uniform_int_distribution<> dist;
    
    double update_time;
    double estimation_time;
    
    std::vector<PackedVector> f;
    int* Zeros;

};


ZRing::ZRing(int sketch_size, int register_size,int g_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());                
    std::uniform_int_distribution<> dist(1, 100);  
    this->gen = gen;
    this->dist = dist;
    
    this->sketch_size = sketch_size / 3;        
    this->G_size = pow(2, g_size);          

    this->range = pow(2, register_size);    
    this->r_max = this->range - 1;
    this->r_min = 0;
    this->offset = pow(2, register_size - 1) - 1;

    this->seed = new uint32_t[sketch_size+1];
    for (int i = 0; i < sketch_size+1; i++) {
        this->seed[i] = rd(); // 
    }

    this->Zeros = new int[this->range];
    for (int i = 0; i < this->range; i++){
        this->Zeros[i] = this->sketch_size;
    }
    
    this->update_time = 0.0;
    this->estimation_time = 0.0;
    this->estimated_card = 0.0;
    this->actual = 0.0;

    this->f.reserve(sketch_size);
    for (int i = 0; i < sketch_size; ++i) {
        this->f.emplace_back(g_size, this->range);
    }
}


inline double ZRing::hash(int data_id, uint32_t seed)
{
    uint32_t hash_result;
    std::string key = std::to_string(data_id);
    MurmurHash3_x86_32(key.data(),key.size(), seed, &hash_result);
    
    return hash_result;
}

inline int ZRing::hash_G(int data_id, int row, uint32_t seed, int G)
{
    uint32_t hash_result;
    std::string key = std::to_string(row) + "|" + std::to_string(data_id);
    MurmurHash3_x86_32(key.data(),key.size(), seed, &hash_result);

    int hash_value = hash_result & (G - 1);
    return hash_value;
}


void ZRing::Update(const DatasetManager& dataset, std::vector<PackedVector> &g, int round, float ratio, bool flag_insert)
{
    if(flag_insert && ratio != 1.0) round++;
    
    std::vector<double> data;
    std::vector<int> indices;
    if(flag_insert){
       data = dataset.insert_data(ratio);
       indices = dataset.insert_indices(round);
    } else{
       data = dataset.delete_data(ratio);
       indices = dataset.delete_indices(ratio);
    }
    size_t data_size = data.size();
    double insert = (flag_insert == true)? 1.0: -1.0;

    std::chrono::high_resolution_clock::time_point qs_begin_update, qs_end_update;
    qs_begin_update = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < data_size; t++) {
        double value = data[t];
        int index = indices[t];

        if (value == 0.0) continue;
        this->actual += insert*value;
        
        uint32_t hv = ZRing::hash(index, this->seed[0]);
        int row_index = hv % this->sketch_size;
        double u = (double)hv / (double)UINT32_MAX;
        int y = floor(-log2(- log(u) / value));
        int column_index = std::min(std::max(y + this->offset, this->r_min), this->r_max);

        int pre = g[row_index].get(column_index);
        int add_value = ZRing::hash_G(index, round, this->seed[2], this->G_size);
        int res = (pre + (int)insert * add_value) & (this->G_size - 1);
        
        // continue;
        int flag_V = int((res == 0) - (pre == 0));
        g[row_index].set(column_index, res);
        if(flag_V == 0) continue;


        if((flag_V == 1 && flag_insert) || (flag_V == 1 && !flag_insert)) 
            this->Zeros[column_index] += flag_V;
        
        double f_add = 0.0, G = (double)this->G_size;
        for (int k = 0; k < this->range; k++){
            int j = k - offset;

            double add = 0.0;
            if(k==0) {add = exp(-data[t] * pow(2, -j-1));}
            else if (k == this->range-1) {add = 1 - exp(-data[t] * pow(2, -j));}
            else{add = exp(-data[t] * pow(2, -j - 1)) - exp(-data[t] * pow(2, -j));}
            
            f_add += (G - 1.0) / G * add * (double)this->Zeros[k];
        }
        if(flag_V == -1 && flag_insert)
            this->estimated_card += (data[t] * 1) / (f_add / this->sketch_size); 
        else if(flag_V == 1 && !flag_insert)
            this->estimated_card -= (data[t] * 1) / (f_add / this->sketch_size); 
        
        if(not((flag_V == 1 && flag_insert) || (flag_V == 1 && !flag_insert))) 
            this->Zeros[column_index] += flag_V;


        qs_end_update = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time1 = qs_end_update - qs_begin_update;
        this->update_time = time1.count();
    }
}



double ZRing::GetAARE()
{
    double AARE = (double) fabs(fabs(this->estimated_card) - this->actual) / this->actual;
    return AARE;
}

double ZRing::Actual()
{
    return this->actual;
}

double ZRing::Estimated_card()
{
    return this->estimated_card;
}




























