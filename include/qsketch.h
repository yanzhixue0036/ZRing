#include <ctime>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <chrono>
#include "MurmurHash3.h"
#include "PackedVector.hpp"
#include "DatasetManager.h"

std::chrono::high_resolution_clock::time_point qs_begin_update, qs_end_update;
std::chrono::high_resolution_clock::time_point qs_begin_estimate, qs_end_estimate;

inline int argmin(PackedVector array, int m) 
{
    int idx = 0, minv = array.get(0);
    for (int i = 1; i < m; i++) {
        if (array.get(i) < minv) {
            minv = array.get(i);
            idx = i;
        }
    }
    return idx;
}

double InitialValue(PackedVector sketch, int m)
{
    double c0 = 0.0;
    double tmp_sum = 0.0;
 
    for(int i=0; i<m; i++) { 
        tmp_sum += pow(2, -sketch.get(i)); 
    }

    c0 = (double)(m-1) / tmp_sum;
    return c0;
}

inline double ffunc(PackedVector sketch, uint32_t k, double w) 
{
    double res = 0;
    double e = 2.718282;
    for (int i = 0; i < k; ++i) {
        double x = pow(2.0, -sketch.get(i) - 1);
        double ex = pow(e, w * x);
        res += x * (2.0 - ex) / (ex - 1.0);
    }
    return res;
}

//f（w）
inline double dffunc(PackedVector sketch, uint32_t k, double w) 
{
    double res = 0;
    double e = 2.718282;
    for (int i = 0; i < k; ++i) {
        double x = pow(2.0, -sketch.get(i) - 1);
        double ex = pow(e, w * x);
        res += -x * x * ex * pow(ex - 1, -2);
    }
    return res;
}

double Newton(PackedVector sketch, uint32_t k, double c0) 
{
    double err = 1e-5;
    double c1 = c0 - ffunc(sketch, k, c0) / dffunc(sketch, k, c0);

    while (abs (c1 - c0) > err) {
        c0 = c1;
        c1 = c0 - (double)ffunc(sketch, k, c0) / dffunc(sketch, k, c0);
    }
    return c1;
}

class QSketch
{
    public:

        QSketch(int sketch_size, int register_size);

        void Update(const DatasetManager& dataset, std::vector<PackedVector> &g, int i, float ratio=1.0, bool flag_insert=true);
        void EstimateCard();
        double hash(int data_id, int counter_id);

        int r_max;
        int r_min;
        int offset;
        int sketch_size;
        int register_size;
        int range;

        uint32_t *pi;
        uint32_t *pii;
        uint32_t *seed;
        double estimated_card;
        double actual;
        double update_time;
        double estimation_time;

        PackedVector qs;
        std::vector<std::vector<int>> hit_cnt;

        double Count();
        double Actual();
        double Estimated_card();
};

QSketch::QSketch(int sketch_size, int register_size) : qs(register_size, sketch_size)
{
    std::random_device rd;
    this->register_size = register_size;
    this->sketch_size = sketch_size / 16;
    this->range = pow(2, register_size);
    this->r_max = this->range - 1;
    this->r_min = 0;
    this->pii = new uint32_t[sketch_size];
    this->seed = new uint32_t[sketch_size];
    
    
    hit_cnt.resize(this->sketch_size);
    for (int i = 0; i < this->sketch_size; i++) {
        hit_cnt[i].assign(this->range, 0);
    }
    

    for (int i = 0; i < sketch_size; i++) {
        this->pii[i] = i;
        this->seed[i] = rd();
    }
    this->update_time = 0.0;
    this->estimation_time = 0.0;

}


inline double QSketch::hash(int data_id, int counter_id)
{
    double hash_value = 0.0;
    uint32_t hash_result;
    std::string key = std::to_string(data_id) + "|" + std::to_string(counter_id);
    MurmurHash3_x86_32(&key, sizeof(key), this->seed[counter_id], &hash_result);

    hash_value = (double)hash_result / (double)UINT32_MAX;
    return hash_value;
}


void QSketch::Update(const DatasetManager& dataset,
                     std::vector<PackedVector> &g,
                     int i, float ratio, bool flag_insert)
{
    if(flag_insert && ratio != 1.0) i++;
    
    std::vector<double> data;
    std::vector<int> indices;
    if(flag_insert){
       data = dataset.insert_data(ratio);
       indices = dataset.insert_indices(i);
    } else{
       data = dataset.delete_data(ratio);
       indices = dataset.delete_indices(ratio);
    }
    int data_num = data.size();


    qs_begin_update = std::chrono::high_resolution_clock::now();

    int j_min = 0, jj = 0;
    double r = 0.0, u = 0.0;
    int32_t y = 0;
    double insert = (flag_insert == true)? 1.0: -1.0;

    for (int t = 0; t < data_num; t++) {
        if (data[t] == 0.0) continue;

        this->actual += insert*data[t];
        this->pi = this->pii;
        r = 0.0;

        for (int i = 0; i < this->sketch_size; i++) {

            u = QSketch::hash(indices[t], i);
            r = r - log(u) / (data[t] * (this->sketch_size - i));
            y = floor(-log2(r));

            // if (y <= qs.get(j_min)) break;

            jj = rand() % (this->sketch_size - i) + i;
            std::swap(this->pi[i], this->pi[jj]);

            int index_y = y - this->r_min;
            if(index_y < 0) index_y = 0;

            if (flag_insert) {
                hit_cnt[i][index_y]++;

                if (y > int32_t(qs.get(this->pi[i]))) {

                    if (this->r_min < y && y < this->r_max) {
                        qs.set(this->pi[i], y);
                    } else if (y >= this->r_max) {
                        qs.set(this->pi[i], this->r_max);
                    } else {
                        continue;
                    }

                    if (this->pi[i] == j_min) {
                        j_min = argmin(qs, this->sketch_size);
                    }
                }
            } else {
                hit_cnt[i][index_y]--;
                if (hit_cnt[i][index_y] == 0 && y == qs.get(this->pi[i])) {

                    int new_max = this->r_min;
                    for (int yy = y - 1; yy >= this->r_min; --yy) {
                        if (hit_cnt[i][yy - this->r_min] > 0) {
                            new_max = yy;
                            break;
                        }
                    }
                    qs.set(this->pi[i], new_max);
                }
            }
        }
    }

    qs_end_update = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time1 = qs_end_update - qs_begin_update;
    this->update_time = time1.count();
}

 
void QSketch::EstimateCard()
{
    qs_begin_estimate = std::chrono::high_resolution_clock::now();
    double c0 = InitialValue(qs, this->sketch_size);
    this->estimated_card = Newton(qs, this->sketch_size, c0);

    qs_end_estimate = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time2 = qs_end_estimate - qs_begin_estimate;
    this->estimation_time = time2.count();

}





double QSketch::Count()
{   
    this->EstimateCard();
    double AARE = (double) fabs(fabs(this->estimated_card) - this->actual) / this->actual;
    return AARE;
}

double QSketch::Actual()
{
    return this->actual;
}

double QSketch::Estimated_card()
{
    return this->estimated_card;
}
