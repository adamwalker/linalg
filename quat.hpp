#include <cstdlib>

template <class T>
void mul_q(T x[4], T y[4], T out[4]){
    out[0] = x[0] * y[0] - x[1] * y[1] - x[2] * y[2] - x[3] * y[3];
    out[1] = x[0] * y[1] + x[1] * y[0] + x[2] * y[3] - x[3] * y[2];
    out[2] = x[0] * y[2] - x[1] * y[3] + x[2] * y[0] + x[3] * y[1];
    out[3] = x[0] * y[3] + x[1] * y[2] - x[2] * y[1] + x[3] * y[0];
}

template <class T>
void mul_q_2nd_real_zero(T x[4], T y[3], T out[4]){
    out[0] =             - x[1] * y[0] - x[2] * y[1] - x[3] * y[2];
    out[1] = x[0] * y[0]               + x[2] * y[2] - x[3] * y[1];
    out[2] = x[0] * y[1] - x[1] * y[2]               + x[3] * y[0];
    out[3] = x[0] * y[2] + x[1] * y[1] - x[2] * y[0]              ;
}

template <class T>
void mul_q_1st_real_zero(T x[3], T y[4], T out[4]){
    out[0] = - x[0] * y[1] - x[3] * y[2] - x[3] * y[3];
    out[1] = + x[0] * y[0] + x[3] * y[3] - x[3] * y[2];
    out[2] = - x[0] * y[3] + x[3] * y[0] + x[3] * y[1];
    out[3] = + x[0] * y[2] - x[3] * y[1] + x[3] * y[0];
}

template <class T>
void mul_q_1st_real_zero_2nd_conj(T x[3], T y[4], T out[4]){
    out[0] = + x[0] * y[1] + x[3] * y[2] + x[3] * y[3];
    out[1] = + x[0] * y[0] - x[3] * y[3] + x[3] * y[2];
    out[2] = + x[0] * y[3] + x[3] * y[0] - x[3] * y[1];
    out[3] = - x[0] * y[2] + x[3] * y[1] + x[3] * y[0];
}

template <class T>
void mul_q_no_1st_out(T x[4], T y[4], T out[3]){
    out[0] = x[0] * y[1] + x[1] * y[0] + x[2] * y[3] - x[3] * y[2];
    out[1] = x[0] * y[2] - x[1] * y[3] + x[2] * y[0] + x[3] * y[1];
    out[2] = x[0] * y[3] + x[1] * y[2] - x[2] * y[1] + x[3] * y[0];
}

template <class T>
void rotate(T q[4], T v[3], T out[3]){
    T q_conj[4] = {q[0], -q[1], -q[2], -q[3]};
    T res[4];
    mul_q_2nd_real_zero(q, v, res);
    mul_q_no_1st_out(res, q_conj, out);
}

template <class T>
void axis_to_quat(T scale, T *axis, T *out){
    T nm = norm(3, axis);
    T res[3];
    divv(nm, 3, axis, res);
    nm = nm * scale;
    out[0] = cos(nm / 2);
    int i;
    for(i=0; i<3; i++){
        out[i+1] = res[i] * sin(nm / 2);
    }
}

