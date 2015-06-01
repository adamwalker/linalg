#include <cstdlib>

/*
 * Matrix multiplication
 * Computes out = x * y
 * x   is x_rows    * inner_dim
 * y   is inner_dim * y_cols
 * out is x_rows    * y_cols
 *
 * out may not alias x or y
 */
template <class T>
void mmult(size_t x_rows, size_t inner_dim, size_t y_cols, T *x, T *y, T *out){
    size_t i, j, k;
    for(i=0; i<x_rows; i++){
        for(j=0; j<y_cols; j++){
            T result = get_zero<T>();
            for(k=0; k<inner_dim; k++){
                result += *(x + i*inner_dim + k) * *(y + k*y_cols + j);
            }
            *(out + i*y_cols + j) = result;
        }
    }
}

/*
 * Matrix multiplication with second argument transposed
 * Computes out = x * transpose(yT)
 * x   is x_rows * inner_dim
 * yT  is y_cols * inner_dim
 * out is x_rows * y_cols
 *
 * out may not alias x or yT
 */
template <class T>
void mmult_yt(size_t x_rows, size_t inner_dim, size_t y_cols, T *x, T *yT, T *out){
    size_t i, j, k;
    for(i=0; i<x_rows; i++){
        for(j=0; j<y_cols; j++){
            T result = get_zero<T>();
            for(k=0; k<inner_dim; k++){
                result += *(x + i*inner_dim + k) * *(yT + j*inner_dim + k);
            }
            *(out + i*y_cols + j) = result;
        }
    }
}

/*
 * Matrix multiplication with first argument transposed
 * Computes out = transpose(xT) * y
 * xT  is inner_dim * x_rows 
 * y   is inner_dim * y_cols 
 * out is x_rows * y_cols
 *
 * out may not alias xT or y
 */
template <class T>
void mmult_xt(size_t x_rows, size_t inner_dim, size_t y_cols, T *xT, T *y, T *out){
    size_t i, j, k;
    for(i=0; i<x_rows; i++){
        for(j=0; j<y_cols; j++){
            T result = get_zero<T>();
            for(k=0; k<inner_dim; k++){
                result += *(xT + k*x_rows + i) * *(y + k*y_cols + j);
            }
            *(out + i*y_cols + j) = result;
        }
    }
}

/*
 * Matrix addition
 * Computes out = x + y;
 * x, y, out are rows * cols
 *
 * out may alias x or y or both
 */
template <class T>
void add(size_t rows, size_t cols, T *x, T *y, T *out){
    size_t i, j;
    for(i=0; i<rows; i++){
        for(j=0; j<rows; j++){
            *(out + i*cols + j) = *(x + i*cols + j) + *(y + i*cols + j);
        }
    }
}

/*
 * Matrix subtraction
 * Computes out = x - y;
 * x, y, out are rows * cols
 *
 * out may alias x or y or both
 */
template <class T>
void sub(size_t rows, size_t cols, T *x, T *y, T *out){
    size_t i, j;
    for(i=0; i<rows; i++){
        for(j=0; j<rows; j++){
            *(out + i*cols + j) = *(x + i*cols + j) + *(y + i*cols + j);
        }
    }
}

/*
 * Matrix multiplication by scalar
 * Computes out = scale * x;
 * x, out are rows * cols
 *
 * out may alias x
 */
template <class T>
void scale(T scale, size_t rows, size_t cols, T *x, T *out){
    size_t i, j;
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            *(out + i*cols + j) = scale * *(x + i*cols + j);
        }
    }
}

/*
 * Multiply vector by matrix on the right
 * Computes out = m * v
 * m   is rows * cols
 * v   is cols long
 * out is rows long
 *
 * out may not alias v
 */
template <class T>
void multv(size_t rows, size_t cols, T *m, T *v, T *out){
    size_t i, j;
    for(i=0; i<rows; i++){
        T result = get_zero<T>();
        for(j=0; j<cols; j++){
            result += *(m + i*cols + j) * v[j];
        }
        out[i] = result;
    }
}

template <class T>
void multv_t(size_t rows, size_t cols, T *m, T *v, T *out){
    size_t i, j;
    for(i=0; i<rows; i++){
        T result = get_zero<T>();
        for(j=0; j<cols; j++){
            result += *(m + j*rows + i) * v[j];
        }
        out[i] = result;
    }
}

/*
 * Vector multiplication by scalar
 * Computes out = scale * v;
 * x, out are rows long
 *
 * out may alias v
 */
template <class T>
void scalev(T scale, size_t rows, T *v, T *out){
    size_t i;
    for(i=0; i<rows; i++){
        out[i] = scale * v[i];
    }
}

/*
 * Vector division by scalar
 * Computes out = v / div;
 * x, out are rows long
 *
 * out may alias v
 */
template <class T>
void divv(T div, size_t rows, T *v, T *out){
    size_t i;
    for(i=0; i<rows; i++){
        out[i] = v[i] / div;
    }
}

/*
 * Compute the norm of a vector
 * v is rows long
 */
template <class T>
T norm(size_t rows, T *v){
    size_t i;
    T norm = get_zero<T>();
    for(i=0; i<rows; i++){
        norm += v[i] * v[i];
    }
    return sqrt(norm);
}

/*
 * Normalize a vector
 * v, out are rows long
 *
 * out may alias v
 */
template <class T>
void normalize(size_t rows, T *v, T *out){
    T n = norm(rows, v);
    divv(n, rows, v, out);
}

/*
 * Vector addition
 * Computes out = x + y;
 * x, y, out are rows long
 *
 * out may alias x or y
 */
template <class T>
void addv(size_t rows, T *x, T *y, T *out){
    size_t i;
    for(i=0; i<rows; i++){
        out[i] = x[i] + y[i];
    }
}

/*
 * Vector subtraction
 * Computes out = x - y;
 * x, y, out are rows long
 *
 * out may alias x or y
 */
template <class T>
void subv(size_t rows, T *x, T *y, T *out){
    size_t i;
    for(i=0; i<rows; i++){
        out[i] = x[i] - y[i];
    }
}

/*
 * QR decomposition
 * (q, r) = QRDecompose(q)
 * Updates q in place
 * q is rows * cols
 * r is rows * rows
 *
 * rows >= columns
 */
template <class T>
void qr(size_t rows, size_t cols, T *q, T *r){
    size_t i, j, k;
    //for each column of x
    for(j=0; j<cols; j++){
        //for each previous column
        for(i=0; i<j; i++){

            //compute the dot product
            T dot_prod = get_zero<T>();
            for(k=0; k<rows; k++){
                dot_prod += *(q + k*cols + i) * *(q + k*cols + j);
            }

            //subtract the projection
            for(k=0; k<rows; k++){
                *(q + k*cols + j) = *(q + k*cols + j) - *(q + k*cols + i) * dot_prod;
            }

            //set the entry in r
            *(r + i*cols + j) = dot_prod;
        }

        //compute the norm
        T norm = get_zero<T>();
        for(k=0; k<rows; k++){
            norm += *(q + k*cols + j) * *(q + k*cols + j);
        }
        norm = sqrt(norm);

        //set the entry in r
        *(r + j*cols + j) = norm;

        //divide by the norm
        for(k=0; k<rows; k++){
            *(q + k*cols + j) = *(q + k*cols + j) / norm;
        }
    }
}

/*
 * Backsubstitute
 * Solve for res: mat * res = vect
 * mat is rows * rows and upper triangular
 * res, vect have rows elements
 */
template <class T>
void backsubstitute(size_t rows, T *mat, T *vect, T *res){
    int i, j;
    for(i = rows - 1; i >= 0; i--){
        T to_subtract = get_zero<T>();
        for(j = rows - 1; j > i; j--){
            to_subtract += *(mat + i*rows + j) * res[j];
        }
        res[i] = (vect[i] - to_subtract) / *(mat + i*rows + i);
    }
}

/*
 * Find the least squares solution
 * Solve for res: q * r * res = v
 * q is rows * cols
 * r is cols * cols
 * res has cols elements
 * v has rows elements
 *
 * rows >= cols
 *
 * if rows == cols, this finds the exact solution
 */
template <class T>
void solve(size_t rows, size_t cols, T *q, T *r, T *v, T *res){
    T closest_proj[cols];
    multv_t(cols, rows, (T *)q, v, closest_proj);
    backsubstitute(cols, r, closest_proj, res);
}

/*
 * Backsubstitutem
 * Solve for res: mat * res = x
 * mat is rows * rows and upper triangular
 * x   is rows * cols
 * res is rows * cols
 */
template <class T>
void backsubstitutem(size_t rows, size_t cols, T *mat, T *x, T *res){
    int i, j, k;
    for(k=0; k<cols; k++){
        for(i = rows - 1; i >= 0; i--){
            T to_subtract = get_zero<T>();
            for(j = rows - 1; j > i; j--){
                to_subtract += *(mat + i*rows + j) * *(res + j*cols + k);
            }
            *(res + i*cols + k) = (*(x + i*cols + k) - to_subtract) / *(mat + i*rows + i);
        }
    }
}

/*
 * Find the least squares solution
 * Solve for res: q * r * res = m
 * q   is rows * cols
 * r   is cols * cols
 * res is cols * v_cols
 * m   is rows * v_cols
 *
 * rows >= cols
 *
 * if rows == cols, this finds the exact solution
 */
template <class T>
void solve(size_t rows, size_t cols, size_t v_cols, T *q, T *r, T *m, T *res){
    T closest_proj[cols][v_cols];
    mmult_xt(cols, rows, v_cols, q, m, (T *)closest_proj);
    backsubstitutem(cols, v_cols, r, (T *)closest_proj, res);
}

template <class T>
void transpose(size_t rows, size_t cols, T *x, T *xT){
    size_t i, j;
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            *(xT + j*rows + i) = *(x + i*cols + j);
        }
    }
}

template <class T>
void transpose_inplace(size_t rows, size_t cols, T *x){
    size_t i, j;
    for(i=0; i<rows; i++){
        for(j=0; j<i; j++){
            T temp = *(x + i*cols + j);
            *(x + i*cols + j) = *(x + j*rows + i);
            *(x + j*rows + i) = temp;
        }
    }
}

