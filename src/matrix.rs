use rand::Rng;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "Matrix must have the same dimension"
        );

        let mut data = Vec::<f64>::with_capacity(self.rows * self.cols);
        for i in 0..self.data.len() {
            data.push(self.data[i] + other.data[i]);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "Matrix must have the same dimension"
        );

        let mut data = Vec::<f64>::with_capacity(self.rows * self.cols);
        for i in 0..self.data.len() {
            data.push(self.data[i] - other.data[i]);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn hadamard(&self, other: &Matrix) -> Matrix {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "Matrix must have the same dimension"
        );

        let mut data = Vec::<f64>::with_capacity(self.rows * self.cols);
        for i in 0..self.data.len() {
            data.push(self.data[i] * other.data[i]);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn mul(&self, scalar: f64) -> Matrix {
        let mut data = Vec::<f64>::with_capacity(self.rows * self.cols);
        for i in 0..self.data.len() {
            data.push(self.data[i] * scalar);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut data = Vec::<f64>::with_capacity(self.rows * self.cols);
        for i in 0..self.cols {
            for j in 0..self.rows {
                data.push(self.data[j * self.cols + i]);
            }
        }

        Matrix {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    pub fn from(data: Vec<Vec<f64>>) -> Matrix {
        let rows = data.len();
        let cols = data[0].len();

        let mut d = Vec::<f64>::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                d.push(data[i][j]);
            }
        }

        Matrix {
            rows,
            cols,
            data: d
        }
    }

    pub fn from_vec(data: Vec<f64>) -> Matrix {
        let rows = data.len();
        let cols = 1;

        Matrix {
            rows,
            cols,
            data
        }
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert!(
            self.cols == other.rows,
            "Attempt to multiply matrices with incorrect dimentions ({}x{} * {}x{})",
            self.rows, self.cols, other.rows, other.cols
        );

        
        let mut data = vec![0.0; self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                data[i * other.cols + j] = sum;
            }
        }

        Matrix {
            rows: self.rows,
            cols: other.cols,
            data,
        }
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let size = rows * cols;
        let mut data = Vec::<f64>::with_capacity(size);
        for _ in 0..size {
            data.push(rand::thread_rng().gen_range(0.0..1.0));
        }

        Matrix { rows, cols, data }
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.rows && j < self.cols, "Index out of bounds");
        self.data[i * self.cols + j]
    }

    pub fn map(&self, func: fn(f64) -> f64) -> Self {
        let size = self.rows * self.cols;
        let mut data = Vec::<f64>::with_capacity(size);
        for i in 0..size {
            data.push(func(self.data[i]));
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }
}



impl std::cmp::PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        for i in 0..self.data.len() {
            if (self.data[i] - other.data[i]).abs() > 1e-6 {
                return false;
            }
        }

        true
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut output = String::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                output.push_str(&self.data[i * self.cols + j].to_string());
                output.push_str(" ");
            }
            output.push_str("\n");
        }
        write!(f, "{}", output)
    }
}

#[macro_export]
macro_rules! matrix {
    ( $( $($val:expr),+ );* $(;)? ) => {
        {
            let mut data = Vec::<f64>::new();
            let mut rows = 0;
            let mut cols = 0;
            $(
                let row_data = vec![$($val),+];
                data.extend(row_data);
                rows += 1;
                let row_len = vec![$($val),+].len();
                if cols == 0 {
                    cols = row_len;
                } else if cols != row_len {
                    panic!("Inconsistent number of elements in the matrix rows");
                }
            )*

            Matrix { rows, cols, data }
        }
    };
}
