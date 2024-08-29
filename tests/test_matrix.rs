
#[cfg(test)]
mod test {
    use std::vec;

    use rand::Rng;

    use neural_network::matrix::Matrix;
    use neural_network::matrix;

    #[test]
    fn test_macro_matrix_square() {
        let matrix = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0;
        ];

        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data, vec![ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ])
    }

    #[test]
    fn test_macro_matrix_rectangle() {
        let matrix = matrix![
            1.0, 2.0, 3.0;
            7.0, 8.0, 9.0;
        ];

        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data, vec![ 1.0, 2.0, 3.0, 7.0, 8.0, 9.0 ])
    }

    #[test]
    #[should_panic(expected = "Inconsistent number of elements in the matrix rows")]
    fn test_macro_matrix_invalid() {
        let _ = matrix![
            1.0, 2.0, 3.0;
            7.0, 8.0
        ];
    }

    #[test]
    fn test_random_matrix() {
        let mut rng = rand::thread_rng();
        for _ in 0..5 {
            let rows: usize = rng.gen_range(3..12);
            let cols: usize = rng.gen_range(3..12);
            let matrix = Matrix::random(rows, cols);

            assert_eq!(matrix.rows, rows);
            assert_eq!(matrix.cols, cols);
            assert_eq!(matrix.data.len(), rows * cols);

            for v in matrix.data {
                assert!(v >= 0.0 && v < 1.0);
            }
        }
    }

    #[test]
    fn test_add_matrix_row() {
        let matrix_a = matrix![1.0, 2.0, 3.0];
        let matrix_b = matrix![9.0, 8.0, 7.0];
        let matrix_total = matrix_a.add(&matrix_b);
        assert_eq!(matrix_total.rows, matrix_a.rows);
        assert_eq!(matrix_total.cols, matrix_b.cols);
        assert_eq!(matrix_total.data, vec![10.0, 10.0, 10.0]);
    }

    #[test]
    fn test_add_matrix_multiple() {
        let matrix_a = matrix![11.0, 22.0, 33.0; 44.0, 55.0, 66.0];
        let matrix_b = matrix![9.0, 8.0, 7.0; 6.0, 5.0, 4.0];
        let matrix_total = matrix_a.add(&matrix_b);
        assert_eq!(matrix_total.rows, matrix_a.rows);
        assert_eq!(matrix_total.cols, matrix_b.cols);
        assert_eq!(matrix_total.data, vec![20.0, 30.0, 40.0, 50.0, 60.0, 70.0]);
    }

    #[test]
    #[should_panic(expected = "Matrix must have the same dimension")]
    fn test_add_matrix_invalid() {
        let matrix_a = matrix![1.0, 2.0, 3.0];
        let matrix_b = matrix![9.0, 8.0, 7.0; 6.0, 5.0, 4.0];
        let _ = matrix_a.add(&matrix_b);
    }

    #[test]
    fn test_sub_matrix_row() {
        let matrix_a = matrix![1.0, 2.0, 3.0];
        let matrix_b = matrix![9.0, 8.0, 7.0];
        let matrix_total = matrix_a.sub(&matrix_b);
        assert_eq!(matrix_total.rows, matrix_a.rows);
        assert_eq!(matrix_total.cols, matrix_b.cols);
        assert_eq!(matrix_total.data, vec![-8.0, -6.0, -4.0]);
    }

    #[test]
    #[should_panic(expected = "Matrix must have the same dimension")]
    fn test_sub_matrix_invalid() {
        let matrix_a = matrix![1.0, 2.0, 3.0];
        let matrix_b = matrix![9.0, 8.0, 7.0; 6.0, 5.0, 4.0];
        let _ = matrix_a.sub(&matrix_b);
    }

    #[test]
    fn test_dot_matrix() {
        let matrix_a = matrix![1.0, 2.0; 3.0, 4.0];
        let matrix_b = matrix![5.0, 6.0; 7.0, 8.0];
        let matrix_total = matrix_a.dot(&matrix_b);
        assert_eq!(matrix_total.rows, matrix_a.rows);
        assert_eq!(matrix_total.cols, matrix_b.cols);
        assert_eq!(matrix_total.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    #[should_panic(expected = "Attempt to multiply matrices with incorrect dimentions")]
    fn test_dot_matrix_invalid() {
        let matrix_a = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];
        let matrix_b = matrix![
            5.0, 6.0;
            7.0, 8.0;
            9.0, 10.0
        ];
        let _ = matrix_a.dot(&matrix_b);
    }

    #[test]
    fn test_map_matrix() {
        let matrix = matrix![1.0, 2.0; 3.0, 4.0];
        let matrix_total = matrix.map(|v| v * 2.0);
        assert_eq!(matrix_total.rows, matrix.rows);
        assert_eq!(matrix_total.cols, matrix.cols);
        assert_eq!(matrix_total.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_eq_matrix() {
        let matrix_a = matrix![1.0, 2.0; 3.0, 4.0];
        let matrix_b = matrix![1.0, 2.0; 3.0, 4.0];
        assert!(matrix_a == matrix_b);
    }

    #[test]
    fn test_ne_matrix() {
        let matrix_a = matrix![1.0, 2.0; 3.0, 4.0];
        let matrix_b = matrix![1.0, 2.0; 3.0, 5.0];
        assert!(matrix_a != matrix_b);
    }

    #[test]
    fn test_display_matrix() {
        let matrix = matrix![1.0, 2.0; 3.0, 4.0];
        let output = format!("{}", matrix);
        assert_eq!(output, "1 2 \n3 4 \n");
    }
}
