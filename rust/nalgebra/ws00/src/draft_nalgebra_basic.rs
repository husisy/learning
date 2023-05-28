use nalgebra as na;

fn demo_basic00() {
    // type Mat_2_3_f32 = na::SMatrix<f32, 2, 3>; //statically sized (na,f32,(2,3))
    // type Mat_2_x_f64 = na::OMatrix<f64, na::U2, na::Dynamic>; //dynamically sized (na,f64,(2,?))
    // type Mat_x_x_i32 = na::OMatrix<i32, na::Dynamic, na::Dynamic>; //dynamically sized (na,i32,(?,?))

    // let x0 = na::Vector3::new(1, 2, 3); //??? int or float
    // let x1 = na::Matrix3x4::new(11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34);

    let x2a = na::Matrix2x3::new(1.1, 1.2, 1.3, 2.1, 2.2, 2.3);
    let x2b = na::Matrix2x3::from_rows(&[
        na::RowVector3::new(1.1, 1.2, 1.3),
        na::RowVector3::new(2.1, 2.2, 2.3),
    ]);
    let x2c = na::Matrix2x3::from_columns(&[
        na::Vector2::new(1.1, 2.1),
        na::Vector2::new(1.2, 2.2),
        na::Vector2::new(1.3, 2.3),
    ]);
    let x2d = na::Matrix2x3::from_row_slice(&[1.1, 1.2, 1.3, 2.1, 2.2, 2.3]);
    let x2e = na::Matrix2x3::from_column_slice(&[1.1, 2.1, 1.2, 2.2, 1.3, 2.3]);
    let x2f = na::Matrix2x3::from_fn(|r, c| (r + 1) as f32 + (c + 1) as f32 / 10.0);
    let x2g = na::Matrix2x3::from_iterator([1.1f32, 2.1, 1.2, 2.2, 1.3, 2.3].iter().cloned());
    assert_eq!(x2a, x2b);
    assert_eq!(x2a, x2c);
    assert_eq!(x2a, x2d);
    assert_eq!(x2a, x2e);
    assert_eq!(x2a, x2f);
    assert_eq!(x2a, x2g);

    let x3a = na::DMatrix::from_row_slice(
        4,
        3,
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    );
    let x3b = na::DMatrix::from_row_slice(
        4,
        3,
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    );
    let x3c = na::DMatrix::from_vec(
        4,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    );
    let x3d = na::DMatrix::from_diagonal_element(4, 3, 1.0);
    let x3e = na::DMatrix::identity(4, 3);
    let x3f = na::DMatrix::from_fn(4, 3, |r, c| if r == c { 1.0 } else { 0.0 });
    let x3g = na::DMatrix::from_iterator(
        4,
        3,
        [
            // Components listed column-by-column.
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]
        .iter()
        .cloned(),
    );
    assert_eq!(x3a, x3b);
    assert_eq!(x3a, x3c);
    assert_eq!(x3a, x3d);
    assert_eq!(x3a, x3e);
    assert_eq!(x3a, x3f);
    assert_eq!(x3a, x3g);

    assert_eq!(na::Vector3::x(), na::Vector3::new(1.0, 0.0, 0.0));
    assert_eq!(na::Vector3::y(), na::Vector3::new(0.0, 1.0, 0.0));
    assert_eq!(na::Vector3::z(), na::Vector3::new(0.0, 0.0, 1.0));

    // ???
    assert_eq!(
        na::Vector6::a(),
        na::Vector6::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    );
    assert_eq!(
        na::Vector6::b(),
        na::Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    );
    // assert_eq!(na::Vector6::c(), na::Vector6::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)); //???

    // fail to build
    // assert_eq!(na::Vector4::x_axis().unwrap(), na::Vector4::x());
    // assert_eq!(na::Vector4::y_axis().unwrap(), na::Vector4::y());
    // assert_eq!(na::Vector4::z_axis().unwrap(), na::Vector4::z());
    // assert_eq!(na::Vector5::a_axis().unwrap(), na::Vector5::a());
    // assert_eq!(na::Vector5::b_axis().unwrap(), na::Vector5::b());
}

pub fn demo_all() {
    demo_basic00();
}
