// use ndarray::prelude::*;
use ndarray as nd;
use ndarray::ShapeBuilder; // for .f()
use std;

fn demo_basic00() {
    println!("\n# draft_ndarray_basic.rs/demo_basic00");
    let x0 = nd::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    assert_eq!(x0.ndim(), 2);
    assert_eq!(x0.len(), 6);
    assert_eq!(x0.shape(), [2, 3]);
    assert_eq!(x0.is_empty(), false);

    println!("x0: {:?}", x0);
    // x0: [[1.0, 2.0, 3.0],
    //   [4.0, 5.0, 6.0]], shape=[2, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2

    // With the trait `ShapeBuilder` in scope, there is the method `.f()` to select
    // column major (“f” order) memory layout instead of the default row major.
    // "nd::Ix3" means 3-dimensional
    let x1 = nd::Array::<f64, nd::Ix3>::zeros((3, 2, 4).f());
    println!("x1: {:?}", x1);

    let x2 = nd::Array::<bool, nd::Ix2>::from_elem((2, 3), false);
    println!("x2: {:?}", x2);

    // range logspace ones eye
    let x3 = nd::Array::<f64, _>::linspace(0.0, 1.0, 5);
    println!("x3: {:?}", x3);
}

fn demo_operation() {
    println!("\n# draft_ndarray_basic.rs/demo_operation");
    let x0 = nd::array![[10., 20., 30., 40.]];
    let x1 = nd::Array::range(0., 4., 1.); //[0., 1., 2., 3.,]
    assert_eq!(&x0 + &x1, nd::array![[10., 21., 32., 43.,]]); // Allocates a new array. Note the explicit `&`.
    assert_eq!(&x0 - &x1, nd::array![[10., 19., 28., 37.,]]);
    assert_eq!(&x0 * &x1, nd::array![[0., 20., 60., 120.,]]);
    assert_eq!(
        &x0 / &x1,
        nd::array![[std::f64::INFINITY, 20., 15., 13.333333333333334,]]
    );

    let x2 = nd::arr2(&[[1., 2., 3.], [4., 5., 6.]]);
    assert!(x2.sum_axis(nd::Axis(0)) == nd::aview1(&[5., 7., 9.]));
    assert!(x2.sum_axis(nd::Axis(1)) == nd::aview1(&[6., 15.]));
    assert!(x2.sum_axis(nd::Axis(0)).sum_axis(nd::Axis(0)) == nd::aview0(&21.));
    assert!(x2.sum_axis(nd::Axis(0)).sum_axis(nd::Axis(0)) == nd::aview0(&x2.sum()));

    let x0 = nd::array![[1., 2., 3., 4.,]];
    let x1 = nd::Array::range(0., 4., 1.);
    let x1_reshape = x1.into_shape((4, 1)).unwrap();
    println!("x0 @ x1: {:?}", x0.dot(&x1_reshape));
    println!("x0.T @ x1.T:\n{:?}", x0.t().dot(&(x1_reshape.t())));
}

fn demo_indexing() {
    println!("\n# draft_ndarray_basic.rs/demo_indexing");
    // equivalent numpy operation
    // np0 = np.arange(0, 7, 1, dtype=np.float64)
    // np1 = np0 ** 3
    // print(np1[2])
    // print(np1[2:5])
    // np1[:6:2] = 1000
    // print(np1**(1/3))
    let x0 = nd::Array::range(0., 7., 1.);
    let mut x1 = x0.mapv(|x: f64| x.powi(3)); // numpy equivlant of `a ** 3`; https://doc.rust-lang.org/nightly/std/primitive.f64.html#method.powi
    println!("x[[2]]: {}", x1[[2]]);
    println!("{}", x1.slice(nd::s![2]));
    println!("{}", x1.slice(nd::s![2..5]));
    x1.slice_mut(nd::s![..6;2]).fill(1000.); // numpy equivlant of `a[:6:2] = 1000`
    println!("after .slice_mut(): {}", x1);
    for i in x1.iter() {
        print!("{}, ", i.powf(1. / 3.))
    }
    println!();


    // a 3D array  2 x 2 x 3
    let x2 = nd::array![[[  0,  1,  2], [ 10, 12, 13]],
                    [[100,101,102], [110,112,113]]
                ];
    let x3 = x2.mapv(|x: isize| x.pow(1));  // numpy equivlant of `a ** 2`;
    println!("x -> \n{}\n", x3);
    println!("`a.slice(s![1, .., ..])` -> \n{}\n", x3.slice(nd::s![1, .., ..]));
    println!("`a.slice(s![.., .., 2])` -> \n{}\n", x3.slice(nd::s![.., .., 2]));
    println!("`a.slice(s![.., 1, 0..2])` -> \n{}\n", x3.slice(nd::s![.., 1, 0..2]));
    println!("`a.iter()` ->");
    for i in x3.iter() {
        print!("{}, ", i)  // flat out to every element
    }
    println!("\n\n`a.outer_iter()` ->");
    for i in x3.outer_iter() {
        print!("row: {}, \n", i)  // iterate through first dimension
    }
}


fn main() {
    demo_basic00();
    demo_operation();
    demo_indexing();
}
