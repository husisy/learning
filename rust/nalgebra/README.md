# nalgebra

1. link
   * [nalgebra/github](https://github.com/dimforge/nalgebra)
   * [nalgebra/documentation](https://nalgebra.org/docs/)
2. usage
   * `nalgebra = "0.32"`
   * `using nalgebra as na`
3. dtype
   * `na::SMatrix<T,R,C>`: `na::SMatrix<f32,2,3>`, `na::Vector6<T>`
   * `na::SVector<T,D>:`: `na::SVector<f32,23>`, `na::Matrix6<T>`, `na::Matrix4x6<T>`
     * `na::SVector3::x()`, `na::Svector6::a()`
   * `na::DMatrix<T>`
   * `na::DVector<T>`
   * `na::Matrix<T,R,C,S>`: the buffer `S`
   * `na::Const<2>`: `na::U127`
4. memory
   * `na::OMatrix`: own data
   * view of data
   * statically-size matrix: on heap, column-major
   * dynamically-size matrix: on stack, column-major
