# tensorflow example

该文件夹的每一个文件应当是可独立运行的，展示了tensorflow不同层次的api使用示例，不用

1. `tf_api1_model_fit.py`：使用`tf.keras.Model`或者`tf.keras.models.Sequential`搭建神经网络，训练与预测直接使用类提供的接口
2. `tf_api2_model_function.py`：使用`tf.keras.Model`搭建神经网络，手动控制训练与预测过程
3. `tf_api3_model_variable.py`：使用`tf.Variable`控制每个参数变量的生成，使用`tf.keras.Model`搭建神经网络。较繁琐，故只使用了一个参数变量
4. `tf_api3_model_complex_variable.py`：complex number神经网络。由于虚数参数无法求导（loss为实数且能求导的纯函数只能是常数函数），故将虚数参数拆分为实部与虚部单独求导（文献中对虚数各种所谓“合理”的骚操作并没有带来显著的提升）
