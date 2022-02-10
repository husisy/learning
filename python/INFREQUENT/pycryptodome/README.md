# pycryptodome

1. link
   * [github](https://github.com/Legrandin/pycryptodome)
   * [doc](https://www.pycryptodome.org/en/latest/)
   * [阮一峰 - RSA算法原理一](http://www.ruanyifeng.com/blog/2013/06/rsa_algorithm_part_one.html)
   * [阮一峰 - RSA算法原理二](http://www.ruanyifeng.com/blog/2013/07/rsa_algorithm_part_two.html)
2. basic command
   * install: `conda install -n python_cpu -c conda-forge pycryptodomex`
   * `import Cryptodome`
   * unittest: `python -m Cryptodome.SelfTest`
3. `TDES` is obsolete
4. `RC4` is unsecure
5. 对称加密算法
   * 甲方选择某一种加密规则，对信息进行加密
   * 乙方使用同一种规则，对信息进行解密
6. 非对称加密
   * 乙方生成两把密钥（公钥和私钥）。公钥是公开的，任何人都可以获得，私钥则是保密的
   * 甲方获取乙方的公钥，然后用它对信息加密
   * 乙方得到加密后的信息，用私钥解密

## 数字签名Digital Signature

1. link
   * [What is a Digital Signature?](http://www.youdzone.com/signature.html)
