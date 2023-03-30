# open quantum safe

1. link
   * [github/liboqs](https://github.com/open-quantum-safe/liboqs/)
   * [github/PQClean](https://github.com/PQClean/PQClean)
   * [NIST/PQC](https://csrc.nist.gov/Projects/post-quantum-cryptography) Post Quantum Cryptography
   * [NIST/PQC-DSS](https://csrc.nist.gov/Projects/pqc-dig-sig) Post Quantum Cryptography: digital signature schemes
2. NIST selection
   * key encapsulation mechanism (KEM): Kyber
   * signature scheme: Dilithium, Falcon, SPHINCS+
3. language
   * core: c
   * wrapper: cpp, go, java, .net, python, rust
   * docker container provided

```bash
sudo apt install libssl-dev unzip xsltproc astyle
# gcc
conda create -n oqs
conda install -n oqs -c conda-forge cmake pytest pytest-xdist ninja doxygen pyyaml graphviz valgrind ipython python==3.10 scipy matplotlib
# 3.11 failed with full-homomorphic-encryption

git clone -b main https://github.com/open-quantum-safe/liboqs.git

mkdir build && cd build
cmake -LAH ..
cmake -GNinja -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=~/.local/lib/liboqs ..
ninja #lib/
ninja run_test #tests/
ninja gen_docs #docs/doxygen/html/index.html
ninja install

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/liboqs/lib

git clone git@github.com:open-quantum-safe/liboqs-python.git
```
