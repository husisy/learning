apt-get update
apt-get install -y gcc-7 g++-7 automake wget make
cd /root
wget -O openmpi-4.0.4.tar.gz https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.4.tar.gz
gunzip -c openmpi-4.0.4.tar.gz | tar xf -
cd openmpi-4.0.4
./configure --prefix=/usr/local
make all install
rm -rf /var/lib/apt/lists/*
rm /root/openmpi-4.0.4.tar.gz
rm -r /root/openmpi-4.0.4
