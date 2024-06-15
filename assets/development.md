# help

## installing c++

```bash
sudo apt update -y
sudo apt install g++-10 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 60
```

## Installing ecCodes

```bash
sudo apt install gfortran libaec-dev cmake -y
wget https://confluence.ecmwf.int/download/attachments/45757960/eccodes-2.35.0-Source.tar.gz
tar -xzf  eccodes-2.35.0-Source.tar.gz
mkdir build ; cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local/ -DENABLE_PNG=ON ../eccodes-2.35.0-Source
make
make install
echo export ECCODES_DIR=\$HOME/.local >> $HOME/.zshrc
echo export ECCODES_DIR=\$HOME/.local >> $HOME/.bashrc
```