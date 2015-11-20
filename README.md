## AWS Setup (Ubuntu)

Setting up a CPU instance

```bash
# Install Git LFS
sudo apt-get update
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install -y git-lfs
git-lfs install
 
sudo apt-get install -y build-essential git python-pip python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot unzip python-numpy swig python-pandas python-sklearn htop
sudo pip install -U pip
sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade pandas

git clone https://github.com/dennybritz/sentiment-analysis.git
cd sentiment-analysis/char-cnn
./train.py

```

```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev  gfortran python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot linux-headers-generic linux-image-extra-virtual unzip python-numpy swig python-pandas python-sklearn unzip
sudo pip install -U pip

# Install CUDA 7
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1410/x86_64/cuda-repo-ubuntu1410_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1410_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda

# Install cuDNN
# You get the CUDNN_URL by logging into your nivida account and downloading cuDNN
wget $CUDNN_URL -O cudnn-6.5-linux-x64-v2.tgz
tar -zxf cudnn-6.5-linux-x64-v2.tgz
cd cudnn-6.5-linux-x64-v2
sudo cp -R lib* /usr/local/cuda/lib64/
sudo cp cudnn.h /usr/local/cuda/include/

# Reboot for CUDA
sudo reboot

export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=$CUDA_HOME
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

# Install Bazel
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install -y oracle-java8-installer
wget https://github.com/bazelbuild/bazel/releases/download/0.1.1/bazel-0.1.1-installer-linux-x86_64.sh
chmod a+x bazel-0.1.1-installer-linux-x86_64.sh
./bazel-0.1.1-installer-linux-x86_64.sh

# Clone Tensorflow
git clone --recurse-submodules https://github.com/tensorflow/tensorflow
cd tensorflow

# Build with GPU support, use 3.0 as CUDA version
TF_UNOFFICIAL_SETTING=1 ./configure
bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer

# Build TF python package
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# Install python packages
sudo pip install --upgrade /tmp/tensorflow_pkg/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```


```bash
# Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install -y git-lfs
git-lfs install

cd /home/ubuntu
git clone https://github.com/dennybritz/sentiment-analysis.git
cd sentiment-analysis/char-cnn
./train.py
```


Docker:

```bash
# Run docker container
export CUDA_SO=$(\ls /usr/lib/x86_64-linux-gnu/libcuda* | xargs -I{} echo '-v {}:{}')
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
export CUDA_SRCS="-v /usr/local/cuda:/usr/local/cuda -v /usr/share/nvidia:/usr/share/nvidia"
docker run --rm -it $CUDA_SO $CUDA_SRCS $DEVICES -v /lib/modules:/lib/modules b.gcr.io/tensorflow/tensorflow-full-gpu
```


Python Example

```python
import tensorflow as tf
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print sess.run(c)
```