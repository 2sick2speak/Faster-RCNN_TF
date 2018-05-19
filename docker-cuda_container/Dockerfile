FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get update \
    && apt-get install -y openssh-server \
    && apt-get clean \
    && (rm -f /var/cache/apt/archives/*.deb \
      /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin || true)

RUN apt-get install -y python3-pip python-dev gfortran locales libglib2.0 libopencv-*
RUN apt-get install -y net-tools mc nano git tmux htop

RUN apt-get install -y libcupti-dev
ENV CUDA_HOME=/usr/local/cuda-8.0
ENV NVIDIA_HOME=/usr/local/nvidia
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NVIDIA_HOME}/lib64:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
ENV PATH=${PATH}:${CUDA_HOME}/bin:/usr/local/nvidia/bin
COPY cuda.sh /etc/profile.d/cuda.sh

RUN locale-gen "en_US.UTF-8"

RUN pip3 install -U pip
RUN pip3 install jupyter
RUN pip3 install numpy pandas scipy xgboost scikit-learn matplotlib
RUN pip3 install Pillow scikit-image seaborn opencv-python
RUN pip3 install cython numba h5py joblib tqdm
RUN pip3 install tensorflow-gpu==1.2.1 theano keras

RUN mkdir /var/run/sshd
RUN mkdir /root/.ssh
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22
COPY current_key /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys
RUN update-rc.d ssh defaults
ENTRYPOINT service ssh start && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
