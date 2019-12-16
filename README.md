# objectmapping.py

This script uses the bounding box and boolean mask outputs of an object detection model in order to map the relationship between the objects as well as the position of the objects in an image. The needed data is extracted using the Mask R-CNN class provided by Matterport (https://github.com/matterport/Mask_RCNN) with the **mask_rcnn_coco.h5** weights (https://github.com/matterport/Mask_RCNN/releases). However, as long as a model outputs bounding boxes (h1, w1, h2, w2) and a boolean array for the mask data, then this script should work, though some re-shaping of the data may be necessary to match the expected input of the script. The script can be used to analyze a single image or to annotate relationship and position data for a batch of images. The **feature_extraction.ipynb** is one such example of how the script was used to annotate training data for batches of images.

## Application Use

The easiest way to start is to try out:

* object_relation_sample.ipynb
* location_grid_sample.ipynb


The script can be run directly from the command line assuming Python is installed. All development was done with python 3.7.4. (Please see the environment information to check further compatibility.) If working in a mixed python environment, specify python3 instead of python. Currently, you must give a path to an image file to start the program.<br><br>

**The following will output the image, masks, mask outlines, and summary information:**
<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```$ python objectmapping.py /path/to/image/file.jpg```
<br><br>
**The following can be used to experiment with other class methods:**
<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```$ python -i objectmapping.py /path/to/image/file.jpg```
<br><br>
**After running the above on a command line, more information on class methods can be found by typing the following at the prompt:**
<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```>>> help(imap)```

<!--## Google Colab

<!--All notebooks can be run in Google Colab, but ImageFont from PIL throws an error looking for the font_type. After the repo [//]:is cloned to colab, the **objectmapping.py** script needs its \_\_init\_\_ method edited so that a valid file path exists [//]:to the .ttf file. Users have used two approaches. 
[//]:1. Copy the .ttf file into the source folder and remove the path:<br><br>
[//]:```self.font_type = 'arialbd.ttf```<br>
[//]:```self.fnt = ImageFont.truetype(self.font_type, self.font_size)```<br><br>
[//]:2. Provide a path to the font_type:<br><br>
[//]:```self.font_type = '/path/to/arialbd.ttf```<br>
[//]:```self.fnt = ImageFont.truetype(self.font_type, self.font_size)```

## Environment

All development was done in an Anaconda environment using Keras with Tensorflow as the backend.


```python
!conda -V
```

    conda 4.7.12



```python
!python --version
```

    Python 3.7.4



```python
!conda list keras
```

    #
    # Name                    Version                   Build  Channel
    keras-applications        1.0.8                      py_0  
    keras-base                2.2.4                    py37_0  
    keras-gpu                 2.2.4                         0  
    keras-preprocessing       1.1.0                      py_1  



```python
!conda list tensorflow
```

    #
    # Name                    Version                   Build  Channel
    tensorflow                1.14.0          gpu_py37h74c33d7_0  
    tensorflow-base           1.14.0          gpu_py37he45bfe2_0  
    tensorflow-estimator      1.14.0                     py_0  
    tensorflow-gpu            1.14.0               h0d30ee6_0  



```python
!conda list
```

    #
    # Name                    Version                   Build  Channel
    _libgcc_mutex             0.1                        main  
    _tflow_select             2.1.0                       gpu  
    absl-py                   0.8.0                    py37_0  
    alabaster                 0.7.12                   py37_0  
    asn1crypto                0.24.0                   py37_0  
    astor                     0.8.0                    py37_0  
    astroid                   2.3.1                    py37_0  
    attrs                     19.1.0                   py37_1  
    babel                     2.7.0                      py_0  
    backcall                  0.1.0                    py37_0  
    blas                      1.0                         mkl  
    bleach                    3.1.0                    py37_0  
    bzip2                     1.0.8                h7b6447c_0  
    c-ares                    1.15.0            h7b6447c_1001  
    ca-certificates           2019.10.16                    0    anaconda
    cairo                     1.16.0            h18b612c_1001    conda-forge
    certifi                   2019.9.11                py37_0    anaconda
    cffi                      1.12.3           py37h2e261b9_0  
    chardet                   3.0.4                 py37_1003  
    click                     7.0                      pypi_0    pypi
    cloudpickle               1.2.2                      py_0  
    colorama                  0.4.1                    pypi_0    pypi
    cryptography              2.7              py37h1ba5d50_0  
    cudatoolkit               10.1.168                      0  
    cudnn                     7.6.0                cuda10.1_0  
    cupti                     10.1.168                      0  
    cycler                    0.10.0                   py37_0  
    cytoolz                   0.10.0           py37h7b6447c_0  
    dask-core                 2.5.0                      py_0  
    dbus                      1.13.6               he372182_0    conda-forge
    decorator                 4.4.0                    py37_1  
    defusedxml                0.6.0                      py_0  
    docutils                  0.15.2                   py37_0  
    entrypoints               0.3                      py37_0  
    expat                     2.2.6                he6710b0_0  
    ffmpeg                    4.1.3                h167e202_0    conda-forge
    filelock                  3.0.12                   pypi_0    pypi
    fontconfig                2.13.1            he4413a7_1000    conda-forge
    freetype                  2.9.1                h8a8886c_1  
    funcsigs                  1.0.2                    pypi_0    pypi
    gast                      0.3.2                      py_0  
    gettext                   0.19.8.1             hd7bead4_3  
    giflib                    5.1.4                h14c3975_1  
    glib                      2.58.3            h6f030ca_1002    conda-forge
    glob2                     0.7                        py_0    anaconda
    gmp                       6.1.2                h6c8ec71_1  
    gnutls                    3.6.5             hd3a4fd2_1002    conda-forge
    google-pasta              0.1.7                      py_0  
    graphite2                 1.3.13               h23475e2_0  
    grpcio                    1.16.1           py37hf8bcb03_1  
    gst-plugins-base          1.14.5               h0935bb2_0    conda-forge
    gstreamer                 1.14.5               h36ae1b5_0    conda-forge
    h5py                      2.9.0            py37h7918eee_0  
    harfbuzz                  2.4.0                h37c48d4_1    conda-forge
    hdf5                      1.10.4               hb1b8bf9_0  
    icu                       58.2                 h9c2bf20_1  
    idna                      2.8                      py37_0  
    imageai                   2.1.3                    pypi_0    pypi
    imageio                   2.5.0                    py37_0  
    imagesize                 1.1.0                    py37_0  
    importlib-metadata        0.23                     pypi_0    pypi
    intel-openmp              2019.4                      243  
    ipykernel                 5.1.2            py37h39e3cac_0  
    ipython                   7.8.0            py37h39e3cac_0  
    ipython_genutils          0.2.0                    py37_0  
    isort                     4.3.21                   py37_0  
    jasper                    1.900.1              hd497a04_4  
    jedi                      0.15.1                   py37_0  
    jeepney                   0.4.1                      py_0  
    jinja2                    2.10.1                   py37_0  
    joblib                    0.13.2                   py37_0  
    jpeg                      9c                h14c3975_1001    conda-forge
    jsonschema                3.0.2                    py37_0  
    jupyter_client            5.3.3                    py37_1  
    jupyter_core              4.5.0                      py_0  
    keras-applications        1.0.8                      py_0  
    keras-base                2.2.4                    py37_0  
    keras-gpu                 2.2.4                         0  
    keras-preprocessing       1.1.0                      py_1  
    keyring                   18.0.0                   py37_0  
    kiwisolver                1.1.0            py37he6710b0_0  
    lame                      3.100             h14c3975_1001    conda-forge
    lazy-object-proxy         1.4.2            py37h7b6447c_0  
    libblas                   3.8.0                    12_mkl    conda-forge
    libcblas                  3.8.0                    12_mkl    conda-forge
    libedit                   3.1.20181209         hc058e9b_0  
    libffi                    3.2.1                hd88cf55_4  
    libgcc-ng                 9.1.0                hdf63c60_0  
    libgfortran               3.0.0                         1    anaconda
    libgfortran-ng            7.3.0                hdf63c60_0  
    libiconv                  1.15                 h63c8f33_5  
    liblapack                 3.8.0                    12_mkl    conda-forge
    liblapacke                3.8.0                    12_mkl    conda-forge
    libpng                    1.6.37               hbc83047_0  
    libprotobuf               3.9.2                hd408876_0  
    libsodium                 1.0.16               h1bed415_0  
    libstdcxx-ng              9.1.0                hdf63c60_0  
    libtiff                   4.0.10               h2733197_2  
    libuuid                   2.32.1            h14c3975_1000    conda-forge
    libwebp                   1.0.1                h8e7db2f_0  
    libxcb                    1.13                 h1bed415_1  
    libxml2                   2.9.9                hea5a465_1  
    lz4-c                     1.8.1.2              h14c3975_0  
    markdown                  3.1.1                    py37_0  
    markupsafe                1.1.1            py37h7b6447c_0  
    matplotlib                3.1.1            py37h5429711_0  
    mccabe                    0.6.1                    py37_1  
    mistune                   0.8.4            py37h7b6447c_0  
    mkl                       2019.4                      243  
    mkl-service               2.3.0            py37he904b0f_0  
    mkl_fft                   1.0.14           py37ha843d7b_0  
    mkl_random                1.1.0            py37hd6b4f25_0  
    modin                     0.6.3                    pypi_0    pypi
    more-itertools            7.2.0                    pypi_0    pypi
    nbconvert                 5.6.0                    py37_1  
    nbformat                  4.4.0                    py37_0  
    ncurses                   6.1                  he6710b0_1  
    nettle                    3.4.1             h1bed415_1002    conda-forge
    networkx                  2.3                        py_0  
    nltk                      3.4.5                    py37_0  
    notebook                  6.0.1                    py37_0  
    numpy                     1.17.2           py37haad9e8e_0  
    numpy-base                1.17.2           py37hde5b4d6_0  
    numpydoc                  0.9.1                      py_0  
    olefile                   0.46                     py37_0  
    openblas                  0.2.19                        0    anaconda
    opencv                    4.1.0            py37h5517eff_4    conda-forge
    openh264                  1.8.0             hdbcaa40_1000    conda-forge
    openssl                   1.1.1                h7b6447c_0    anaconda
    packaging                 19.2                       py_0  
    pandas                    0.25.3           py37he6710b0_0    anaconda
    pandoc                    2.2.3.2                       0  
    pandocfilters             1.4.2                    py37_1  
    parso                     0.5.1                      py_0  
    patsy                     0.5.1                    py37_0    anaconda
    pcre                      8.43                 he6710b0_0  
    pexpect                   4.7.0                    py37_0  
    pickleshare               0.7.5                    py37_0  
    pillow                    6.1.0            py37h34e0f95_0  
    pip                       19.2.3                   py37_0  
    pixman                    0.38.0               h7b6447c_0  
    pluggy                    0.13.0                   pypi_0    pypi
    prometheus_client         0.7.1                      py_0  
    prompt_toolkit            2.0.9                    py37_0  
    protobuf                  3.9.2            py37he6710b0_0  
    psutil                    5.6.3            py37h7b6447c_0  
    pthread-stubs             0.3                  h0ce48e5_1  
    ptyprocess                0.6.0                    py37_0  
    py                        1.8.0                    pypi_0    pypi
    pycodestyle               2.5.0                    py37_0  
    pycparser                 2.19                     py37_0  
    pyflakes                  2.1.1                    py37_0  
    pygments                  2.4.2                      py_0  
    pylint                    2.4.2                    py37_0  
    pyopenssl                 19.0.0                   py37_0  
    pyparsing                 2.4.2                      py_0  
    pyqt                      5.9.2            py37h05f1152_2  
    pyrsistent                0.15.4           py37h7b6447c_0  
    pysocks                   1.7.1                    py37_0  
    pytest                    5.3.0                    pypi_0    pypi
    python                    3.7.4                h265db76_1  
    python-dateutil           2.8.0                    py37_0  
    pytz                      2019.2                     py_0  
    pywavelets                1.0.3            py37hdd07704_1  
    pyyaml                    5.1.2            py37h7b6447c_0  
    pyzmq                     18.1.0           py37he6710b0_0  
    qt                        5.9.7                h52cfd70_2    conda-forge
    qtawesome                 0.6.0                      py_0  
    qtconsole                 4.5.5                      py_0  
    qtpy                      1.9.0                      py_0  
    ray                       0.7.3                    pypi_0    pypi
    readline                  7.0                  h7b6447c_5  
    redis                     3.3.11                   pypi_0    pypi
    requests                  2.22.0                   py37_0  
    rope                      0.14.0                     py_0  
    scikit-image              0.15.0           py37he6710b0_0  
    scikit-learn              0.21.3           py37hd81dba3_0  
    scipy                     1.3.1            py37h7c811a0_0  
    seaborn                   0.9.0              pyh91ea838_1    anaconda
    secretstorage             3.1.1                    py37_0  
    selenium                  3.141.0         py37h14c3975_1000    conda-forge
    send2trash                1.5.0                    py37_0  
    setuptools                41.2.0                   py37_0  
    sip                       4.19.8           py37hf484d3e_0  
    six                       1.12.0                   py37_0  
    snowballstemmer           1.9.1                      py_0  
    sphinx                    2.2.0                      py_0  
    sphinxcontrib-applehelp   1.0.1                      py_0  
    sphinxcontrib-devhelp     1.0.1                      py_0  
    sphinxcontrib-htmlhelp    1.0.2                      py_0  
    sphinxcontrib-jsmath      1.0.1                      py_0  
    sphinxcontrib-qthelp      1.0.2                      py_0  
    sphinxcontrib-serializinghtml 1.1.3                      py_0  
    spyder                    3.3.6                    py37_0  
    spyder-kernels            0.5.2                    py37_0  
    sqlite                    3.30.0               h7b6447c_0  
    statsmodels               0.10.1           py37hdd07704_0    anaconda
    tensorboard               1.14.0           py37hf484d3e_0  
    tensorflow                1.14.0          gpu_py37h74c33d7_0  
    tensorflow-base           1.14.0          gpu_py37he45bfe2_0  
    tensorflow-estimator      1.14.0                     py_0  
    tensorflow-gpu            1.14.0               h0d30ee6_0  
    termcolor                 1.1.0                    py37_1  
    terminado                 0.8.2                    py37_0  
    testpath                  0.4.2                    py37_0  
    tk                        8.6.8                hbc83047_0  
    toolz                     0.10.0                     py_0  
    tornado                   6.0.3            py37h7b6447c_0  
    traitlets                 4.3.2                    py37_0  
    urllib3                   1.24.2                   py37_0  
    wcwidth                   0.1.7                    py37_0  
    webencodings              0.5.1                    py37_1  
    werkzeug                  0.16.0                     py_0  
    wget                      1.20.1               h20c2e04_0  
    wheel                     0.33.6                   py37_0  
    wrapt                     1.11.2           py37h7b6447c_0  
    wurlitzer                 1.0.3                    py37_0  
    x264                      1!152.20180806       h14c3975_0    conda-forge
    xorg-kbproto              1.0.7             h14c3975_1002    conda-forge
    xorg-libice               1.0.10               h516909a_0    conda-forge
    xorg-libsm                1.2.3             h84519dc_1000    conda-forge
    xorg-libx11               1.6.8                h516909a_0    conda-forge
    xorg-libxau               1.0.9                h14c3975_0    conda-forge
    xorg-libxdmcp             1.1.3                h516909a_0    conda-forge
    xorg-libxext              1.3.4                h516909a_0    conda-forge
    xorg-libxrender           0.9.10            h516909a_1002    conda-forge
    xorg-renderproto          0.11.1            h14c3975_1002    conda-forge
    xorg-xextproto            7.3.0             h14c3975_1002    conda-forge
    xorg-xproto               7.0.31            h14c3975_1007    conda-forge
    xz                        5.2.4                h14c3975_4  
    yaml                      0.1.7                had09818_2  
    zeromq                    4.3.1                he6710b0_3  
    zipp                      0.6.0                    pypi_0    pypi
    zlib                      1.2.11               h7b6447c_3  
    zstd                      1.3.7                h0b5b093_0  



```python

```
