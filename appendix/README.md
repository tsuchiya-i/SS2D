# Simple robot Simulator 2D
## Python
推奨Pythonバージョン = 3.6以上
## 要件 
- Python3.6
- OpenAI Gym
- OpenCV
- Pillow
- [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2)
- tensorflow
- Keras-rl2

 動作には上記ライブラリが必須です。下のインストール・ビルド方法を参考に進めてください。
## RVO2をビルド&インストール
下記コマンドでRVO2をインストール([詳細](https://github.com/sybrenstuvel/Python-RVO2))
```
$ git clone https://github.com/sybrenstuvel/Python-RVO2.git
$ cd Python-RVO2
$ pip3 install -r requirements.txt
$ python3 setup.py build
$ python3 setup.py install
```
最後のinstallでpermission errorが出た場合は下のコマンドでインストール
```
$ sudo python3 setup.py install
```

## OpenAI Gymのインストール
（おすすめ）
```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```
or
```
pip3 install gym
```

## OpenCVのインストール
(おすすめ)
```
$ sudo apt-get install python3-opencv
```
or
```
$ pip3 install opencv-python
```
環境によるかもしれないが上のaptでインストールしたopencvの方が速かった
## TensorFlowのインストール
```
$ pip3 install tensorflow
```
GPUを使用して学習など行いたい場合はtensorflow-gpuをインストールしTensorFlowのバージョンにあったCUDAやDriverを設定し行う必要がある。([TensorFlowとCUDAの対応バージョン](https://www.tensorflow.org/install/source?hl=ja#tested_build_configurations))([CUDAとDriverの対応バージョン](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html))

## SS2Dをインストール
```
$ git clone https://github.com/tsuchiya-i/SS2D.git
$ cd SS2D
$ pip3 install -e .
```
必ず-eをつけて編集可能なパッケージとしてインストール
# デモ動画
![demo](https://github.com/tsuchiya-i/SS2D/blob/main/appendix/navigation_sample.gif)

# Note
#### エラーコード1:
OpenCVをインストールできない時に発生
```
ModuleNotFoundError: No module named 'skbuild'
```
pipをアップデートするとうまくいく
```
pip3 install -U pip
```
#### エラーコード2:
```
ModuleNotFoundError: No module named 'tkinter'
```
標準ライブラリのtkinterがインストールされていない場合があるのでaptでインストール
```
sudo apt-get install python3-tk
```
#### エラーコード3:
```
ModuleNotFoundError: No module named 'PIL.ImageTk'
```
Pillowをインストールしても入ってないときはaptからインストール
```
sudo apt install python3-pil.imagetk
```

