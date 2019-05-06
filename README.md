# PFNインターン課題

## Requirements
python == 3.7.0

```
$ pip install -r requirements.txt
```


## アノテーションツール
ツールはすべての指で共通する指の付け根のkey pointとそれぞれの指で3箇所ずつのkey pointの設定を行い画像ファイルと同一名のjson ファイルを保存する
### ツールの起動
```
$ python annotate.py
```

### key point の設定 
ツールはVIEW, SEQUENTIAL, EDIT, ROOT の4つのモードがありVIEW モードのときにnを押下すると次の画像を表示，sを押下すると座標を保存する．

- Sequential: VIEWモードで a を押下するとすべてのkey pointをクリックで設定できる
- Edit: VIEWモードで 0-4(親指から小指に対応) を押下すると対応する指のkey pointをクリックで設定できる
- Root: VIEWモードで r を押下すると指の付け根のkey pointをクリックで設定できる

### 学習データのよみこみ
学習後に予測された座標をデフォルト値として読み込みたい場合は `-p` をつけて実行すると予測結果があるファイルはそれを読み込んで表示する.
※本課題では精度は求められていないのでtrainデータのみ読み込んでいる.
```
$ python annotate.py -p
```

## 学習/予測
### 学習の開始
```
$ python train.py -e [epoch] -g [GPU ID] -b [batch size] -lr [learning rate]
```
学習した結果(ログやスナップショット)は `result` 以下のタイムスタンプで示されるディレクトリに保存される.

### 予測結果の保存
timestampを指定して実行すると `data/pred_coordinates` へjsonファイルが保存される.
```
$ python predict.py -t [timestamp]
```

## 設定

`settings.py` にて以下の設定ができる.

- WINDOWSIZE: アノテーションツールのウィンドウサイズ.
- COLORS: アノテーションツールの指の色、上から下へ親指から小指に対応している.
- DEFAULT_POSITIONS: 224x224の画像に対して設定する.
- SEED: random, numpy, cupyのseed値を設定する.
- IMAGE_DIR: データセットのディレクトリ.
- COORDINATES_DIR: アノテーションの出力先.
- ANNOTATED_LOG: アノテーションしたファイルを記録するログファル名.
- PRED_COORDINATES_DIR: 予測されたアノテーションの出力先.
