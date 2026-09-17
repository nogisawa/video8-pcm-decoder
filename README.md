# v8pcm-rs

CXADC/CX Cardで取り込んだVideo8 PCM信号のFLACを復調し、CRC検証、
デインタリーブ、WAV変換を行うRust製の実験的デコーダです。

PythonやFFmpegを使わずに、次の処理を実行できます。

```text
CXADC FLAC
  -> bi-phase mark復調
  -> Video8 PCM物理ブロック
  -> CRC-16検証
  -> L/Rデインタリーブ
  -> 8-bit非線形PCM展開
  -> 31,469 Hz / 16-bit / stereo WAV
```

## 必要なもの

- Rust 1.92以降
- Cargo
- libsndfileと開発ヘッダ
- Cリンカ

AlmaLinux/RHEL系の例:

```sh
sudo dnf install rust cargo libsndfile-devel pkgconf-pkg-config gcc
```

`flac-devel`も導入して構いませんが、現在のFLAC入力はlibsndfile経由です。

## ビルド

```sh
cd v8pcm-rs
cargo build --release
```

実行ファイルは`target/release/`に生成されます。

```text
target/release/v8tune
target/release/v8demod
target/release/v8crc
target/release/v8decode
target/release/v8wav
```

テスト:

```sh
cargo test --release
```

## 最短の使い方

### 1. 復調パラメーターを調べる

実サンプルレートやNTSC/PALが分からない場合は、最初に`v8tune`を実行します。

```sh
target/release/v8tune ../capture.flac
```

出力例:

```text
Recommended:
v8demod INPUT.flac --system ntsc --sample-rate 28636000.000 --phases 12
```

### 2. FLACを生ブロックへ復調する

```sh
target/release/v8demod ../capture.flac \
  --system ntsc \
  --sample-rate 28636360 \
  -o pcm.bin
```

### 3. CRCを検証する

```sh
target/release/v8crc pcm.bin
```

不良ブロックの位置も表示する場合:

```sh
target/release/v8crc pcm.bin --show-bad
```

### 4. WAVへ変換する

```sh
target/release/v8wav pcm.bin output.wav
```

出力WAVは以下の形式です。

```text
サンプルレート: 31,469 Hz
量子化:         signed 16-bit PCM
チャンネル:     stereo
```

## パイプライン処理

中間ファイルを保存せず、標準入出力で接続できます。

```sh
target/release/v8demod ../capture.flac \
  --system ntsc \
  --sample-rate 28636360 \
  | target/release/v8crc --passthrough \
  | target/release/v8decode > output.s16le
```

`output.s16le`は次のヘッダなしPCMです。

```text
signed 16-bit little-endian / 31,469 Hz / stereo interleaved
```

WAVが必要なら、中間ブロックを保存して`v8wav`を使うのが簡単です。

```sh
target/release/v8demod ../capture.flac -o pcm.bin
target/release/v8wav pcm.bin output.wav
```

注意: CRC不良が1件でもあると`v8crc`は全データを通過させた後に終了コード1を
返します。通常のシェルパイプでは後段まで処理されますが、`set -o pipefail`を
使用しているスクリプトではパイプ全体が失敗扱いになります。

## v8tune: パラメーター自動探索

```sh
target/release/v8tune [OPTIONS] INPUT.flac
```

短い区間を候補パラメーターごとに実際に復調し、次の情報で採点します。

- CRCが一致したブロック数
- addressが連続すること
- PCMフィールド開始位置を検出できること

基本例:

```sh
target/release/v8tune ../capture.flac
```

広い範囲を細かく調べる例:

```sh
target/release/v8tune ../capture.flac \
  --system auto \
  --range-percent 2 \
  --steps 11 \
  --refine-passes 3 \
  --duration 0.1 \
  --phases 12
```

主なオプション:

| オプション | 意味 | 初期値 |
|---|---|---:|
| `--system auto\|ntsc\|pal` | 調査するテレビ方式 | `auto` |
| `--sample-rate HZ` | サンプルレート探索の中心 | FLACメタデータ |
| `--range-percent PCT` | 中心値の上下を探す割合 | `1` |
| `--steps N` | 1パスあたりの候補数 | `9` |
| `--refine-passes N` | 勝者周辺を再探索する回数 | `2` |
| `--duration SEC` | 評価する実時間 | `0.05` |
| `--start SEC` | 評価開始位置 | `0` |
| `--phases N` | 各候補で試すbit位相数 | `8` |
| `--window-ms MS` | 解析窓 | `8` |

FLACメタデータのサンプルレートが1 MHz未満の場合、CXADC FLACでよく使われる
1/1000表記と判断し、自動的に1000倍した値を探索中心にします。

探索精度を上げると処理時間も増えます。最初は既定値で方式と概算レートを調べ、
必要ならその値を`--sample-rate`に指定して狭い範囲を再探索してください。

## v8demod: FLAC復調

```sh
target/release/v8demod [OPTIONS] INPUT.flac
```

主なオプション:

| オプション | 意味 | 初期値 |
|---|---|---:|
| `-o, --output FILE` | 生ブロック出力。省略時はstdout | stdout |
| `--sample-rate HZ` | ADCの実サンプルレート | メタデータまたは1000倍値 |
| `--system ntsc\|pal` | テレビ方式 | `ntsc` |
| `--phases N` | 試行するクロック位相数 | `12` |
| `--window-ms MS` | 重複解析窓 | `8` |
| `--overlap-ms MS` | 解析窓の重複幅 | `4` |
| `--start SEC` | 復調開始位置 | `0` |
| `--duration SEC` | 復調する実時間 | ファイル末尾まで |

例:

```sh
target/release/v8demod ../capture.flac \
  --sample-rate 28636360 \
  --system ntsc \
  --phases 16 \
  --start 10 \
  --duration 5 \
  -o section.bin
```

`--phases`を増やすとクロック位相の取りこぼしを減らせる可能性がありますが、
処理時間はほぼ比例して増えます。

## v8crc: CRC検査

```sh
target/release/v8crc [OPTIONS] [INPUT|-]
```

入力を省略するか`-`を指定するとstdinから読みます。

| オプション | 意味 |
|---|---|
| `--passthrough` | 入力ブロックを変更せずstdoutへ出す |
| `--show-bad` | CRC不良ブロックの詳細をstderrへ出す |

統計例:

```text
blocks=40920 crc_ok=40795 crc_bad=125 success=99.695% address_errors=124
```

終了コード:

| コード | 意味 |
|---:|---|
| `0` | 全ブロックCRC一致 |
| `1` | 1件以上のCRC不一致 |
| その他 | 入出力または引数エラー |

## v8decode: ヘッダなしPCM出力

```sh
target/release/v8decode [OPTIONS] [INPUT|-] > output.s16le
```

CRC不良ブロックに対応するサンプルは、既定では正常な前後サンプルから補間します。

| オプション | 意味 |
|---|---|
| `--use-bad-crc` | CRC不良ブロックも補間せずデコードする |

## v8wav: WAV出力

```sh
target/release/v8wav [OPTIONS] INPUT.bin OUTPUT.wav
```

例:

```sh
target/release/v8wav pcm.bin output.wav
```

CRC不良ブロックをそのまま使う場合:

```sh
target/release/v8wav --use-bad-crc pcm.bin output.wav
```

通常は指定せず、補間を有効にしてください。

## 生ブロック形式

`v8demod`が出力するデータにはヘッダやJSONを含みません。

1ブロックは13 bytesです。

```text
offset  size  内容
0       1     address
1       1     Q parity
2       4     W0, W1, W2, W3
6       1     P parity
7       4     W4, W5, W6, W7
11      2     CRC-16、little-endian
```

物理順では次の並びです。

```text
address, Q, W0, W1, W2, W3, P, W4, W5, W6, W7, CRC-low, CRC-high
```

NTSCでは132ブロック、1,716 bytesが1フィールドです。

CRC生成多項式:

```text
x^16 + x^12 + x^5 + 1
preset = 0xffff
LSB first
```

## 現在の制限

- WAVデコードはNTSCの525 samples/channel/fieldに対応しています。
- `v8tune`はPAL候補も評価できますが、PALブロック出力とWAV化は未実装です。
- P/Qによる誤り訂正は未実装です。CRC不良サンプルは補間します。
- Rust復調器は実験段階です。入力によってはPython版よりCRC成功率が低くなる場合が
  あります。
- FLACメタデータのレートが実ADCクロックと異なる場合、`v8tune`または
  `--sample-rate`が必要です。
- 近似ノイズリダクションはRust版には入れていません。素のPCMを出力します。

## ヘルプ

各コマンドの現在のオプションは`--help`で確認できます。

```sh
target/release/v8tune --help
target/release/v8demod --help
target/release/v8crc --help
target/release/v8decode --help
target/release/v8wav --help
```
