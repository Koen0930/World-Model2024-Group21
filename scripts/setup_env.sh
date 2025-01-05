#!/bin/bash

# エラーハンドリング: 何かのコマンドが失敗した場合、スクリプトを停止
set -e

# 0. numpy の特定バージョンをインストール
echo "Installing numpy==1.26.4..."
pip install --upgrade pip
pip install numpy==1.26.4


echo "依存関係をインストール中 (pip install -e .)..."
# 環境を整理して必要な依存関係を明示的にインストール
pip uninstall jax jaxlib chex optax -y
pip install "ruamel.yaml<0.18"
pip install chex==0.1.86
pip install optax==0.1.7
# `dynalang` をインストール
pip install "jax[cuda]"
pip install sentencepiece
pip install ffmpeg
pip install datasets

echo "セットアップが完了しました！"
