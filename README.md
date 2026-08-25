# BioExplorer

CLIベースのバイオインフォマティクス・ワークベンチ。[ChemExplorer](../chemexplorer)のバイオインフォマティクス版として、配列管理・検索・アノテーション・類似性解析・プロファイル解析・アライメント・クラスタリング・系統解析・可視化を一貫して実行できる環境を提供する。

Biopythonをコアライブラリとし、専門処理(高精度類似性検索、多重整列、系統樹推定、構造予測など)は外部ツールとの統一インターフェースとして提供する。

## インストール

```bash
uv sync                                              # コア機能(biopython, click)
uv sync --extra cluster --extra embed --extra viz --extra parquet    # クラスタリング/埋め込み/可視化/Parquet出力の追加機能
```

外部ツール(MAFFT, BLAST, DSSPなど)のインストール方法は **[docs/TUTORIAL.md](docs/TUTORIAL.md)** の対応表を参照。

## クイックスタート

```bash
uv run bio import examples/opuntia/opuntia_raw.fasta
uv run bio descriptor
uv run bio status
```

実データ(NCBI由来の実際の配列)を使った全機能のウォークスルーは **[docs/TUTORIAL.md](docs/TUTORIAL.md)** を参照。実データの入手方法(NCBI/UniProt/PDB)、必要な外部ツール/DBのインストール手順、典型的なワークフローを解説している。

## 実装済み機能

| コマンド | 機能 |
|---|---|
| `bio import` / `bio export` / `bio status` | 配列の取り込み・書き出し・プロジェクト状態確認(FASTA/FASTQ/GenBank/EMBL, CSV/TSV/JSON/Parquet) |
| `bio descriptor` | 配列記述子(GC%, コドン使用, 分子量, pI, GRAVY等) |
| `bio search` | 検索・フィルタ(ID/Name/Tag/Length/Motif/メタデータ)+ 類似性検索(k-mer/MinHash/BLAST/DIAMOND/MMseqs2) |
| `bio align` | Pairwise(Needleman-Wunsch/Smith-Waterman)+ Multiple(MAFFT/MUSCLE/Clustal Omega) |
| `bio profile` / `bio logo` | PFM/PPM/PWM/PSSM, consensus, Shannon entropy, conservation score, relative entropy |
| `bio cluster` | greedy(CD-HIT式, 内蔵)/ hierarchical(scipy凝集型, 内蔵)/ CD-HIT / MMseqs2、Representative/Centroid/Consensus |
| `bio tree` | NJ/UPGMA(内蔵, bootstrap対応)/ IQ-TREE / FastTree / RAxML |
| `bio dnds` | dN/dS(Ka/Ks): NG86/LWL85(内蔵)/ YN00(PAML) |
| `bio structure` | PDB/mmCIF解析(配列抽出・RMSD・二次構造・conservation mapping)+ 予測/ビューア外部連携 |
| `bio embed` | Sequence Space: k-mer/MinHash/ESM/ProtT5埋め込み + PCA/t-SNE/UMAP |
| `bio plot` | Alignment Viewer / Phylogenetic Tree / Sequence Space(+ `bio profile --plot`でLogo/Heatmap/Conservation Plot) |
| `bio annotate` | ORF/CDS・GT-AG intron候補・TATA box(DNA/RNA、内蔵)、Signal Peptide/TM領域/Coiled Coil/Low Complexity/PROSITE風motif(protein、内蔵)、Pfam(ローカルHMMER)/UniProt/InterPro(外部DB・API連携) |
| `bio db` | BLAST/MMseqs2/Pfamの公式ダウンローダのラッパー(`bio search --db`/`bio annotate pfam`用) |
| `bio replay` | 記録済みワークフロー(`.bioexplorer/log.json`)の再実行・再現性検証 |
| `bio clean` | 配列クリーンアップ(仕様書外): 重複除去・gap除去・アダプタトリミング・FASTQ品質トリミング・曖昧塩基端トリミング・長さ/曖昧度フィルタ |
| `bio report` | tag/type/metadataの任意次元クロス集計(CSV出力可、仕様書外) |

仕様書の全21セクションを実装完了。加えて、ChemExplorerと揃える形で以下の「仕上げ」を追加(いずれも仕様書外):
- `--tag`/`--field`条件の`!=`除外(`--exclude-tag`/`--field-not-equals`)
- motif正規表現除外(`--exclude-motif`)とIDブラックリスト(`--exclude-id`)
- `bio export`/`bio align`/`bio profile`/`bio logo`/`bio tree`/`bio plot`/`bio cluster`/`bio embed`/`bio dnds`/`bio report`への非破壊selectionオプション(`bio search`と同じ語彙、プロジェクトを書き換えずその場だけ絞り込み)

## テスト

```bash
uv run pytest -q                                             # コア機能のみ
uv sync --extra cluster --extra embed --extra viz --extra parquet && uv run pytest -q  # 全機能(拡張機能込み)
```

### CI

`.github/workflows/test.yml`でGitHub Actionsによる自動テストを構成済み(push/pull request時に実行、`workflow_dispatch`で手動実行も可)。ProteinExplorer/ChemExplorerと同じ2ジョブ構成:

- **core**: `uv sync --extra dev`のみ(拡張機能なし)でテスト実行。scikit-learn/umap-learn/matplotlibが必要なテスト(`bio embed --reduce pca/tsne/umap`、`bio plot`等)は`pytest.mark.skipif`で自動スキップされ、失敗にはならない
- **full**: `cluster`/`embed`/`viz`/`parquet`の全extrasを`libcairo2`(cairosvgの実行時依存)込みでインストールし、全テスト実行 + 実データ(Opuntia)でのCLIスモークテスト(import→descriptor→search→cluster→embed→report→parquet export)

coreで199 passed/8 skipped、fullで212 passedをこの環境で実地検証済み。リポジトリをGitHubにpushして`git remote add origin ...`すれば、Actionsタブでそのまま動く。まだリモート未設定のためバッジは付けていない。
