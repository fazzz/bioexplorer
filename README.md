# BioExplorer

CLIベースのバイオインフォマティクス・ワークベンチ。[ChemExplorer](../chemexplorer)のバイオインフォマティクス版として、配列管理・検索・アノテーション・類似性解析・プロファイル解析・アライメント・クラスタリング・系統解析・可視化を一貫して実行できる環境を提供する。

Biopythonをコアライブラリとし、専門処理(高精度類似性検索、多重整列、系統樹推定、構造予測など)は外部ツールとの統一インターフェースとして提供する。

## インストール

```bash
uv sync                                              # コア機能(biopython, click)
uv sync --extra cluster --extra embed --extra viz    # クラスタリング/埋め込み/可視化の追加機能
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
| `bio cluster` | greedy(CD-HIT式, 内蔵)/ CD-HIT / MMseqs2、Representative/Centroid/Consensus |
| `bio tree` | NJ/UPGMA(内蔵, bootstrap対応)/ IQ-TREE / FastTree / RAxML |
| `bio dnds` | dN/dS(Ka/Ks): NG86/LWL85(内蔵)/ YN00(PAML) |
| `bio structure` | PDB/mmCIF解析(配列抽出・RMSD・二次構造・conservation mapping)+ 予測/ビューア外部連携 |
| `bio embed` | Sequence Space: k-mer/MinHash/ESM/ProtT5埋め込み + PCA/t-SNE/UMAP |
| `bio plot` | Alignment Viewer / Phylogenetic Tree / Sequence Space(+ `bio profile --plot`でLogo/Heatmap/Conservation Plot) |
| `bio annotate` | ORF/CDS・GT-AG intron候補・TATA box(DNA/RNA、内蔵)、Signal Peptide/TM領域/Coiled Coil/Low Complexity/PROSITE風motif(protein、内蔵)、Pfam(ローカルHMMER)/UniProt/InterPro(外部DB・API連携) |
| `bio db` | BLAST/MMseqs2/Pfamの公式ダウンローダのラッパー(`bio search --db`/`bio annotate pfam`用) |
| `bio replay` | 記録済みワークフロー(`.bioexplorer/log.json`)の再実行・再現性検証 |

仕様書の全21セクションを実装完了。

## テスト

```bash
uv run pytest -q
```
