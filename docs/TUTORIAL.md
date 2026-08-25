# BioExplorer チュートリアル

このドキュメントは、BioExplorerの各機能を実データで動かすための実践ガイドです。

1. インストール(コア機能 + 各機能に必要な外部ツール/DB)
2. 実データの入手方法
3. 実データを使った全機能の実行例(このリポジトリの`examples/opuntia/`に同梱の実データで実際に動かした結果を掲載)
4. 典型的なワークフロー
5. トラブルシューティング

---

## 1. インストール

### 1.1 コア機能

```bash
git clone <this-repo>
cd bioexplorer
uv sync            # biopython, click が入る。これだけで import/descriptor/search(kmer,minhash)/
                    # align(pairwise)/profile/logo/cluster(greedy)/tree(nj,upgma)/dnds(NG86,LWL85)/
                    # structure(解析系)/plot(alignment,tree)/replay が動く
uv run bio --help
```

### 1.2 オプション機能(Python extras)

```bash
uv sync --extra cluster   # scipy, scikit-learn -> クラスタリングの一部内部計算、bio embed --reduce pca/tsne
uv sync --extra embed     # umap-learn -> bio embed --reduce umap
uv sync --extra viz       # matplotlib, cairosvg -> bio plot / bio logo / bio profile --plot
uv sync --extra parquet   # pandas, pyarrow -> bio export ....parquet
# 複数同時指定も可能
uv sync --extra cluster --extra embed --extra viz --extra parquet
```

### 1.3 外部ツール(バイナリ)一覧

BioExplorerは仕様書の方針通り、専門的な計算処理の多くを外部ツールに委譲します。**未導入でも該当コマンドを打った時に「何が足りないか」を明示するエラーが出るだけで、他の機能には影響しません**。必要になった機能の分だけ入れてください。

大半は[Bioconda](https://bioconda.github.io/)で一括インストールできます(推奨):

```bash
conda create -n bioexplorer-tools -c bioconda -c conda-forge \
    mafft muscle clustalo \
    blast diamond mmseqs2 cd-hit \
    iqtree fasttree raxml \
    paml dssp
conda activate bioexplorer-tools
```

個別に入れる場合の対応表:

| 機能 | コマンド | 必要なツール | インストール例 |
|---|---|---|---|
| 高精度類似性検索 | `bio search FILE --method blast` | BLAST+ | `apt install ncbi-blast+` / `conda install -c bioconda blast` |
| 高精度類似性検索 | `bio search FILE --method diamond` | DIAMOND | `conda install -c bioconda diamond` / [公式](https://github.com/bbuchfink/diamond/releases) |
| 高精度類似性検索・クラスタリング | `--method mmseqs` | MMseqs2 | `conda install -c bioconda mmseqs2` / [公式](https://github.com/soedinglab/MMseqs2) |
| 多重整列 | `bio align --tool mafft` | MAFFT | `apt install mafft` / `conda install -c bioconda mafft` |
| 多重整列 | `bio align --tool muscle` | MUSCLE (v5系) | `conda install -c bioconda muscle` |
| 多重整列 | `bio align --tool clustalo` | Clustal Omega | `apt install clustalo` / `conda install -c bioconda clustalo` |
| クラスタリング | `bio cluster --method cdhit` | CD-HIT | `apt install cd-hit` / `conda install -c bioconda cd-hit` |
| 系統樹 | `bio tree --method iqtree` | IQ-TREE2 | `conda install -c bioconda iqtree` |
| 系統樹 | `bio tree --method fasttree` | FastTree | `apt install fasttree` / `conda install -c bioconda fasttree` |
| 系統樹 | `bio tree --method raxml` | RAxML / raxml-ng | `conda install -c bioconda raxml raxml-ng` |
| dN/dS(高精度) | `bio dnds --method YN00` | PAML(yn00) | `conda install -c bioconda paml` |
| 二次構造 | `bio structure ss` | DSSP | `apt install dssp` / `conda install -c bioconda dssp` |
| 構造ビューア | `bio structure view --viewer vmd` | VMD | [公式サイトで要登録DL](https://www.ks.uiuc.edu/Research/vmd/) |
| 構造ビューア | `bio structure view --viewer chimerax` | ChimeraX | [公式サイトで要登録DL](https://www.cgl.ucsf.edu/chimerax/) |
| 構造ビューア | `bio structure view --viewer pymol` | PyMOL | `conda install -c conda-forge pymol-open-source` |
| 構造予測 | `bio structure predict --engine colabfold` | ColabFold | [GitHub](https://github.com/sokrypton/ColabFold)(GPU推奨、`pip install colabfold`) |
| 構造予測 | `bio structure predict --engine alphafold` | AlphaFold | [GitHub](https://github.com/google-deepmind/alphafold)(大容量DB + GPU必須) |
| 構造予測 | `bio structure predict --engine modeller` | MODELLER | [salilab.org](https://salilab.org/modeller/)(要ライセンス、`conda install -c salilab modeller`) |
| 埋め込み(ESM) | `bio embed --method esm` | fair-esm + PyTorch | `pip install fair-esm torch` |
| 埋め込み(ProtT5) | `bio embed --method prott5` | transformers + PyTorch | `pip install transformers torch sentencepiece` |

---

## 2. 実データの入手方法

### 2.1 NCBI(DNA/RNA/タンパク質、GenBank形式)

NCBI E-utilitiesで配列を直接ダウンロードできます(APIキーなしでも動きますが、大量アクセスするなら[APIキー取得](https://www.ncbi.nlm.nih.gov/account/)推奨):

```bash
# 例: ヒトのBRCA1 mRNA配列(NM_007294)をFASTAで取得
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NM_007294&rettype=fasta&retmode=text" \
  -o brca1.fasta

# 例: 特定の生物種・遺伝子で検索してIDリストを取得 -> まとめて取得
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=protein&term=globin+AND+human[organism]&retmax=20" \
  -o search_result.xml
# search_result.xml の <Id> を集めて efetch に渡す
```

### 2.2 UniProt(タンパク質)

```bash
# 例: ヒトインスリン(P01308)をFASTAで取得
curl "https://rest.uniprot.org/uniprotkb/P01308.fasta" -o insulin.fasta

# 例: クエリ検索(globin, reviewed, human)して複数配列を一括取得
curl "https://rest.uniprot.org/uniprotkb/search?query=globin+AND+organism_id:9606+AND+reviewed:true&format=fasta&size=50" \
  -o globins_human.fasta
```

### 2.3 PDB / AlphaFold DB(タンパク質構造)

```bash
# 例: PDBエントリ 1A91 の構造ファイル
curl "https://files.rcsb.org/download/1A91.pdb" -o 1A91.pdb

# 例: UniProt ID P01308 に対応するAlphaFold予測構造
curl "https://alphafold.ebi.ac.uk/files/AF-P01308-F1-model_v4.pdb" -o insulin_af.pdb
```

### 2.4 このリポジトリに同梱の実データ

`examples/opuntia/` に、本チュートリアルの実行例で実際に使った**本物の実データ**を同梱しています。ウチワサボテン(*Opuntia*)の葉緑体rpl16イントロン領域、NCBI accession `AF191658`〜`AF191665`の7配列で、[Biopythonの公式チュートリアル](https://biopython.org/)でも系統解析の例として使われている定番データセットです。

- `opuntia_raw.fasta` — 生配列(`bio import`用、ギャップなし)
- `opuntia_aligned.fasta` — 上記のNCBI公開アラインメント済みバージョン(`bio profile`/`bio tree`用)
- `outputs/` — 本チュートリアルで実際に生成した出力(画像・CSV)

`examples/illustrative/` には、dN/dSと構造解析の説明用に**実データではない**簡易サンプル(`dnds_pair.fasta`、`toy_structure.pdb`)を置いています。本物のCDSペアやPDB構造は2.1〜2.3の方法で取得してください。

---

## 3. 実行例(実データ)

以下はすべて `examples/opuntia/` の実データに対して実際に実行し、そのままの出力を貼っています。

### 3.1 import / status

```bash
$ bio import examples/opuntia/opuntia_raw.fasta
imported 7 record(s); project total 7 (dna=7)

$ bio status
records: 7
  dna: 7
```

### 3.2 descriptor

```bash
$ bio descriptor
computed descriptors for 7 record(s)
```

GC%は24〜26%前後(葉緑体イントロンらしいAT-rich配列)であることが分かります:

```bash
$ bio search --field descriptor.gc_percent --field-max 25
1640bf476ee3    AF191658.1  dna 148
9387eef79fb2    AF191661.1  dna 146
71d3fd9e6ca5    AF191660.1  dna 146
a3e910bedbf9    AF191663.1  dna 150
26e83057c7a5    AF191665.1  dna 156
-- 5/7 record(s) matched
```

(seq_idは`bio import`のたびにランダムに再生成されるため、実行するたびに変わります。)

### 3.3 search(類似性検索)

```bash
$ bio search examples/opuntia/opuntia_raw.fasta --method kmer --top-n 3
note: examples/opuntia/opuntia_raw.fasta has 7 sequences; using the first ('AF191659.1') as query
1c32b0b3ceeb    AF191659.1  dna 146 score=1.0
1640bf476ee3    AF191658.1  dna 148 score=0.9518
9387eef79fb2    AF191661.1  dna 146 score=0.9518
-- 3 hit(s) (method=kmer)
```

**公共DBに対する検索**: `--method blast/diamond/mmseqs`はデフォルトではプロジェクト内の配列から一時DBを作って検索しますが、`--db`で既存のDBを指定すれば、NCBI nr/nt、UniRef、Pfam等の公共DBに対してそのまま検索できます。DBは各ツール公式のダウンローダを`bio db fetch`でラップしています(車輪の再発明はしていません):

```bash
# BLAST用DBをダウンロード(update_blastdb.plのラッパー)
bio db fetch swissprot --tool blast --output ./blastdb/swissprot

# MMseqs2用DBをダウンロード(mmseqs databasesのラッパー)
bio db fetch UniRef50 --tool mmseqs --output ./mmseqs_db/uniref50

# ダウンロードしたDBに対して検索
bio search query.fasta --method blast --db ./blastdb/swissprot --program blastn
bio search query.fasta --method mmseqs --db ./mmseqs_db/uniref50

# よく使うDB名の一覧
bio db list
```

外部DBに対する検索結果は、プロジェクト内の配列とは限らない(むしろ大半は含まれない)ため、`--type`/`--tag`等のプロジェクト向けフィルタは適用されず、素の`hit_id`とスコアがそのまま表示されます。`--export hits.csv`で保存可能です。DIAMONDには専用ダウンローダがないため、`update_blastdb.pl --decompress`等で落としたFASTAから`diamond makedb`で自分でDBを作る運用になります。

### 3.4 align(pairwise)

```bash
$ bio align --pairwise 1c32b0b3ceeb 1640bf476ee3 --mode global
mode=global  score=278.5
AF191659.1  TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGA--ATATATAAT
            |||||||||||||||||||||||||||||||||||||||||||||||||  |||||||||
AF191658.1  TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATAAT
...
```

**多重整列(MAFFT/MUSCLE/Clustal Omega)について**: この実行例を作った環境にはこれらのバイナリが入っていなかったため、`bio align --type dna --tool mafft` の代わりに、NCBIが公開している既知のアラインメント結果(`opuntia_aligned.fasta`)をそのまま使っています。MAFFT等をインストール済みなら、通常は次のように多重整列から始められます:

```bash
bio import examples/opuntia/opuntia_raw.fasta
bio align --type dna --tool mafft --name default
```

### 3.5 profile / logo / conservation plot / alignment viewer

```bash
$ bio profile --export profile.csv
7 sequences x 156 positions (dna)
consensus: TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA
mean conservation score: 0.5109
total information content: 159.40 bits
```

```bash
bio logo --output logo.png
bio profile --plot conservation --plot-output conservation.png
bio plot alignment --output alnview.png
```

| Sequence Logo | Conservation Plot |
|---|---|
| ![logo](../examples/opuntia/outputs/opuntia_logo.png) | ![conservation](../examples/opuntia/outputs/opuntia_conservation.png) |

Alignment Viewer:

![alignment viewer](../examples/opuntia/outputs/opuntia_alnview.png)

### 3.6 cluster

```bash
$ bio cluster --method greedy --threshold 0.9 --save-as project
7 record(s) -> 2 cluster(s) (method=greedy)
cluster sizes: [4, 3]
  cluster 0: n=4 rep=AF191665.1 centroid=AF191665.1 consensus_len=156 (approx)
  cluster 1: n=3 rep=AF191658.1 centroid=AF191659.1 consensus_len=148 (approx)
```

(`consensus_len=... (approx)`となっているのは、このクラスタ内で配列長が揃っていない=インデル差があるため、MSAツール未導入時はrepresentative配列を近似値として使っているからです。MAFFT等があれば厳密なconsensusが計算されます。)

**`hierarchical`(仕様書外、ChemExplorer/ProteinExploerer対応)**: `greedy`(依存なしの貪欲法)と`cdhit`/`mmseqs`(外部バイナリ、正確な配列同一性)の中間として、scipyベースの凝集型クラスタリングも用意しています。外部バイナリ不要、`uv sync --extra cluster`だけで使えます:

```bash
$ bio cluster --method hierarchical --n-clusters 2
7 record(s) -> 2 cluster(s) (method=hierarchical)
cluster sizes: [4, 3]
  cluster 0: n=3 rep=AF191665.1 centroid=AF191664.1 consensus_len=156 (approx)
  cluster 1: n=4 rep=AF191658.1 centroid=AF191661.1 consensus_len=148 (approx)

$ bio cluster --method hierarchical --linkage complete --distance-threshold 0.15
```

`--n-clusters`(欲しいクラスタ数を指定)と`--distance-threshold`(距離がこれを超えたら併合を止める、デフォルト0.3)はどちらか一方だけ指定してください。`--linkage`は`single`/`complete`/`average`/`weighted`から選べます(`ward`は今回のような事前計算済み距離行列とは相性が悪いため意図的に除外しています)。

### 3.7 tree(NJ + bootstrap)

```bash
$ bio tree --method nj --bootstrap 100 --save-as opuntia_nj
built nj tree: 7 taxa, 5 internal nodes
total branch length: 0.1276
saved tree as 'opuntia_nj' -> .bioexplorer/trees/opuntia_nj.nwk

$ bio plot tree --name opuntia_nj --output tree.png
```

![tree](../examples/opuntia/outputs/opuntia_tree.png)

内部枝の数字はbootstrap信頼度(100複製中の支持率)です。

### 3.8 embed(Sequence Space)

```bash
$ bio embed --method kmer --k 3 --reduce pca --plot space.png --color-by cluster_id
7 sequences -> kmer embedding -> pca (2D)
AF191659.1  -0.0272  0.0080
AF191658.1  -0.0132  0.0022
...
```

![sequence space](../examples/opuntia/outputs/opuntia_space.png)

色は3.6でつけた`cluster_id`。7配列と少なく、また埋め込み手法(kmer)とクラスタリング手法(greedy/kmer、閾値0.9)が別のパラメータで動いているため、PCA平面上できれいに分離するとは限りません — 実データはこのくらい素直にいかないものだ、という一例です。

### 3.9 dN/dS(illustrative)

dN/dSはコドン境界が揃った実際のCDS(タンパク質コード配列)ペアが必要です。今回のOpuntiaデータはイントロン(非コード領域)なので使えません。計算方法自体の確認用に、`examples/illustrative/dnds_pair.fasta`(実データではない最小サンプル)で示します:

```bash
$ bio import examples/illustrative/dnds_pair.fasta
$ bio dnds --all
geneA_illustrative  geneB_illustrative  dN=0.1093  dS=0.7166  omega(dN/dS)=0.1525
```

実際のオルソログCDSペアで解析したい場合は、2.1のNCBI E-utilitiesで`rettype=fasta_cds_na`を指定して取得してください:

```bash
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=<accession>&rettype=fasta_cds_na&retmode=text" -o cds.fasta
```

### 3.10 structure(illustrative)

同様に、構造解析には実際のPDB/mmCIFファイルが必要です。`examples/illustrative/toy_structure.pdb`はCLIの動作を示すための最小限の合成座標(5残基、CA原子のみ)です:

```bash
$ bio structure rmsd examples/illustrative/toy_structure.pdb examples/illustrative/toy_structure.pdb --chain-a A --chain-b A
RMSD: 0.000 A over 5 equivalent CA atoms
```

実際の構造で試すには2.3の方法で本物のPDBファイルを取得してください。DSSPやVMD等の外部ツールが入っていれば、そのまま`bio structure ss`/`bio structure view`が使えます。

### 3.11 annotate(Sequence Annotation)

**DNA/RNA(内蔵アルゴリズム、外部DB不要)**:

```bash
$ bio annotate orf --min-length 10
AF191659.1  2 orf(s)  longest=23aa
...
$ bio annotate promoter --search-window 156
AF191659.1  3 hit(s)
...
-- 7/7 record(s) have a candidate TATA box
```

Opuntiaデータ(葉緑体イントロン領域)ではGT-AG候補イントロンは0件でした。これは正しい挙動です — このデータ自体がすでにイントロン内部の断片なので、その中にさらに別のイントロン境界が見つからなくて当然です。

**Protein(内蔵アルゴリズム)**: 実際に2本の膜貫通ヘリックスを持つことが知られている大腸菌ATP合成酵素サブユニットC(PDB 1A91、`Tests/Fasta/f001`からNCBI/PDBの実データを取得)で試すと:

```bash
$ curl "https://raw.githubusercontent.com/biopython/biopython/master/Tests/Fasta/f001" -o atpsynthase.fasta
$ bio import atpsynthase.fasta
$ bio annotate protein-features
gi|3318709|pdb|1A91|  signal=False  tm=2  coiled_coil=2  low_complexity=3
```

内蔵のKyte-Doolittle疎水性スキャンだけで、実際の膜貫通ヘリックス2本(残基14-24, 59-79付近)を検出できています。単純な閾値ベースのヒューリスティックですが、この程度の検証には十分機能することが分かります。

```bash
$ bio annotate motif --list      # 内蔵PROSITE風パターン一覧
$ bio annotate motif             # 全パターンでスキャン
```

**外部DB/API連携**(この環境にはHMMER/インターネット経路がないため未検証、コードはドキュメント通りに実装):

```bash
# Pfamドメイン注釈(要HMMER + Pfam-A.hmm)
bio db fetch --tool pfam --output ./pfam/Pfam-A.hmm    # 約1.5GB、hmmpressまで自動実行
bio annotate pfam --hmm-db ./pfam/Pfam-A.hmm

# UniProt単体lookup
bio annotate uniprot P01308 --export insulin.json

# InterPro(EBI REST API、メールアドレス必須 -- EBI側の利用規約)
bio annotate interpro --email you@example.com
```

`bio replay`では`annotate`グループもデフォルトで`--skip`対象です(pfam/uniprot/interproがネットワーク・外部DB依存のため、structureのGUI起動と同じ扱い)。

### 3.12 clean(配列クリーンアップ、仕様書外、ChemExplorer/ProteinExplorer対応)

ChemExplorerの`chem standardize`、ProteinExplorerの`prot fix`に相当する、生データを解析にかける前のクリーンアップコマンドです。重複除去・gap除去・アダプタトリミング・FASTQ品質トリミング・曖昧塩基端トリミング・長さ/曖昧度フィルタを組み合わせて使えます。デフォルトではプロジェクトに直接書き戻すので、まず`--dry-run`で件数を確認するのがおすすめです。

```bash
$ bio import examples/opuntia/opuntia_raw.fasta
$ bio clean --dedup-sequence --dry-run
7 record(s) selected -> 7 kept
(dry run -- project not modified)
```

**FASTQ品質トリミング**の例(実際に動かした結果):

```bash
$ cat reads.fastq
@read1
TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGA
+
!!!!!IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII!!

$ bio import reads.fastq --format fastq
$ bio clean --min-quality 20 --quality-window 3
1 record(s) selected -> 1 kept
  quality-trimmed: 1
project updated: 1 record(s) total
```

先頭5塩基(`!`=Phred 0)と末尾2塩基が正しくトリムされ、配列と品質値の長さが常に一致していることも確認済みです。FASTQの品質値は`bio import`時に保持され、プロジェクト状態にも永続化されます(以前は破棄されていましたが、`bio clean`の実装に合わせて修正しました)。

他の主なオプション:

```bash
bio clean --strip-gaps                              # 誤ってgap付きで取り込んだ配列からgapを除去
bio clean --trim-ambiguous-ends                      # 先頭・末尾のN/曖昧塩基(proteinはX)を除去
bio clean --max-ambiguous-fraction 0.1                # 曖昧塩基が10%を超える配列を除外
bio clean --adapter AGATCGGAAGAGC --adapter-end 3     # 3'側のアダプタ配列を除去(完全一致)
bio clean --result-min-length 50 --result-max-length 500  # トリミング後の長さで除外
```

`--min-length`/`--max-length`(絞り込み対象を選ぶ、`bio search`と共通の語彙)と、`--result-min-length`/`--result-max-length`(トリミング後の結果を長さで落とす、cleanコマンド自身のオプション)は別物なので注意してください。前者は「どの配列を掃除するか」、後者は「掃除した後に短すぎる/長すぎるものを捨てるか」を制御します。

対象範囲は他のコマンドと同じ非破壊selection語彙(`--tag`/`--type`/`--field`等)で絞り込めます。選択されなかった配列はそのまま残ります:

```bash
bio clean --strip-gaps --tag needs_cleanup   # needs_cleanupタグの付いた配列だけ処理
```

### 3.13 仕上げ機能(仕様書外、ChemExplorer対応)

仕様書には無いが実用上効く4つの機能を追加しています。実データ(Opuntia、`bio cluster --save-as project`まで実行済みとする)で確認します。

**(1) `!=`除外条件**:

```bash
$ bio search --field descriptor.gc_percent --field-equals 25.0
-- 1/7 record(s) matched
$ bio search --field descriptor.gc_percent --field-not-equals 25.0
-- 6/7 record(s) matched   # 1件除外されてちょうど逆になる
```

タグには値がないので、`!=`に相当するのは`--exclude-tag`(そのタグを持たないものだけ残す):

```bash
$ bio search --exclude-tag cluster_centroid
-- 5/7 record(s) matched   # centroidタグの2件を除外
```

**(2) motif除外・IDブラックリスト**:

```bash
$ bio search --exclude-motif "CTAATAAATTAGATGAATAT"
-- 0/7 record(s) matched   # 全配列に共通の高保存領域なので、除外すると0件になる
$ bio search --exclude-id <seq_id>
-- 6/7 record(s) matched
```

**(3) 出力系コマンドへの非破壊selection**: `bio export`/`bio align`/`bio profile`/`bio logo`/`bio tree`/`bio plot`/`bio cluster`/`bio embed`/`bio dnds`が`bio search`と同じ語彙(`--tag`/`--type`/`--field`/`--motif`/`--exclude-*`)を非破壊で受け付けます。プロジェクトは変更されないので、条件を変えるたびに再importする必要がありません:

```bash
$ bio export cluster0_only.fasta --tag cluster_0
exported 4/7 record(s) to cluster0_only.fasta (fasta)
$ bio status   # プロジェクトは7件のまま変わらない
records: 7

$ bio profile --tag cluster_0       # 4配列分だけのプロファイル
4 sequences x 156 positions (dna)
mean conservation score: 0.3798
$ bio profile --exclude-tag cluster_0   # 残り3配列分
3 sequences x 156 positions (dna)
mean conservation score: 0.2971
$ bio profile                            # 全7配列(比較用)
7 sequences x 156 positions (dna)
mean conservation score: 0.5109

$ bio tree --tag cluster_0 --save-as cluster0_tree    # 4taxaの部分木
built nj tree: 4 taxa, 2 internal nodes
$ bio plot tree --tree-name full_tree --tag cluster_1 --output pruned.png  # 保存済みの全体木を後からprune
```

アラインメント由来のコマンド(`bio profile`/`bio logo`/`bio tree`/`bio plot alignment`)はアラインメントファイル自体にタグ・メタデータが無いため、プロジェクト内の同名レコードと突き合わせてタグを補完してからフィルタします。`bio plot tree`はNewick木そのものにタグが無いので、プロジェクトと名前で突き合わせて該当しないtaxaを`Bio.Phylo`の`prune`で落とします。

**(4) `bio report`(任意次元クロス集計)**:

```bash
$ bio report --by type --by tag_prefix:cluster_
type    tag_prefix:cluster_       count
dna     0                         3
dna     0,centroid,representative 1
dna     1                         1
dna     1,centroid                1
dna     1,representative          1
-- 5 combination(s) over 7 record(s)

$ bio report --by tag_prefix:cluster_ --export report.csv   # CSV出力
```

`--by`の軸指定は`type`(seq_type)、`tag:<name>`(そのタグの有無、yes/no)、`tag_prefix:<prefix>`(前方一致するタグをカテゴリ化、例: `cluster_0`→`0`)、`field:<dotted.key>`(メタデータの値そのもの)、`field:<dotted.key>:bin<幅>`(数値メタデータを固定幅でビン分け)の5種類。`--by`を複数指定すればN次元クロス集計になります。`bio report`自体も上記の非破壊selectionオプションで集計範囲を絞れます。

### 3.14 replay

```bash
$ bio replay --dry-run
[1] would run: bio import examples/opuntia/opuntia_raw.fasta
[2] would run: bio descriptor
[3] would run: bio align --pairwise 7877691521df f70db3ff8da3 --mode global

$ bio replay
backed up previous project state to .bioexplorer_prereplay_1787478995
[1] ok: bio import examples/opuntia/opuntia_raw.fasta
[2] ok: bio descriptor
[3] ok: bio align --pairwise 011540817805 657536d8c5dc --mode global (id rewritten)
-- 3 executed, 0 skipped, 0 failed
```

**`seq_id`の非決定性への対応**: `seq_id`は`bio import`のたびにランダムに再生成されるため、`bio align --pairwise <seq_id> <seq_id>`のように**seq_idを直接指定したコマンド**は、そのままではreplay先のプロジェクトに存在しないIDを参照することになります。上の実行例では、記録時のID(`7877691521df`/`f70db3ff8da3`)がreplay時には`011540817805`/`657536d8c5dc`に変わっていますが、`(id rewritten)`の表示通り自動的に読み替えられて成功しています。

仕組みは、リセット前(=元の実行が残した状態)から「旧ID→配列名」の対応表を作っておき、各ステップを実行する直前に「その配列名の現在のID」に変換してからargvを渡す、というものです。ProteinExplorer側で先に解決されていた同じ問題を移植しました。

これでも救えないケースが2つあります: (1) `--dry-run`はプレビューなので実際には何も実行されず、書き換えも行われません(記録された生のIDがそのまま表示されます)。(2) `--from`で該当する`bio import`ステップ自体をスキップした場合や、レコードが元々存在しない場合は、対応する新IDが見つからないため書き換えられず、そのまま失敗します(想定通りの挙動です)。

---

## 4. 典型的なワークフロー

```
実データ取得(NCBI/UniProt)
        │
        ▼
   bio import
        │
        ▼
  bio descriptor ─────► bio search(フィルタ・類似性検索で対象を絞り込み)
        │
        ▼
  bio align --tool mafft --name default
        │
        ├──► bio profile / bio logo(保存度解析・可視化)
        │
        ├──► bio cluster(--save-as project でタグ付け)
        │           │
        │           ▼
        │     bio embed --color-by cluster_id(Sequence Space可視化)
        │
        └──► bio tree --bootstrap N(系統樹)
                    │
                    ▼
              bio plot tree

(コード配列があれば) bio dnds --all で選択圧を確認
(構造があれば)        bio structure map-conservation で保存度を構造にマッピング

最後に bio replay --dry-run で記録済みワークフローを確認・共有
```

## 5. トラブルシューティング

- **`Error: 'xxx' was not found on PATH`**: 1章の対応表を見て該当ツールを入れてください。入れたくない場合、`--method kmer`/`--method minhash`(類似性検索・埋め込み)、`--method nj`/`--method upgma`(系統樹)など、外部ツール不要な代替手段が大抵用意されています。
- **`bio profile`/`bio tree`が「アラインメントが必要」と言ってくる**: 配列長が揃っていません。先に`bio align`で整列するか、`--alignment-file`で外部で作った整列済みFASTAを指定してください。
- **`bio replay`が特定ステップで失敗する**: seq_id直指定のステップはID自動書き換えで大半は再現できますが、`--from`で該当importをスキップした場合など、対応するレコードが今回のreplayに存在しないと書き換えようがなく失敗します。ログ(`.bioexplorer/log.json`)を見て、該当ステップを`--skip`で除外するか、名前ベースのコマンドに書き換えてから再実行してください。
- **matplotlib関連のエラー**: `uv sync --extra viz`を実行してください。
