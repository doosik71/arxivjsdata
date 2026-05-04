# Visual Attention Exploration in Vision-Based Mamba Models

Junpeng Wang, Chin-Chia Michael Yeh, Uday Singh Saini, Mahashweta Das (2025)

## 🧩 Problem to Solve

본 연구는 최근 Transformer의 효율적인 대안으로 부상한 State Space Models (SSMs), 특히 Mamba 모델이 시각 데이터(Vision)를 처리하는 방식에서의 Attention 메커니즘을 해석하는 것을 목표로 한다.

Vision-based Mamba 모델은 2D 이미지를 작은 패치(patch)로 분해하고 이를 1D 시퀀스로 배열하여 입력으로 사용한다. Transformer는 모든 패치를 동시에 입력하며 Positional Encoding을 통해 위치 정보를 구분하므로 패치의 배열 순서가 상대적으로 덜 중요하지만, Mamba는 RNN과 유사하게 패치를 순차적으로 처리한다. 이로 인해 각 패치는 이전 패치들로부터만 정보를 수집할 수 있으며, 이는 패치의 배열 순서가 모델의 Attention 분포에 결정적인 영향을 미침을 의미한다.

따라서 본 논문은 다음과 같은 세 가지 구체적인 의문을 해결하고자 한다. 첫째, 동일한 스테이지 내의 서로 다른 Mamba 블록들이 일관된 Attention 패턴을 보이는가? 둘째, 모델의 층이 깊어짐에 따라 Attention 패턴이 어떻게 진화하는가? 셋째, 패치 배열 전략(patch-ordering strategies)이 학습된 Attention 패턴에 구체적으로 어떤 영향을 미치는가?

## ✨ Key Contributions

본 논문의 핵심 기여는 Vision-based Mamba 모델의 Attention 패턴을 탐색하고 요약할 수 있는 시각적 분석(Visual Analytics) 도구를 설계하고 구현한 것이다. 이 도구는 고차원의 Attention 행렬을 차원 축소 기술을 통해 시각화함으로써, Mamba 블록 간의 유사성과 개별 블록 내 패치 간의 상호작용을 분석할 수 있게 한다. 또한, 다양한 패치 배열 전략을 제안하고 이를 실제 모델에 적용하여 Attention 패턴이 입력 시퀀스의 순서에 따라 어떻게 변화하는지를 정량적, 정성적으로 분석하여 모델의 동작 원리에 대한 새로운 통찰을 제공한다.

## 📎 Related Works

기존의 State Space Models는 S4와 같은 구조에서 시작하여, 입력 토큰에 학습 가능한 가중치를 할당하는 Selective Scan 메커니즘을 도입한 Mamba(S6)로 발전하였다. 시각 영역에서는 Vim과 VMamba가 대표적이며, 특히 VMamba는 2D 공간 정보의 손실을 최소화하기 위해 네 가지 방향의 Cross-scan 메커니즘을 사용하여 패치를 처리한다.

Attention 메커니즘의 해석을 위한 기존 연구들은 주로 Transformer 기반의 NLP나 ViT(Vision Transformer)에 집중되어 왔다. 이들은 주로 Heatmap, Parallel Coordinate Plot, 또는 차원 축소 기반의 Scatterplot을 사용하여 Attention 강도를 시각화하였다. 그러나 Mamba 모델의 경우, 시퀀스 기반의 처리 특성상 Transformer와는 다른 Attention 양상을 보일 가능성이 높음에도 불구하고, 이를 전문적으로 분석한 연구는 지금까지 없었다. 본 연구는 이러한 공백을 메우기 위해 Mamba 전용 시각 분석 도구를 제안하며 기존의 Transformer 분석 방법론을 Mamba의 특성에 맞게 확장하였다.

## 🛠️ Methodology

본 연구는 VMamba 모델을 대상으로 분석을 수행하며, ImageNet 데이터셋으로 학습된 모델을 사용한다. 입력 이미지 크기는 $224 \times 224 \times 3$이며, 4개의 스테이지를 거치며 패치 크기가 $56 \times 56 \to 28 \times 28 \to 14 \times 14 \to 7 \times 7$로 점진적으로 줄어든다.

### 1. Inter-Block Attention Pattern 분석

동일 스테이지 내의 서로 다른 블록들 간의 Attention 패턴 유사성을 측정한다.

- 각 스테이지의 $m$개 블록에서 $n$개의 테스트 이미지에 대한 Attention 행렬을 추출한다.
- 각 Attention 행렬의 크기는 $p^2 \times p^2$이며, 이를 하나의 데이터 포인트로 간주한다.
- 전체 데이터 형태는 $(m \times n) \times (p^2 \times p^2)$가 되며, 이를 PCA, tSNE, UMAP과 같은 차원 축소(Dimensionality Reduction, DR) 기술을 사용하여 2차원 평면에 투영한다.
- 만약 서로 다른 블록의 포인트들이 독립된 클러스터를 형성한다면, 이는 블록 간 Attention 패턴이 서로 다름을 의미한다.

### 2. Intra-Block Attention Pattern 분석

단일 블록 내에서 각 패치가 가지는 Attention의 공간적 분포를 분석한다.

- 특정 블록과 특정 이미지에 대한 Attention 행렬 $A \in \mathbb{R}^{p^2 \times p^2}$에서 $i$번째 행은 패치 $i$가 다른 모든 패치에 부여하는 Attention 강도를 나타낸다.
- 이미지 콘텐츠에 의한 영향을 배제하고 일반적인 패턴을 찾기 위해, $n$개 이미지의 Attention 행렬을 평균하여 하나의 대표 행렬 $\text{avg\_attn}$을 생성한다.
$$\text{avg\_attn} = \frac{1}{n} \sum_{i=1}^{n} \text{Attention}(image_i, stage, block)$$
- 이 $\text{avg\_attn}$ 행렬의 각 행(패치별 패턴)을 차원 축소하여 $p^2 \times 2$의 좌표로 변환하고 Scatterplot으로 시각화한다.

### 3. 시각적 분석 시스템 구성

- **ScatterplotView**: 차원 축소 결과를 보여준다. Mode 1에서는 블록 간 유사성을, Mode 2에서는 패치 간 유사성을 탐색한다. 특히 패치 간 분석 시, 포인트의 색상과 크기를 패치의 열(column)과 행(row) 인덱스에 매핑하여 공간적 상관관계를 파악한다.
- **PatchView**: 선택된 패치들을 2D 그리드 형태로 보여준다. Scatterplot에서 선택한 클러스터가 실제 이미지의 어느 위치에 해당하는지 하이라이트하며, 특정 패치를 클릭하면 해당 패치가 다른 패치들에 부여하는 Attention 강도를 Heatmap 형태로 시각화한다.

## 📊 Results

### 1. 블록 및 스테이지별 Attention 특성

- **블록 간 차별성**: 동일 스테이지 내의 블록들이 Scatterplot에서 명확히 분리된 클러스터를 형성함을 확인하였다. 이는 각 블록이 서로 다른 Attention 역할을 수행하고 있음을 시사한다.
- **상호 보완적 패턴**: 일부 블록은 자기 자신과 주변 패치에 강하게 집중하는 반면, 다른 블록은 오히려 주변을 무시하고 먼 거리의 패치에 집중하는 상호 보완적(complementary)인 양상을 보였다.
- **계층적 진화**: 초기 스테이지(Stage 0, 1)에서는 공간적으로 인접한 패치들이 유사한 Attention 패턴을 가지는 강한 공간적 상관관계가 나타난다. 그러나 후반 스테이지(Stage 3)로 갈수록 클러스터 구조가 희미해지며, 이는 Attention이 단순한 위치 정보보다는 이미지의 실제 콘텐츠(content-dependent)에 더 많이 의존하게 됨을 의미한다.

### 2. 패치 배열 순서(Patch Order)의 영향

연구팀은 패치 배열 순서를 변경한 세 가지 대안 전략(Diagonal, Morton/z-order, Spiral)을 제안하고 모델을 처음부터 다시 학습시켜 비교하였다.

- **정량적 결과**: 세 가지 새로운 배열 전략 모두 ImageNet에서 $82.6\%$ 이상의 정확도를 달성하여, 원래의 VMamba 모델과 유사한 성능을 보였다.
- **정성적 결과**: Attention 패턴은 배열 순서에 따라 완전히 달라졌다. 예를 들어 Diagonal order를 사용하면 Attention이 대각선 방향의 이전 패치들에 집중되는 경향이 나타났다. 이는 Mamba의 Attention이 '공간적 위치' 그 자체보다는 '시퀀스 상의 선후 관계(preceding patches)'에 강하게 결합되어 있음을 입증한다.

## 🧠 Insights & Discussion

본 연구는 Vision-based Mamba 모델의 내부 동작 방식에 대해 중요한 통찰을 제공한다. 가장 핵심적인 발견은 Mamba의 Attention이 입력 시퀀스의 순서에 극도로 의존적이라는 점이다. 이는 Transformer가 Positional Encoding을 통해 위치를 '참조'하는 것과 달리, Mamba는 시퀀스라는 '통로'를 통해 정보를 전달받기 때문에 발생하는 현상이다.

특히 초기 층에서 나타나는 공간적 유사성은 Mamba가 효율적으로 지역적 특징(local features)을 추출하고 있음을 보여주며, 후반 층에서 콘텐츠 중심의 Attention으로 전환되는 과정은 CNN이나 ViT에서 관찰되는 계층적 특징 추출 과정과 일맥상통한다.

다만, 본 연구는 Attention 패턴의 시각적 분석에 집중하여, 배열 순서의 변경이 실제 추론 성능의 미세한 차이나 특정 엣지 케이스에서의 강건성(robustness)에 어떤 영향을 미치는지에 대해서는 심도 있게 다루지 않았다. 또한, 콘텐츠-불가지론적(content-agnostic) 패턴을 찾기 위해 평균값을 사용했으므로, 개별 이미지의 특수한 상황에서 발생하는 Attention 오류를 진단하는 데에는 한계가 있다.

## 📌 TL;DR

본 논문은 Vision-based Mamba 모델의 Attention 메커니즘을 분석하기 위한 시각적 분석 도구를 제안하고, 이를 통해 블록 간의 상호 보완적 역할과 스테이지별 계층적 패턴(지역적 $\to$ 콘텐츠 중심)을 발견하였다. 특히, 패치 배열 순서를 변경했을 때 모델의 정확도는 유지되지만 Attention 패턴은 시퀀스 순서에 따라 완전히 재구성됨을 확인하여, Mamba 모델의 시퀀스 의존적 특성을 명확히 규명하였다. 이 연구는 향후 공간 지역성을 더 잘 보존하는 효율적인 패치 배열 전략을 설계하는 데 기초 자료로 활용될 가능성이 크다.
