# SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning

Zhi Chen, Zecheng Zhao, Jingcai Guo, Jingjing Li, Zi Huang (2025)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Learning (ZSL)에서 발생하는 **Semantic Misalignment (의미론적 불일치)** 문제를 해결하고자 한다. ZSL의 핵심은 학습 단계에서 본 적 없는 클래스를 인식하기 위해 속성(Attribute)과 같은 클래스 수준의 시맨틱 기술자를 활용하는 것이다.

그러나 실제 이미지에는 속성 벡터에 기술되지 않은 배경 소음(Background clutter), 조명 변화, 주변 객체 등 시맨틱과 무관한 정보(Semantic-unrelated information)가 포함되어 있다. 이러한 정보는 시각적 특징 추출 과정에서 핵심적인 시맨틱 단서를 희석시키며, 결과적으로 시각적 특징과 시맨틱 공간 사이의 정렬을 방해하여 미학습 클래스에 대한 인식 성능을 저하시킨다.

기존 연구들은 특징 공간(Feature space)에서 특징을 정제하거나, 모델 공간(Model space)에서 불필요한 토큰을 가지치기(Pruning)하는 사후 처리 방식(Post hoc)을 사용했다. 하지만 이러한 방식은 이미 얽힌(Entangled) 특징에서 정보를 제거하는 것이므로, 불필요한 정보가 학습된 표현에 영향을 주는 것을 완전히 막지 못한다는 한계가 있다. 따라서 본 논문의 목표는 **입력 단계(Input stage)에서 시맨틱 무관 패치를 미리 제거하거나 재구성**함으로써, 불필요한 정보가 네트워크 전체로 전파되는 것을 원천적으로 방지하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Vision Transformer (ViT) 구조를 기반으로, 입력 단계에서 시맨틱 무관 패치를 식별하고 이를 시맨틱 정보가 풍부한 학습 가능한 임베딩으로 교체하는 **SVIP (Semantically Contextualized Visual Patches)** 프레임워크를 제안하는 것이다.

1. **Self-Supervised Patch Selection (SSPS)**: 모든 Transformer 블록의 Self-attention 가중치를 집계하여 각 패치의 중요도를 추정하고, 이를 통해 입력 공간에서 시맨틱 무관 패치를 식별하는 이진 분류기를 학습시킨다.
2. **Patch Semantic Contextualization (PSC)**: 식별된 시맨틱 무관 패치를 단순히 제거하는 대신, 속성 수준의 단어 임베딩(Word embeddings)으로 초기화된 학습 가능한 패치 임베딩으로 대체하여 객체의 구조적 손실을 막고 시맨틱 상호작용을 강화한다.
3. **Attribute Localization**: Class token에만 의존하지 않고, 시맨틱 관련성이 높은 상위 $M$개 패치를 활용하여 속성 값을 직접 국소화(Localization)함으로써 보다 정밀한 시각-시맨틱 정렬을 달성한다.

## 📎 Related Works

ZSL 연구는 크게 생성 기반 방법(Generative methods)과 임베딩 기반 방법(Embedding-based methods)으로 나뉜다. 생성 기반 방법은 GAN이나 VAE를 통해 미학습 클래스의 가상 시각 특징을 합성하고, 임베딩 기반 방법은 시각 공간과 시맨틱 공간 사이의 직접적인 매핑을 학습한다.

최근에는 Semantic Misalignment를 해결하기 위해 특징 공간에서 불필요한 정보를 분리(Disentangling)하는 연구들(예: RFF, SDGZSL)이 진행되었으며, ViT를 도입하여 모델 공간에서 시맨틱 무관 토큰을 제거하는 ZSLViT와 같은 연구가 등장했다. 그러나 이러한 방법들은 특징이 이미 추출된 이후에 처리하는 방식이므로, 입력 단계에서부터 노이즈를 제어하여 학습된 표현의 오염을 막으려는 시도는 부족했다. SVIP는 이러한 기존 접근법과 달리 **입력 공간에서의 조기 개입**을 통해 차별점을 둔다.

## 🛠️ Methodology

### 1. Self-Supervised Patch Selection (SSPS)

입력 이미지 $x$를 $N$개의 패치 $\{P_1, \dots, P_N\}$로 나누고 이를 임베딩 $\{v_1, \dots, v_N\}$으로 투영한다. ViT의 $L$개 Transformer 블록을 통과하며 생성되는 Self-attention 행렬 $T^l \in \mathbb{R}^{(N+1) \times (N+1)}$을 다음과 같이 누적 집계하여 패치의 전역적 중요도를 산출한다.

$$W^l = W^{l-1} + W^{l-1} \times T^l, \quad l=1, \dots, L$$

여기서 $W^1 = T^1$이며, 최종 집계 행렬 $W^L$의 Class token 행(0번 인덱스)을 추출하여 각 패치의 시맨틱 점수 $r_i = W^L[0; i]$를 구한다. 이를 pseudo ground truth로 사용하여, 입력 패치 임베딩 $v_i$로부터 점수 $\hat{r}_i$를 예측하는 이진 분류기 $\text{PatchCls}$를 학습시킨다. 손실 함수로는 Binary Cross Entropy (BCE) Loss를 사용한다.

$$\mathcal{L}_{patch} = -\frac{1}{N} \sum_{i=1}^{N} [r_i \log \hat{r}_i + (1-r_i) \log(1-\hat{r}_i)]$$

### 2. Patch Semantic Contextualization (PSC)

시맨틱 무관 패치를 단순히 제거하면 객체의 구조가 파괴될 수 있다. 이를 방지하기 위해, 속성 단어 임베딩 $\{w_1, \dots, w_K\}$를 Word-to-Patch (W2P) 투영 층을 통해 학습 가능한 임베딩 $e$로 변환한다. 예측 점수 $\hat{r}_i$가 낮거나 상위 $M$개에 들지 못하는 패치들에 대해 다음과 같이 임베딩을 교체(Contextualization)한다.

$$\hat{v}_i = \begin{cases} v_i, & \text{if } i \in S_M \\ v_i + e, & \text{otherwise} \end{cases}$$

이를 통해 입력 단계에서부터 속성 수준의 시맨틱 정보가 주입되어 시각-시맨틱 상호작용이 강화된다.

### 3. Attribute Localization

최종 출력에서 상위 $M$개의 시맨틱 관련 패치 $Z'_L \in \mathbb{R}^{M \times C}$를 추출한다. 이를 Patch-to-Attribute (P2A) 함수를 통해 속성 공간으로 투영한 뒤, Max Pooling을 적용하여 각 속성에 가장 관련이 깊은 패치를 식별한다.

$$\hat{a} = \text{MaxPool}(\text{P2A}(Z'_L))$$

최종 클래스 예측 확률 $p(y|Z_0)$는 예측된 속성 벡터 $\hat{a}$와 실제 속성 벡터 $a^s_y$ 간의 코사인 유사도를 기반으로 SoftMax를 통해 계산된다.

### 4. Model Optimization

모델은 원본 패치 $Z_0$와 컨텍스트화된 패치 $Z'_0$ 두 경로를 모두 통과하며 학습된다. 전체 손실 함수는 다음과 같다.

$$\ell_{overall} = \ell_{cls} + \lambda_1 \ell_{JSD} + \lambda_2 \ell_{patch}$$

여기서 $\ell_{cls}$는 분류 손실이며, $\ell_{JSD}$는 두 경로의 예측 분포 간의 Jensen-Shannon Divergence를 통해 학습의 안정성을 도모한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CUB (조류), AwA2 (동물), SUN (장면)의 세 가지 벤치마크 데이터셋을 사용한다.
- **평가 지표**: Zero-Shot Learning (ZSL)의 Top-1 정확도와 Generalized ZSL (GZSL)의 Seen/Unseen 클래스 정확도 및 조화 평균(Harmonic Mean, $H$)을 측정한다.
- **구현**: ViT-base를 백본으로 사용하며, 효율성을 위해 $2 \times 2$ 패치를 하나로 통합하여 총 49개의 패치를 사용한다.

### 주요 결과

- **성능 향상**: SVIP는 세 데이터셋 모두에서 기존의 ResNet101 기반 방법 및 최신 ViT 기반 ZSL 방법(ZSLViT 등)보다 우수한 성능을 보였다. 특히 CUB 데이터셋의 GZSL $H$ 값에서 75.0%를 기록하며 SOTA 성능을 달성했다.
- **구성 요소 분석 (Ablation Study)**:
  - SSPS(패치 선택)를 제거했을 때 성능 하락이 가장 컸으며, 이는 정밀한 패치 선택의 중요성을 입증한다.
  - PSC(컨텍스트화) 대신 단순히 패치를 제거했을 때 객체 구조 파괴로 인해 성능이 하락했다.
  - JSD 손실과 W2P(단어 임베딩 기반 초기화) 역시 성능 유지에 필수적임이 확인되었다.
- **정성적 분석**: t-SNE 시각화 결과, SVIP가 baseline ViT보다 예측된 속성 벡터들을 더 조밀하고 명확하게 군집화함으로써 시맨틱 모호성을 효과적으로 제거했음을 보여주었다. 또한, 어텐션 맵 시각화를 통해 SSPS가 실제로 시맨틱 무관 영역을 정확히 식별하여 배제하고 있음을 확인했다.

## 🧠 Insights & Discussion

본 논문은 ZSL의 고질적인 문제인 시맨틱 불일치를 해결하기 위해 '사후 정제'가 아닌 '사전 제어'라는 관점을 제시했다는 점에서 매우 강력한 통찰을 제공한다. 특히 단순한 제거가 아니라 시맨틱 프라이어(Semantic Prior)를 주입하는 컨텍스트화(Contextualization) 방식을 도입하여 ViT의 구조적 특성과 시맨틱 정보를 동시에 보존한 점이 돋보인다.

다만, 본 연구는 고정된 속성 벡터와 단어 임베딩에 의존하고 있다. 향후 연구에서는 데이터셋의 특성에 따라 패치 선택 기준과 컨텍스트화 전략을 동적으로 조정하는 적응형(Adaptive) 전략을 도입한다면 더욱 범용적인 성능 향상이 가능할 것으로 판단된다. 또한, 계산 효율성을 위해 패치를 통합(Aggregation)하여 사용했는데, 더 세밀한 패치 단위에서도 동일한 효과가 나타나는지에 대한 분석이 추가된다면 방법론의 완결성이 높아질 것이다.

## 📌 TL;DR

SVIP는 ZSL에서 배경 소음과 같은 시맨틱 무관 정보가 성능을 저해하는 문제를 해결하기 위해, **입력 단계에서 불필요한 패치를 식별(SSPS)하고 이를 속성 기반의 학습 가능한 임베딩으로 대체(PSC)**하는 ViT 기반 프레임워크이다. 이를 통해 시각-시맨틱 정렬을 획기적으로 개선하여 CUB, AwA2, SUN 벤치마크에서 SOTA 성능을 달성했으며, 이는 입력 공간에서의 조기 정제가 ZSL 성능 향상에 결정적인 역할을 함을 시사한다.
