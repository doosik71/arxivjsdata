# Visual Space Optimization for Zero-shot Learning

Xinsheng Wang, Shanmin Pang, Jihua Zhu, Zhongyu Li, Zhiqiang Tian, and Yaochen Li (2019)

## 🧩 Problem to Solve

본 논문은 훈련 세트에 포함되지 않은 새로운 카테고리를 인식하는 Zero-shot Learning (ZSL)에서 발생하는 시각적 공간의 구조적 문제와 그로 인한 성능 저하 문제를 해결하고자 한다. ZSL 모델은 일반적으로 시각적 특징(visual features)과 시맨틱 묘사(semantic descriptions)를 동일한 임베딩 공간(embedding space)에 투영하여 최근접 이웃 검색(nearest neighbor search)을 통해 클래스를 분류한다.

최근의 연구들은 Hubness problem(고차원 공간에서 일부 데이터가 지나치게 많은 데이터의 최근접 이웃이 되는 현상)을 완화하기 위해 시각적 공간(visual space)을 임베딩 공간으로 사용하는 경향이 있다. 그러나 CNN을 통해 추출된 시각적 특징들은 공간상에서 이산적으로 분포하며, 클래스 내 분산이 클래스 간 분산보다 더 큰 경우가 빈번하여 데이터 구조가 불분명하다는 문제가 있다. 이러한 불분명한 시각적 데이터 구조는 시맨틱 벡터가 시각적 공간에 효과적으로 임베딩되는 것을 방해하며, 결과적으로 ZSL의 인식 성능을 저하시킨다. 따라서 본 논문의 목표는 시각적 공간의 구조를 최적화하여 시맨틱 벡터가 더 정확하게 매핑될 수 있도록 하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시각적 공간을 그대로 사용하는 대신, 이를 최적화하여 클래스 간 구별력을 높이는 것이다. 이를 위해 다음과 같은 두 가지 전략을 제안한다.

1. **Visual Prototype 기반 방법 (VPB):** 각 클래스에 대해 단순히 특징들의 평균값(centroid)을 사용하는 것이 아니라, 역전파(backpropagation)를 통해 학습 가능한 시각적 프로토타입(visual prototype)을 생성한다. 이를 통해 시맨틱 벡터가 수많은 개별 인스턴스 특징이 아닌, 최적화된 하나의 프로토타입에 가깝게 매핑되도록 유도한다.
2. **시각적 특징 구조 최적화 방법 (SRS/BRS):** 시각적 특징과 시맨틱 표현을 공통의 중간 임베딩 공간(intermediate embedding space)으로 투영하며, 이때 시각적 데이터의 구조 자체를 최적화하는 손실 함수를 도입한다. 동일 클래스 내의 거리는 좁히고 클래스 간의 경계는 명확히 하여 데이터 구조를 더욱 변별력 있게 만든다.

## 📎 Related Works

ZSL은 일반적으로 시각적 특징과 시맨틱 벡터를 공통 공간에 매핑하는 임베딩 기반 방식을 사용한다. 임베딩 공간의 선택에 따라 다음과 같은 특성이 있다.

- **Semantic Space:** 클래스당 하나의 벡터로 표현되어 구조가 명확하지만, Hubness problem이 심화되는 단점이 있다.
- **Visual Space:** Hubness problem을 완화할 수 있으나, 앞서 언급한 대로 인스턴스들의 분포가 이산적이고 클래스 내/간 변별력이 낮다는 한계가 있다.
- **Intermediate Space:** 두 공간의 특성을 조정할 수 있어 유연하지만, 여전히 시각적 데이터 구조의 불분명함이라는 문제가 남아 있다.

기존의 많은 연구들이 원래의 시각적 구조를 보존(preserve)하려는 매니폴드 학습(manifold learning)이나 인코더-디코더(encoder-decoder) 구조를 사용했다. 하지만 본 논문은 기존 구조를 유지하는 것이 아니라, 데이터의 분포 자체를 최적화(optimize)하여 변별력을 높여야 한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. ZSL with Visual Prototypes (VPB)

**학습 가능한 시각적 프로토타입 생성**
단순 평균 기반의 센트로이드는 변별력이 낮으므로, 클래스 $i$에 대한 학습 가능한 프로토타입 $z_i$를 정의한다. 시각적 특징 $x_i$와 프로토타입 $z_j$ 사이의 유사도를 내적(inner product)으로 계산하며, Softmax 함수를 통해 예측 확률 $\hat{h}_{i,j}$를 구한다.

$$\hat{h}_{i,j} = \frac{\exp(h_{i,j})}{\sum_{k=1}^{p} \exp(h_{i,k})}$$

이후, 일반적인 교차 엔트로피 손실 함수(cross-entropy loss) $L_{proto}$를 사용하여 프로토타입 $z_i$만을 업데이트한다.

$$L_{proto} = -\sum_{i=1}^{N_s} \sum_{j=1}^{p} s_{i,j} \log(\hat{h}_{i,j})$$

**시맨틱 표현의 임베딩**
학습된 프로토타입 $z_i$가 준비되면, MLP 기반의 함수 $\psi(\cdot)$를 통해 시맨틱 벡터 $y_i^{(s)}$를 해당 프로토타입 $z_i$에 가깝게 매핑한다. 손실 함수 $L_{emb}$는 다음과 같다.

$$L_{emb} = \sum_{i=1}^{p} \| f(W_2 f(W_1 y_i^{(s)})) - z_i \|^2 + \lambda_{emb} (\|W_1\|^2 + \|W_2\|^2)$$

### 2. ZSL with Visual Data Structure Optimization (SRS/BRS)

이 방법은 시각적 특징을 $\phi(\cdot)$로, 시맨틱 표현을 $\psi(\cdot)$로 투영하는 두 개의 MLP 브랜치를 가진 네트워크 구조를 사용한다.

**임베딩 손실 함수 (Embedding Loss)**
두 가지 형태의 랭킹 손실을 고려한다.

- **Simple Ranking Loss ($L_{embs}$):** 매칭된 시각적 특징과 시맨틱 벡터 쌍의 거리를 최소화한다.
- **Bi-directional Ranking Loss ($L_{embb}$):** 시각적 특징 $\to$ 시맨틱 벡터, 시맨틱 벡터 $\to$ 시각적 특징 양방향으로 triplet-wise 제약을 가하며, self-adaptive margin ($m_1, m_2$)을 사용하여 학습의 안정성을 높인다.

**시각적 데이터 구조 최적화 손실 ($L_{opts}$)**
시각적 공간의 변별력을 높이기 위해, 동일 클래스 내의 샘플 간 거리보다 다른 클래스 샘플과의 거리가 더 멀어지도록 강제하는 제약 조건을 추가한다.

$$L_{opts} = \sum_{i,j,k} [m_3 + \|\phi(x_i^{(s)}) - \phi(x_j^{(s)})\|^2 - \|\phi(x_i^{(s)}) - \phi(x_k^{(s)})\|^2]^+$$

여기서 $x_j$는 동일 클래스, $x_k$는 다른 클래스의 샘플이며, $m_3$ 또한 self-adaptive margin으로 계산된다.

**최종 목적 함수**
최종 손실 함수는 임베딩 손실과 구조 최적화 손실의 가중 합으로 구성된다.

- $L_{SRS} = L_{embs} + \lambda_1 L_{opts} + \text{regularization}$
- $L_{BRS} = L_{embb} + \lambda_2 L_{opts} + \text{regularization}$

**추론 과정**
테스트 단계에서는 입력 이미지 $x_i^{(u)}$를 시각적 임베딩 함수 $\phi(\cdot)$에 통과시키고, 모든 unseen 클래스의 시맨틱 벡터 $\psi(y_j^{(u)})$ 중 가장 거리가 가까운 클래스를 선택한다.

$$\hat{y} = \text{argmin}_{y_j^{(u)} \in Y_u} \|\phi(x_i^{(u)}) - \psi(y_j^{(u)})\|^2$$

## 📊 Results

### 실험 설정

- **데이터셋:** AwA1, AwA2 (동물), CUB (새), SUN (장면) 4종의 벤치마크 데이터셋을 사용하였다.
- **특징 추출:** ResNet-101에서 추출된 2048차원 시각적 특징을 사용하였다.
- **평가 지표:** Top-1 Accuracy 및 Generalized ZSL (GZSL) 성능 측정을 위한 Harmonic Mean ($H$)을 사용하였다.

### 주요 결과

- **ZSL 성능:** 제안된 SRS와 BRS 방법은 AwA1, AwA2, SUN 데이터셋에서 기존 SOTA 모델들을 능가하였다. 특히 VPB 방법은 AwA1과 AwA2에서 기존 최고 성능 대비 각각 $3.5\%$, $5.4\%$의 성능 향상을 보이며 압도적인 결과(SOTA)를 달성하였다.
- **GZSL 성능:** VPB 방법은 GZSL에서 매우 놀라운 개선을 보여주었다. AwA1의 경우 Harmonic Mean이 $55.6\%$로, 기존 최고 결과보다 $16.5\%$나 향상되었다. AwA2에서도 $16.8\%$의 큰 폭의 향상이 관찰되었다.
- **Ablation Study:**
  - $L_{opts}$를 제거했을 때 성능이 크게 하락함을 확인하였으며, 특히 클래스 수가 많은 CUB, SUN 데이터셋에서 구조 최적화의 효과가 더 크게 나타났다.
  - 학습 가능한 프로토타입(VPB)이 단순 평균 기반 센트로이드(VCB)보다 GZSL의 일반화 성능이 훨씬 뛰어나며, seen 클래스에 대한 과적합(overfitting) 문제를 효과적으로 완화함을 입증하였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 시각적 공간의 '보존'이 아닌 '최적화'에 집중했다는 점이다. 기존의 임베딩 방식은 시각적 공간의 불분명한 분포를 그대로 수용하여 매핑을 시도했기 때문에, 시맨틱 벡터가 매핑될 명확한 타겟이 없었다. VPB는 '학습 가능한 프로토타입'이라는 명확한 목표 지점을 설정함으로써 매핑의 난이도를 낮추고 효율성을 높였다.

특히 GZSL에서 VPB가 거둔 비약적인 성능 향상은 시사하는 바가 크다. 단순 센트로이드를 사용하면 seen 클래스의 데이터 분포에 과하게 의존하여 unseen 클래스로의 일반화 능력이 떨어지지만, 교차 엔트로피 손실로 학습된 프로토타입은 더 변별력 있는 위치에 자리 잡게 되어 seen-class bias 문제를 효과적으로 해결한다.

다만, Bi-directional ranking loss (BRS)가 Simple ranking loss (SRS)보다 GZSL에서는 우세하지만 일반 ZSL에서는 그 차이가 미미하다는 점은 흥미롭다. 이는 구조 최적화 손실($L_{opts}$)이 이미 시각적 공간의 구조를 충분히 잡아주기 때문에, 추가적인 시맨틱 제약의 영향력이 상대적으로 적게 나타난 것으로 해석할 수 있다.

## 📌 TL;DR

본 논문은 ZSL에서 시각적 특징의 불분명한 분포 문제를 해결하기 위해 **학습 가능한 시각적 프로토타입(VPB)**과 **시각적 구조 최적화 손실($L_{opts}$)**을 제안하였다. 실험 결과, 특히 GZSL 작업에서 기존 SOTA 대비 비약적인 성능 향상을 이루어냈으며, 이는 시각적 공간의 구조 최적화가 ZSL의 일반화 성능을 결정짓는 핵심 요소임을 입증한다. 이 연구는 향후 시각적-시맨틱 공간 간의 매니폴드 정렬 연구에 중요한 기초를 제공할 것으로 보인다.
