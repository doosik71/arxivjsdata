# On Evolving Attention Towards Domain Adaptation

Kekai Sheng, Ke Li, Xiawu Zheng, Jian Liang, Weiming Dong, Feiyue Huang, Rongrong Ji, Xing Sun (2021)

## 🧩 Problem to Solve

본 논문은 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)에서 성능을 향상시키기 위해 attention 모듈의 구성(configuration), 즉 attention의 종류와 배치 위치를 최적화하는 문제를 다룬다. 

기존의 UDA 연구들은 공간적(spatial) 또는 채널(channel) attention 메커니즘을 도입하여 부정적 전이(negative transfer)를 완화하고 도메인 특화 특징을 강화함으로써 성능을 높여왔다. 그러나 이러한 attention 모듈들은 주로 연구자의 직관에 따라 수동으로 설계(handcrafted)되었으며, 이는 실제 적용 환경에서 최적의 솔루션이 아닐 가능성이 크다.

특히, 신경망 구조 탐색(Neural Architecture Search, NAS)을 통해 이를 자동화하려는 시도가 있었으나, 기존 NAS 알고리즘들은 전이 학습(transfer learning)의 특성을 고려하지 않아 큰 도메인 변화(domain shift)에 취약하며, 타겟 도메인에 레이블이 없는 UDA 상황에서 탐색된 구조의 성능을 어떻게 평가할 것인가에 대한 근본적인 어려움이 존재한다. 따라서 본 논문의 목표는 인간의 개입 없이 특정 UDA 태스크에 최적화된 attention 구성을 자동으로 찾아내는 EvoADA 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 UDA의 특성인 '전이 가능성(transferability)'과 '판별력(discrimination)'을 동시에 고려하여 attention의 구성(종류와 위치)을 진화적으로 탐색하는 것이다.

주요 기여 사항은 다음과 같다.
1. **UDA 특화 Search Space 설계**: 단순한 CNN 연산 조합이 아니라, SE, GSoP, CBAM 등 검증된 다양한 attention 모듈의 종류와 백본 네트워크 내의 배치 위치를 포함하는 새로운 탐색 공간을 정의하였다.
2. **UDA 지향적 평가 전략**: 타겟 도메인에 정답 레이블이 없는 한계를 극복하기 위해, 의사 레이블(pseudo-label)의 판별력을 측정하는 새로운 평가 지표를 제안하여 탐색 과정을 가이드한다.
3. **EvoADA 프레임워크**: 진화 알고리즘(Evolutionary Algorithm)을 기반으로 최적의 attention 구성을 효율적으로 탐색하는 파이프라인을 구축하였으며, 이를 통해 다양한 SOTA UDA 방법론들의 성능을 일관되게 향상시킴을 입증하였다.

## 📎 Related Works

### Unsupervised Domain Adaptation (UDA)
UDA는 레이블이 있는 소스 도메인의 지식을 활용해 레이블이 없는 타겟 도메인의 학습을 돕는 기술이다. 기존 연구들은 크게 세 가지 방향으로 발전해 왔다: 1) 특징 분리(feature disentanglement), 2) 도메인 정렬(domain alignment), 3) 판별력 강화(discrimination-aware) 방법론이다. 최근에는 TADA, CADA, DCAN과 같이 attention 모듈을 결합한 방식들이 제안되었으나, 이들은 모두 수동으로 설계된 모듈에 의존한다는 한계가 있다.

### Neural Architecture Search (NAS)
NAS는 주어진 태스크에 최적화된 네트워크 구조를 자동으로 찾는 기술로, 강화 학습, 베이지안 최적화, 진화 알고리즘 등 다양한 접근법이 존재한다. 본 논문은 기존 NAS가 도메인 전이 상황을 고려하지 않는다는 점을 지적하며, 특히 AdaptNAS나 ABAS와 같은 유사 연구와 차별화된다. EvoADA는 단순한 셀 구조나 보조 브랜치 변경이 아니라, 백본 전체에 걸친 다양한 attention 모듈의 최적 조합을 찾는다는 점에서 더 일반적이고 포괄적인 접근 방식을 취한다.

## 🛠️ Methodology

### 1. Search Space for Diverse Attention Configuration
EvoADA는 attention의 **종류(Type)**와 **위치(Position)**라는 두 가지 관점에서 탐색 공간 $\mathcal{A}$를 정의한다.

- **Type**: 공간적 attention과 채널 attention을 모두 고려한다. 구체적으로 SE, GSoP, CBAM 모듈과 Identity(attention 없음) 중에서 선택하며, 각 모듈의 하이퍼파라미터인 중간 채널 수(`#channel` $\in \{256, 512, 1024, 2048\}$)와 그룹 수(`#group` $\in \{1, 2, 4, 8\}$)를 조합한다. 가능한 모듈의 수는 $(3 \times 4 \times 4 + 1) = 49$가지이다.
- **Position**: 백본 네트워크(예: ResNet-50)의 중간 레이어 $L$개 중 어디에 배치할지를 결정한다. 저수준 특징(low-level features)은 전이 가능성이 높으므로, 탐색 효율을 위해 더 깊은 $L/2$개 레이어만을 대상으로 하며, 각 레이어에는 하나의 attention 모듈만 배치한다.

최종적인 attention 구성 파라미터 $\alpha$는 다음과 같이 정의된다:
$$\alpha = [\alpha_1, \alpha_2, \dots, \alpha_{L/2}]$$
여기서 $\alpha_i$는 $i$번째 레이어의 attention 구성을 나타낸다.

### 2. UDA-oriented Evaluation Strategy
타겟 도메인에 레이블이 없으므로, 본 논문은 정보 최대화(information maximization) 원리를 이용한 pseudo-label 판별력 측정 방식을 제안한다. 전체 최적화 문제는 다음과 같은 이단계 최적화(bi-level optimization) 형태로 정식화된다:

$$\alpha = \arg \min_{\alpha \in \mathcal{A}} \cdot \mathcal{L}_{PE}^T(F(x; \alpha, \theta^*(\alpha)))$$
$$\text{s.t. } \theta^*(\alpha) = \arg \min_{\theta} \mathcal{L}_{DA}^{S+T}(y, F(x; \alpha, \theta))$$

여기서 $\mathcal{L}_{DA}^{S+T}$는 CDAN이나 SHOT과 같은 일반적인 UDA 손실 함수이며, $\mathcal{L}_{PE}^T$는 타겟 도메인에서의 판별력을 측정하는 평가 함수로 다음과 같이 구성된다:
$$\mathcal{L}_{PE}^T = \mathcal{L}_{ent}^T + \mathcal{L}_{div}^T + \mathcal{L}_{pse}^T$$
- $\mathcal{L}_{ent}^T$: 각 출력 예측의 정보 엔트로피(Entropy)로, 예측의 확신도를 측정한다.
- $\mathcal{L}_{div}^T$: 타겟 도메인 예측값들의 다양성을 측정하여 특정 클래스로 쏠리는 현상을 방지한다.
- $\mathcal{L}_{pse}^T$: self-training을 통해 생성된 의사 레이블($\hat{y}$)에 기반한 교차 엔트로피 손실이다.

### 3. Overall Search Algorithm (EvoADA)
탐색 과정은 진화 알고리즘을 기반으로 하며, 다음과 같은 절차로 진행된다:
1. **Seed Initialization**: 탐색 공간에서 $K$개의 초기 attention 구성(seeds)을 무작위로 샘플링한다.
2. **Inference**: 각 seed에 대해 병렬적으로 네트워크 가중치를 학습시키고, 위에서 정의한 $\mathcal{L}_{PE}^T$를 통해 성능을 평가한다.
3. **Crossover & Mutation**: 성능이 우수한 상위 seed들은 교차(crossover)시켜 더 나은 조합을 찾고, 성능이 낮은 하위 seed들은 돌연변이(mutation)를 일으키거나 새로운 seed로 교체하여 탐색 범위를 넓힌다.
4. **Early Stop**: 소스 도메인 정확도가 너무 높거나(negative transfer 의심), 타겟 도메인의 pseudo-label 정확도가 지속적으로 낮을 경우 해당 seed의 학습을 조기에 종료하여 효율성을 높인다.

## 📊 Results

### 실험 설정
- **데이터셋 및 태스크**: Office-Home, Office-31 (Closed-set, PDA, ODA), CUB-Paintings (FGDA), Duke-Market-1510 (Person Re-ID).
- **백본**: ResNet-50.
- **기준선(Baselines)**: SHOT, PAN, MMT 등 각 분야의 SOTA 방법론.
- **지표**: 분류 정확도(Accuracy, %) 및 mAP.

### 주요 결과
1. **일반적 성능 향상**: EvoADA를 통해 탐색된 attention 구성은 다양한 UDA 시나리오에서 기준 모델의 성능을 일관되게 높였다.
    - **Office-Home (SHOT 기반)**: Closed-set UDA (+2.3%), PDA (+1.9%), ODA (+3.5%)의 평균 정확도 상승을 보였다.
    - **FGDA (PAN 기반)**: CUB-Paintings 태스크에서 평균 4.1%의 큰 성능 이득을 얻었다.
    - **Person Re-ID (MMT 기반)**: Market-1501 $\rightarrow$ Duke 등의 태스크에서 ResNet-50이나 IBN-Net-50보다 높은 mAP를 달성하였다.
2. **탐색 공간의 유효성**: NASNet이나 DARTS와 같은 일반적인 CNN 연산 기반 search space보다, 본 논문이 제안한 attention 기반 search space가 UDA 태스크에서 더 높은 성능을 보였다 (평균 74.6% vs 72.6%/72.3%/67.3%).
3. **평가 전략의 신뢰성**: 소스 도메인 정확도보다 제안된 평가 프로토콜($\mathcal{L}_{PE}^T$)이 타겟 도메인의 최종 정확도와 훨씬 높은 순위 상관관계(Spearman $\rho$)를 가짐을 확인하였다 (Office-Home 기준 0.54 vs 0.23).

## 🧠 Insights & Discussion

### 강점 및 분석
- **자동화의 이점**: 수동 설계된 attention 모듈보다 데이터와 도메인 특성에 맞게 자동으로 구성된 모듈이 더 강력함을 입증하였다.
- **위치의 중요성**: 분석 결과, ResNet-50의 Layer 3와 Layer 4에 attention 모듈을 배치하는 것이 성능 향상에 가장 효과적이었다. 이는 너무 낮은 층보다 중간-상위 층의 특징이 도메인 전이 시 판별력을 높이는 데 유리함을 시사한다.
- **효율성**: 무작위 탐색(Random Search)보다 훨씬 빠르게 최적의 구성에 도달하며, 적은 수의 파라미터 추가만으로도 상당한 성능 향상을 이끌어냈다.

### 한계 및 논의사항
- **탐색 비용**: 한 태스크당 평균 20시간(V100 GPU 8장 기준)의 탐색 시간이 소요된다. 비록 NAS 치고는 효율적이지만, 모든 새로운 도메인 쌍에 대해 이 과정을 반복하는 것은 여전히 부담이 될 수 있다.
- **가정**: 본 연구는 attention 구성의 최적화에 집중하였으며, 백본 네트워크 자체의 구조 변경이나 더 복잡한 하이퍼파라미터 최적화는 고려하지 않았다.

## 📌 TL;DR

본 논문은 UDA에서 수동으로 설계된 attention 모듈의 한계를 극복하기 위해, **attention의 종류와 위치를 자동으로 최적화하는 EvoADA 프레임워크**를 제안한다. 타겟 도메인의 레이블 부재 문제를 해결하기 위해 **의사 레이블 판별력 기반의 평가 지표**를 도입하고, 진화 알고리즘을 통해 최적의 구성을 탐색한다. 실험 결과, 이 방법은 Closed-set, Partial, Open-set UDA는 물론 Fine-grained 분류 및 Person Re-ID 등 다양한 도메인 적응 태스크에서 SOTA 모델들의 성능을 일관되게 향상시켰으며, 특히 백본의 중간-상위 레이어에 attention을 배치하는 것이 효과적임을 밝혀냈다. 이 연구는 향후 도메인 적응을 위한 효율적인 네트워크 설계 자동화에 중요한 기초를 제공한다.