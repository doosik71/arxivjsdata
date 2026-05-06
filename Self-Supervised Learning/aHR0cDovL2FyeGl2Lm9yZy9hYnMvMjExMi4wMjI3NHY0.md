# Self-supervised Graph Learning for Occasional Group Recommendation

Bowen Hao, Hongzhi Yin, Cuiping Li, and Hong Chen (2022)

## 🧩 Problem to Solve

본 논문은 추천 시스템의 중요한 분야인 **Occasional Group Recommendation**(일시적 그룹 추천)에서 발생하는 데이터 희소성 문제를 해결하고자 한다. Occasional group은 소수의 구성원이 일시적으로 모여 형성된 그룹으로, 과거에 함께 상호작용한 아이템이 없거나 매우 적은 **Cold-start group**의 특성을 가진다.

이러한 시나리오에서는 그룹 수준의 상호작용 데이터가 극도로 부족하기 때문에, 전통적인 그룹 추천 방법으로는 고품질의 그룹 표현(Group Representation)을 학습하는 것이 불가능하다. 최근 Graph Neural Networks(GNNs)를 활용하여 타겟 그룹의 고차 이웃(High-order neighbors) 정보를 통합함으로써 이 문제를 완화하려는 시도가 있었으나, 여전히 상호작용이 적은 고차 이웃들의 임베딩 품질이 낮아 최종적인 그룹 임베딩의 정확도를 떨어뜨리는 한계가 존재한다. 따라서 본 논문의 목표는 **Self-supervised Learning(SSL)** 기술을 통해 Cold-start 유저, 아이템, 그룹의 임베딩 품질을 명시적으로 강화하는 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Self-supervised Graph Learning (SGG)** 프레임워크를 통해 유저, 아이템, 그룹의 임베딩을 재구성(Reconstruction)하는 Pretext task를 수행하여 Cold-start 노드의 표현력을 높이는 것이다.

주요 기여 사항은 다음과 같다:

1. **Meta-learning 설정 기반의 임베딩 재구성**: 상호작용이 충분한 노드의 임베딩을 Ground-truth로 설정하고, 이웃의 상당 부분을 마스킹하여 Cold-start 상황을 시뮬레이션한 뒤 이를 재구성하는 SSL 태스크를 설계하였다.
2. **Embedding Enhancer 도입**: Self-attention 메커니즘을 활용하여 고차 Cold-start 이웃들의 임베딩 품질을 명시적으로 개선하는 모듈을 제안하였으며, 여기서 생성된 Meta-embedding을 GNN의 각 컨볼루션 단계에 통합하였다.
3. **다양한 GNN 백본과의 호환성**: 특정 GNN 구조에 종속되지 않고 LightGCN, NGCF, GCMC 등 다양한 GNN 모델에 적용하여 성능 향상을 입증하였다.

## 📎 Related Works

그룹 추천 연구는 크게 두 가지 전략으로 나뉜다:

1. **Score Aggregation 전략**: 평균(Average), Least Misery, Maximum Satisfaction 등 미리 정의된 함수를 통해 구성원들의 선호도를 합산하는 방식이다. 그러나 이는 그룹 의사결정의 복잡한 동적 과정을 캡처하지 못해 성능이 불안정하다는 한계가 있다.
2. **Profile Aggregation 전략**: 그룹 구성원의 프로필을 먼저 합산한 뒤 개별 추천 모델에 입력하는 방식이다. 최근에는 Attention 메커니즘을 통해 구성원별 영향력을 학습하는 방식이 제안되었으나, 구성원 중 Cold-start 유저가 포함된 경우 Attention 가중치가 희석되어 편향된 그룹 프로필이 생성되는 문제가 발생한다.

최근에는 GNN 기반 방법론들이 등장하여 고차 협업 신호를 활용함으로써 Cold-start 문제를 해결하려 하였다. 특히 HHGR과 같은 모델은 하이퍼그래프 컨볼루션과 Self-supervised node dropout을 통해 문제를 완화하려 했으나, 본 논문은 이러한 GNN 모델들이 여전히 고차 Cold-start 이웃의 임베딩 품질을 명시적으로 강화하지 못한다는 점을 지적하며 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 그래프 정의

SGG는 다섯 가지 상호작용 그래프를 활용한다:

- **관측된 그래프**: Group-Item ($G^{GI}$), User-Item ($G^{UI}$), Group-User ($G^{GU}$)
- **암시적 그래프**: User-User ($G^{UU}$, 공통 아이템 수 기준), Group-Group ($G^{GG}$, 공통 아이템 수 기준)

### 2. Base GNN for Group Recommendation

SGG의 기반이 되는 GNN 구조는 표현 학습 모듈과 공동 학습 모듈로 구성된다.

- **표현 학습**: 유저 임베딩은 $G^{UI}$와 $G^{UU}$에서, 그룹 임베딩은 $G^{GI}, G^{GU}, G^{GG}$에서 각각 컨볼루션을 수행한 후 Soft-attention을 통해 최종 임베딩 $h_u^L, h_g^L$을 생성한다. 특히 그룹-유저 관계($G^{GU}$)의 경우, 유저들의 임베딩을 먼저 집계(Aggregation)한 후 통합하는 과정을 거친다.
- **공동 학습**: BPR(Bayesian Personalized Ranking) Loss를 사용하여 유저-아이템 및 그룹-아이템의 선호도를 최적화한다.
$$L_{main} = L_g + \lambda L_u$$
여기서 $L_u$와 $L_g$는 각각 유저와 그룹의 BPR 손실 함수이다.

### 3. Embedding Reconstruction with GNN (SSL Task)

Cold-start 노드의 품질을 높이기 위해 다음과 같은 재구성 태스크를 수행한다:

1. 상호작용이 많은 노드를 타겟으로 설정하고, 기존 추천 모델로 학습된 임베딩을 Ground-truth $h_g$로 사용한다.
2. 타겟 노드의 이웃을 대량으로 마스킹하여 Cold-start 상황을 모사한다.
3. 남은 소수의 이웃 정보를 바탕으로 GNN 컨볼루션을 수행하여 예측 임베딩 $h_g^L$을 생성한다.
4. 두 임베딩 간의 코사인 유사도를 최대화하는 방향으로 학습한다:
$$L_R^g : \arg \max_{\Theta_f} \sum_{g} \cos(h_g^L, h_g)$$

### 4. Embedding Enhancer

고차 이웃의 품질 문제를 해결하기 위해 **Self-attention learner** 기반의 Enhancer를 도입한다.

- 이 모듈은 타겟 노드의 1차 이웃 임베딩만을 입력받아 smoothed 임베딩을 생성하고, 이를 평균내어 **Meta-embedding** $\hat{h}$를 산출한다.
- 학습된 Meta-embedding은 GNN의 각 컨볼루션 단계에 다음과 같이 더해져 집계 능력을 강화한다:
$$h_g^{l, GI} = \text{CONV}(\hat{h}_g^{GI}, h_g^{l-1, GI}, h_{N(g^{GI})}^l)$$

### 5. 최종 학습 절차

모델은 Multi-task learning 패러다임을 통해 추천 목표와 SSL 목표를 동시에 최적화한다:
$$L = L_{main} + \lambda_1 L_R + \lambda_2 ||\Theta||_2^2$$

## 📊 Results

### 실험 설정

- **데이터셋**: Weeplaces, CAMRa2011, Douban (상세 통계는 Table 2 참조)
- **비교 대상**: MoSAN, AGREE, SIGR, GroupIM, GAME, HHGR 및 GNN 백본 모델들(LightGCN, NGCF, GCMC)
- **측정 지표**: Recall@20, NDCG@20

### 주요 결과

1. **전체 성능**: SGG(논문 내 $\text{GNN}^*$로 표기)는 모든 데이터셋에서 기존 SOTA 모델인 HHGR을 포함한 모든 베이스라인보다 뛰어난 성능을 보였다. 특히 LightGCN을 백본으로 사용했을 때 가장 높은 성능을 기록하였다 (Table 3).
2. **Cold-start 강건성**: 상호작용 수($n_g, n_u$)와 희소율($c\%$)을 낮추어 Cold-start 상황을 심화시켰을 때, 타 모델들에 비해 SGG의 성능 하락폭이 훨씬 적었으며 개선 효과가 더 뚜렷하게 나타났다 (Figure 3).
3. **Ablation Study**:
   - Basic-GNN(단순 재구성) $\rightarrow$ Meta-GNN(Enhancer 추가) $\rightarrow$ $\text{GNN}^*$(전체 프레임워크) 순으로 성능이 향상되었다.
   - 특히 Meta-GNN의 결과는 Cold-start 이웃의 품질을 개선하는 것이 그룹 표현 학습에 매우 중요하다는 것을 입증한다 (Table 4).
4. **학습 패러다임 분석**: Pre-training 후 Fine-tuning 하는 방식($\text{SGG-P}$)보다 Multi-task learning 방식이 성능과 수렴 속도 면에서 모두 우수함을 확인하였다 (Table 6).

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구는 Cold-start 그룹 추천의 핵심 문제가 단순한 그룹 데이터 부족이 아니라, **그 그룹과 연결된 이웃(유저/아이템)들조차 Cold-start 상태**라는 점을 정확히 짚어냈다. 이를 해결하기 위해 Meta-learning 설정에서 임베딩 재구성 태스크를 정의하고, Self-attention 기반의 Enhancer를 통해 고차 이웃의 품질을 보정함으로써 GNN의 정보 전파(Propagation) 과정에서 발생하는 노이즈를 효과적으로 억제하였다.

### 한계 및 논의사항

- **범용성 부족**: 저자들은 본 모델이 새로운 데이터셋에 바로 적용 가능한 일반적인 Pre-trained 모델은 아니라고 명시하였다. 즉, 각 데이터셋마다 SSL 태스크를 위한 학습 과정이 필요하다.
- **계산 복잡도**: SSL 손실 함수 계산으로 인해 학습 시간이 증가하지만, 마스킹된 그래프 $\hat{G}$의 크기가 원본 그래프 $G$보다 훨씬 작기 때문에 전체적인 시간 복잡도는 vanilla GNN과 동일한 수준($O$ magnitude)으로 유지된다고 주장한다.

## 📌 TL;DR

SGG는 **Cold-start 그룹 추천**에서 발생하는 데이터 희소성 문제를 해결하기 위해, **Meta-learning 기반의 임베딩 재구성(Self-supervised Learning)**과 **Self-attention 기반의 Embedding Enhancer**를 제안한 프레임워크이다. 이를 통해 고차 이웃의 임베딩 품질을 명시적으로 개선함으로써, 데이터가 극도로 부족한 상황에서도 고품질의 그룹 표현을 학습할 수 있게 하였으며, 실험적으로 기존 SOTA 모델들을 능가하는 성능을 보였다. 이 연구는 향후 BERT와 같은 일반적인 추천 시스템용 사전학습 모델 설계에 중요한 영감을 줄 수 있다.
