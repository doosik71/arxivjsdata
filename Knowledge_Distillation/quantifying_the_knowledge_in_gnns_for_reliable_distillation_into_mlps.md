# Quantifying the Knowledge in GNNs for Reliable Distillation into MLPs

Lirong Wu, Haitao Lin, Yufei Huang, Stan Z. Li (2023)

## 🧩 Problem to Solve

본 논문은 위상 구조를 인식하는 Graph Neural Networks (GNNs)의 높은 성능과 추론 효율성이 뛰어난 Multi-Layer Perceptrons (MLPs) 사이의 간극을 메우고자 한다. GNN은 인접 노드의 정보를 수집하는 neighborhood-fetching 과정에서 발생하는 데이터 의존성으로 인해 지연 시간(latency)이 커서 실시간 응용 프로그램에 배포하기 어렵다. 반면, MLP는 추론 속도가 매우 빠르지만 GNN에 비해 성능이 낮다는 단점이 있다.

기존의 GNN-to-MLP 지식 증류(Knowledge Distillation, KD) 연구들은 교사 모델(Teacher GNN)의 지식을 학생 모델(Student MLP)로 전달하여 이 문제를 해결하려 했으나, 모든 노드(Knowledge Points)를 동일하게 중요하다고 가정하고 처리했다. 이로 인해 학생 MLP가 교사 GNN만큼 확신을 가지고 예측하지 못하는 **Under-confidence 문제**가 발생하며, 이는 특히 클래스 경계에 위치한 샘플들이 잘못 예측되는 결과로 이어진다. 따라서 본 논문의 목표는 GNN 내의 지식 포인트별 신뢰도를 정량화하고, 이를 통해 신뢰할 수 있는 지식만을 선택적으로 증류하여 MLP의 성능과 예측 확신도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 GNN의 모든 노드가 동일한 가치의 지식을 가지고 있지 않다는 점에 착안하여, **지식의 신뢰도(Knowledge Reliability)를 정량화**하고 이를 기반으로 **신뢰할 수 있는 지식 포인트만을 샘플링하여 추가적인 감독 신호(Supervision)로 활용**하는 것이다.

중심적인 설계 아이디어는 다음과 같다.

1. **Perturbation Invariance 기반의 신뢰도 측정**: 노드 특성에 노이즈를 추가했을 때 정보 엔트로피(Information Entropy)가 얼마나 변하지 않고 유지되는지를 측정하여 해당 노드의 지식 신뢰도를 정의한다.
2. **Knowledge-inspired Reliable Distillation (KRD) 프레임워크**: 정량화된 신뢰도를 바탕으로 학습 가능한 Power Distribution을 사용하여 신뢰할 수 있는 노드를 샘플링하고, 이를 다중 교사(Multi-teacher) 형태로 활용하여 MLP를 학습시킨다.

## 📎 Related Works

본 논문은 지식 증류를 크게 두 가지 방향으로 구분하여 설명한다.

1. **GNN-to-GNN Distillation**: 대규모 GNN을 소규모 GNN으로 증류하는 방식(예: RDD, TinyGNN)이다. 하지만 학생 모델 역시 GNN 구조를 유지하므로 neighborhood-fetching으로 인한 지연 시간 문제를 완전히 해결하지 못한다.
2. **GNN-to-MLP Distillation**: GNN의 위상 인식 능력을 MLP에 이식하여 효율성을 극대화하는 방식이다. 대표적으로 GLNN이 있으며, 이는 교사 GNN과 학생 MLP 간의 KL-divergence를 최소화한다. 또한 RKD-MLP는 메타 정책을 통해 신뢰할 수 없는 soft label을 필터링하는 다운샘플링(down-sampling) 전략을 취한다.

본 연구는 기존의 단순한 전체 노드 증류나 단순 필터링과 달리, 신뢰도가 높은 지식 포인트를 능동적으로 선택하여 추가 감독을 제공하는 **업샘플링(up-sampling) 스타일의 전략**을 취한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 지식 신뢰도 정량화 (Knowledge Quantification)

교사 GNN $f_\theta(\cdot, \cdot)$에서 특정 노드 $v_i$의 지식 신뢰도는 노이즈 섭동(noise perturbation)에 대한 정보 엔트로피의 불변성을 측정하여 정의한다.

$$ \rho_i = \frac{1}{\delta^2} \mathbb{E}_{X' \sim \mathcal{N}(X, \Sigma(\delta))} [H(Y'_i) - H(Y_i)]^2 $$

여기서 $Y = f_\theta(A, X)$는 원래의 예측값, $Y' = f_\theta(A, X')$는 노이즈가 추가된 특성 $X'$에 대한 예측값이며, $H(\cdot)$은 정보 엔트로피를 의미한다. $\delta$는 가우시안 노이즈의 분산이다. 지표 $\rho_i$ 값이 작을수록 해당 노드의 지식은 외부 섭동에 강건하며, 따라서 신뢰도가 높다고 판단한다.

### 2. 샘플링 확률 모델링 (Sampling Probability Modeling)

정량화된 $\rho_i$를 기반으로 노드 $v_i$가 정보 제공 능력이 있고 신뢰할 수 있는 지식 포인트일 확률 $s_i$를 학습 가능한 파라미터 $\alpha$를 가진 Power Distribution으로 모델링한다.

$$ p(s_i | \rho_i, \alpha) = 1 - \left( \frac{\rho_i}{\rho_M} \right)^\alpha $$

여기서 $\rho_M = \arg\max_j \rho_j$이다. 파라미터 $\alpha$는 학습 과정에서 학생 MLP와 교사 GNN의 예측 일치도(True Positive 샘플의 밀도)를 기반으로 히스토그램 피팅을 통해 동적으로 업데이트되며, 모멘텀 업데이트 방식을 따른다.

$$ \alpha^{(t)} \leftarrow \eta \alpha^{(t-1)} + (1-\eta) * \alpha^{(t)}_{new} $$

### 3. KRD 학습 절차 및 손실 함수

KRD는 타겟 노드 $v_i$의 이웃 $\mathcal{N}_i$ 중에서 위에서 정의한 확률 $p$에 따라 신뢰할 수 있는 지식 포인트 $v_j$를 샘플링한다. 이후, 샘플링된 포인트들을 다중 교사로 삼아 다음과 같은 $L_{KRD}$ 손실 함수를 정의한다.

$$ L_{KRD} = \mathbb{E}_i \mathbb{E}_{j \in \mathcal{N}_i, j \sim p} [D_{KL}(\sigma(z^{(L)}_j / \tau), \sigma(h^{(L)}_i / \tau))] $$

최종 전체 손실 함수 $L_{total}$은 레이블 기반의 교차 엔트로피 손실($L_{label}$), 일반적인 GNN-to-MLP 증류 손실($L_{KD}$), 그리고 제안된 KRD 손실의 가중 합으로 구성된다.

$$ L_{total} = \lambda \sum_{i \in V^L} H(y_i, \sigma(z^{(L)}_i)) + (1-\lambda)(L_{KD} + L_{KRD}) $$

## 📊 Results

### 실험 설정

- **데이터셋**: Cora, Citeseer, Pubmed, Coauthor-CS, Coauthor-Physics, Amazon-Photo, ogbn-arxiv 등 7개의 실세계 그래프 데이터셋을 사용하였다.
- **기준선(Baselines)**: Vanilla MLP, GLNN, 그리고 GNN-to-GNN 방식인 RDD, TinyGNN 등과 비교하였다.
- **교사 모델**: GCN, GraphSAGE, GAT의 세 가지 아키텍처를 모두 적용하여 범용성을 검증하였다.

### 주요 결과

1. **분류 성능 향상**: KRD는 모든 데이터셋과 교사 모델 조합에서 GLNN보다 뛰어난 성능을 보였다. 7개 데이터셋 평균 기준, Vanilla MLP 대비 12.62% 향상되었으며, 교사 GNN보다도 평균 2.16% 높은 성능을 기록하였다.
2. **전이/귀납적 설정**: Transductive 설정에서 더 높은 성능 향상을 보였는데, 이는 학습에 사용할 수 있는 노드 특성이 더 많아 신뢰할 수 있는 지식 포인트를 더 많이 확보할 수 있기 때문으로 분석된다.
3. **확신도(Confidence) 개선**: 실험 결과, KRD를 통해 증류된 MLP는 GLNN 대비 예측 확신도가 크게 증가하였으며, 이는 Under-confidence 문제를 효과적으로 해결했음을 시사한다.
4. **효율성**: 샘플링을 전체 노드가 아닌 이웃 노드 범위 내에서 수행함으로써 시간 복잡도를 $O(|V|^2 F)$에서 $O(|E|F)$ 수준으로 낮추어 GCN과 동일한 선형 복잡도를 유지하였다.

## 🧠 Insights & Discussion

### 공간적 및 시간적 분석

- **공간적 분포(Spatial Distribution)**: 시각화 분석 결과, 신뢰도가 높은 지식 포인트들은 주로 클래스의 중심부(Center)에 분포하는 반면, 신뢰도가 낮은 포인트들은 클래스 경계(Boundary)에 분포함을 확인하였다. 이는 경계 지역의 불확실성이 MLP의 성능 저하를 유발한다는 가설을 뒷받침한다.
- **시간적 분포(Temporal Distribution)**: 학습 과정에서 학생 MLP는 신뢰도가 높은 지식 포인트들을 먼저 빠르게 학습하고, 이후 점차 신뢰도가 낮은 포인트들로 학습 범위를 넓혀가는 경향을 보였다.

### 비판적 해석 및 한계

본 논문은 GNN의 지식 신뢰도를 정량화하는 새로운 관점을 제시하고 이를 통해 MLP의 성능을 교사 GNN 이상으로 끌어올렸다는 점에서 매우 고무적이다. 특히, 단순히 데이터를 쳐내는 것이 아니라 '어떤 지식이 더 가치 있는가'를 기준으로 추가 감독 신호를 설계한 점이 주효했다.
다만, 제안된 방법론은 여전히 사전 학습된 교사 GNN의 존재를 전제로 하며, $\lambda$나 $\eta$와 같은 하이퍼파라미터에 대한 민감도가 존재한다. 또한, 더 강력한 표현력을 가진 교사/학생 모델과의 결합 가능성에 대해서는 향한 연구 과제로 남겨두고 있다.

## 📌 TL;DR

본 논문은 GNN에서 MLP로의 지식 증류 시 발생하는 **Under-confidence 문제**를 해결하기 위해, 노이즈 섭동에 대한 불변성을 이용해 **지식의 신뢰도를 정량화**하는 방법을 제안한다. 이를 통해 신뢰도가 높은 노드를 선택적으로 샘플링하여 추가 학습 신호로 사용하는 **KRD(Knowledge-inspired Reliable Distillation)** 프레임워크를 구축하였으며, 실험적으로 MLP의 추론 효율성을 유지하면서도 성능을 교사 GNN 수준 혹은 그 이상으로 향상시킬 수 있음을 증명하였다. 이 연구는 향후 고효율 그래프 추론 시스템 구축 및 모델 압축 연구에 중요한 기여를 할 것으로 보인다.
