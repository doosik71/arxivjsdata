# Meta-Learning Siamese Network for Few-Shot Text Classification

Chengcheng Han, Yuhe Wang, Yingnan Fu, Xiang Li, Minghui Qiu, Ming Gao, and Aoying Zhou (2023)

## 🧩 Problem to Solve

본 논문은 텍스트 분류 작업에서 레이블 데이터가 극도로 부족한 상황을 해결하기 위한 Few-Shot Learning(FSL) 문제를 다룬다. 특히 기존의 대표적인 메타 러닝 방법론인 Prototypical Networks (PROTO)가 가진 세 가지 핵심적인 한계점을 해결하는 것을 목표로 한다.

첫째, PROTO는 샘플링된 Support Set의 평균을 통해 Prototype Vector를 계산하는데, 이때 샘플링 과정의 무작위성(Randomness)으로 인해 추정된 프로토타입이 실제 클래스의 중심을 정확히 대표하지 못해 오분류가 발생할 가능성이 크다. 둘째, Support Set 내의 모든 샘플에 동일한 가중치를 부여함으로써, 쿼리 인스턴스를 예측할 때 개별 샘플이 가지는 중요도의 차이를 반영하지 못한다. 셋째, 메타 태스크(Meta-task)를 구성할 때 단순히 무작위로 샘플링하기 때문에, 분류하기 쉬운 단순한 태스크 위주로 학습이 이루어져 모델의 일반화 성능이 저하되는 문제가 있다.

결과적으로 본 연구는 외부 지식을 활용한 프로토타입 초기화, Siamese Network를 통한 임베딩 정제, 그리고 난이도 기반의 태스크 샘플링 전략을 통해 Few-Shot 텍스트 분류 성능을 향상시키고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 프로토타입 벡터의 생성 방식을 데이터 기반의 추정에서 지식 기반의 정의로 전환하고, 이를 Siamese Network 구조로 최적화하는 것이다.

가장 중심적인 설계는 클래스 이름과 위키피디아(Wikipedia)와 같은 외부 설명 텍스트(External Descriptive Texts)를 활용하여 프로토타입 벡터를 초기화하는 것이다. 이는 Support Set의 무작위성에 의존하지 않는 고정된 기준점을 제공한다. 또한, 단순한 거리 측정에 그치지 않고 Siamese Network를 도입하여 클래스 간 거리는 멀게, 클래스 내 샘플과 프로토타입 간 거리는 가깝게 만드는 임베딩 공간을 학습한다. 마지막으로, 분류하기 어려운 'Hard' 샘플과 클래스들을 우선적으로 학습하도록 유도하는 새로운 태스크 샘플링 전략을 제안하여 모델의 강건성을 높였다.

## 📎 Related Works

Few-Shot 텍스트 분류를 위한 메타 러닝 접근 방식은 크게 세 가지로 분류된다.

1.  **Metric-based methods**: Siamese Network, Matching Network, PROTO 등이 있으며, 쿼리 샘플과 학습 샘플 간의 적절한 거리 척도를 학습하는 방식이다. 본 논문의 Meta-SN 역시 이 범주에 속하며 특히 PROTO를 확장하였다.
2.  **Optimization-based methods**: MAML과 같이 새로운 태스크에 빠르게 적응할 수 있도록 최적화 프로세스 자체를 학습하는 방식이다.
3.  **Model-based methods**: MANNs, Meta networks처럼 숨겨진 특징 공간을 학습하여 엔드-투-엔드로 예측하는 방식이나, 해석력이 부족하고 분포 외(Out-of-distribution) 태스크에 대한 일반화 능력이 떨어진다는 한계가 있다.

기존의 PROTO 확장 연구들(HATT-Proto, LM-ProtoNet 등)은 어텐션 메커니즘이나 손실 함수를 추가하여 성능을 높였으나, 여전히 프로토타입 계산 시의 무작위성 문제와 태스크 구성의 무작위성 문제를 간과하고 있다는 점에서 Meta-SN과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인
Meta-SN은 단어 임베딩 생성, 난이도 기반 태스크 샘플링, 샘플 쌍(Sample Pair) 구성 및 가중치 생성, Siamese Network를 통한 공간 정제, 그리고 MAML 스타일의 파라미터 업데이트 단계로 구성된다.

### 주요 구성 요소 및 절차

**1. Word Representation Layer**
fastText나 BERT와 같은 사전 학습 모델을 사용하여 단어를 $d$-차원 벡터로 표현한다. 각 클래스 $c_j$의 초기 프로토타입 벡터 $f_0(c_j)$는 해당 클래스의 외부 설명 텍스트에 포함된 단어 임베딩들의 평균으로 계산된다.

**2. Task Sampler (난이도 기반 샘플링)**
무작위 샘플링 대신, 분류하기 어려운 태스크에 높은 확률을 부여한다. 클래스 $c_i$와 $c_j$ 사이의 상관관계(근접도)를 나타내는 확률 점수 $p^c_{i,j}$를 다음과 같이 정의한다.
$$p^c_{i,j} = \frac{e^{-dis(f_0(c_i), f_0(c_j))}}{\sum_{k=1}^{|C|} e^{-dis(f_0(c_i), f_0(c_k))}}$$
또한, 클래스 내에서 프로토타입과 멀리 떨어진(분류하기 어려운) 샘플을 선택하기 위한 확률 $p^s_{i,j}$를 정의한다.
$$p^s_{i,j} = \frac{e^{dis(f_0(c_i), f_0(s_j))}}{\sum_{k=1}^{m_i} e^{dis(f_0(c_i), f_0(s_k))}}$$
이 식들을 이용해 서로 거리가 가까운 클래스들을 묶고, 그 안에서 프로토타입과 먼 샘플들을 선택하여 Hard Meta-task를 구성한다.

**3. Weight Generator**
Support Set의 샘플이 쿼리 셋 $Q$와 가까울수록 예측에 더 중요한 정보라고 판단하여 가중치를 부여한다. 샘플 $s_i$와 프로토타입 $\phi_j$ 쌍의 가중치 $w_{\langle s_i, \phi_j \rangle}$는 다음과 같다.
$$w_{\langle s_i, \phi_j \rangle} = \text{softmax} \left[ -\frac{1}{L} \sum_{l=1}^L dis(f_\theta(s_i), f_\theta(q_l)) \right]$$

**4. Siamese Network 및 손실 함수**
두 개의 동일한 서브 네트워크(TextCNN + FC layer)를 사용하여 샘플과 프로토타입을 저차원 공간으로 매핑한다. 이때 Contrastive Loss를 사용하여 클래스 내 거리는 좁히고 클래스 간 거리는 넓힌다.
$$L_c(\theta) = \sum_{i=1}^n w_{\langle x_{il}, x_{ir} \rangle} [y_i dis(f_\theta(x_{il}), f_\theta(x_{ir})) + (1-y_i) \max(0, \delta - dis(f_\theta(x_{il}), f_\theta(x_{ir})))]$$
여기서 $y_i$는 동일 클래스 여부(1 또는 0)이며, $\delta$는 마진(margin)이다.

**5. 학습 절차 (Meta-learning Update)**
MAML의 최적화 전략을 따른다. 
1.  Inner Loop: Contrastive Loss $L_c$를 통해 파라미터를 한 단계 업데이트하여 $\theta'$를 얻는다.
2.  Outer Loop: 업데이트된 $\theta'$를 사용하여 쿼리 인스턴스의 분류 정확도를 측정하는 Cross-Entropy Loss $L_{ce}$를 계산하고, 이를 바탕으로 최종 파라미터 $\theta$를 업데이트한다.
$$L_{ce}(\theta') = \sum_{i=1}^L -\log \left( \frac{e^{-dis(f_{\theta'}(q_i), f_{\theta'}(c_j))}}{\sum_{k=1}^N e^{-dis(f_{\theta'}(q_i), f_{\theta'}(c_k))}} \right)$$

## 📊 Results

### 실험 설정
- **데이터셋**: HuffPost, Amazon, Reuters, 20 News, RCV1, FewRel 등 총 6개의 벤치마크 데이터셋을 사용하였다.
- **비교 대상**: PROTO, MAML, ContrastNet, MLADA, DS-FSL 등 7개의 최신 모델과 비교하였다.
- **측정 지표**: 5-way 1-shot 및 5-way 5-shot 분류 정확도를 측정하였다.

### 주요 결과
- **정량적 성과**: Meta-SN은 모든 데이터셋에서 가장 높은 성능을 기록하였다. fastText 기반 비교에서 1-shot 평균 정확도 69.1%, 5-shot 평균 정확도 85.2%를 달성하여, 차순위 모델인 MLADA보다 각각 3.8%, 2.4% 향상된 결과를 보였다.
- **BERT 기반 성능**: BERT 임베딩을 사용했을 때 Meta-SN은 ContrastNet보다 우수한 성능을 보이며 1-shot 기준 73.6%의 평균 정확도를 기록하였다.
- **Ablation Study**: 외부 지식 기반 프로토타입을 제거($\text{Meta-SN-rpv}$)하거나 가중치 학습을 제거($\text{Meta-SN-ew}$)했을 때 성능이 크게 하락함을 확인하여, 제안한 각 구성 요소의 유효성을 입증하였다.
- **시각화**: t-SNE 결과, Meta-SN이 생성한 임베딩이 PROTO나 Hatt-Proto에 비해 클래스 간 분별력이 훨씬 뛰어나고 명확하게 군집화됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 데이터가 부족한 Few-Shot 상황에서 단순히 주어진 샘플의 통계값(평균)에 의존하는 것이 얼마나 위험한지를 잘 지적하였다. 외부 지식을 통해 '정답에 가까운 기준점'을 먼저 설정하고, 이를 Siamese Network로 정제하는 전략은 매우 실용적인 접근이다.

특히, 학습 과정에서 단순히 무작위로 데이터를 뽑는 것이 아니라, 모델이 어려워할 만한(Hard) 태스크를 집중적으로 학습하게 하는 Task Sampler의 도입은 모델의 일반화 능력을 끌어올리는 핵심 요소로 작용하였다. 

다만, 외부 지식(Wikipedia 등)을 확보할 수 없는 특수 도메인이나 신조어가 많은 데이터셋의 경우, 프로토타입 초기화 단계에서 성능 저하가 발생할 가능성이 있다. 또한, Ablation Study 결과에서 클래스 이름만 사용했을 때($\text{Meta-SN-ln}$)와 상세 설명을 모두 사용했을 때의 성능 차이가 0.7%로 미미했다는 점은, 외부 지식의 '양'보다는 '존재 여부'와 그 이후의 '정제 과정(Siamese Network)'이 더 중요함을 시사한다.

## 📌 TL;DR

본 연구는 Few-Shot 텍스트 분류에서 프로토타입 벡터의 무작위성 문제를 해결하기 위해 **외부 지식 기반 초기화**와 **Siamese Network 정제**, 그리고 **난이도 기반 태스크 샘플링**을 결합한 **Meta-SN**을 제안하였다. 실험 결과, 6개 데이터셋 모두에서 SOTA 성능을 달성하였으며, 특히 데이터 희소성으로 인한 불안정성을 획기적으로 줄였다. 이 연구는 외부 지식을 딥러닝 모델의 초기 가이드로 활용하는 메타 러닝 구조의 효율성을 입증하였으며, 향후 다른 도메인의 Few-Shot 학습 연구에 중요한 기초가 될 것으로 보인다.