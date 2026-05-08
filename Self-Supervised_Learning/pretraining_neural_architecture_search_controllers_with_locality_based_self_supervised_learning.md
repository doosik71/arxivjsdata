# Pretraining Neural Architecture Search Controllers with Locality-based Self-Supervised Learning

Kwanghee Choi, Minyoung Choe, Hyelee Lee (2021)

## 🧩 Problem to Solve

본 논문은 Neural Architecture Search (NAS) 과정에서 발생하는 막대한 계산 비용 문제를 해결하고자 한다. 기존의 Controller 기반 NAS 방식은 네트워크 구조와 그에 따른 성능 쌍(architecture-performance pairs)을 Controller에 학습시켜 더 나은 구조를 추론하도록 유도한다. 그러나 이러한 데이터 쌍을 생성하기 위해서는 각 후보 구조를 실제로 학습시키고 검증해야 하므로 엄청난 시간과 자원이 소모된다.

또한, 기존 방식들은 네트워크 구조 간의 구조적 유사성(structural similarity)을 충분히 활용하지 못한다는 한계가 있다. 이론적으로 그래프가 동형(isomorphic)이거나 구조적으로 유사한 네트워크들은 타겟 태스크에서 비슷한 성능을 보일 가능성이 높음에도 불구하고, 단순히 성능 수치에만 의존하여 임베딩을 학습하는 것은 비효율적이다. 따라서 본 연구의 목표는 구조적 유사성을 활용한 자기지도학습(Self-Supervised Learning) 기반의 사전 학습(Pretraining) 기법을 제안하여 NAS의 효율성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 네트워크 구조의 '국소성(Locality)'을 활용한 자기지도 분류 태스크를 통해 Controller를 사전 학습시키는 것이다.

구조적 유사성을 정의하기 위해 Edit-distance(편집 거리) 개념을 도입하며, 두 구조 간의 Edit-distance가 가까우면 임베딩 공간에서도 가깝게 배치되도록 Metric Learning을 적용한다. 이를 통해 성능 측정 데이터 없이도 네트워크 구조 자체의 특성을 반영한 유의미한 초기 임베딩 공간을 구축할 수 있으며, 이는 이후 NAS 과정에서 더 빠르게 최적의 구조를 찾는 정보 제공적 사전 지식(informative prior)으로 작용한다.

## 📎 Related Works

기존의 NAS 효율화 연구는 크게 세 가지 방향으로 진행되었다. 첫째, 검색 공간을 작게 분해하거나 Weight sharing을 사용하는 One-shot 방식, 둘째, 성능 예측기(performance predictor)를 사용하는 방식, 셋째, 준지도학습(semi-supervised) 접근 방식 등이 있다. 본 논문에서 제안하는 사전 학습 기법은 이러한 기존 효율화 방법론들과 결합하여 함께 사용할 수 있다.

또한, 그래프의 국소성을 활용하는 Local search나 Evolutionary algorithm 등이 존재하지만, 이들은 주로 학습 과정 중에만 국소성을 활용한다. 반면, 본 논문은 이를 '사전 학습' 단계로 끌어올려 Controller의 초기 가중치를 최적화한다는 점에서 차별성을 가진다. 구조 임베딩을 위해 GCN, Adjacency matrix encoding, Path encoding 등이 연구되었으며, 특히 본 연구는 임베딩 간의 거리 함수를 직접 학습하는 Metric Learning 관점에서 접근한다.

## 🛠️ Methodology

### 1. Locality-based Classification Task

연구진은 구조적 유사성을 정량화하기 위해 Edit-distance를 사용한다. 계산 복잡도를 줄이기 위해 이를 분류 문제로 단순화하였다.

- **Positive samples**: 임의의 앵커 그래프(anchor graph)에서 몇 번의 무작위 편집(random edits)을 가해 생성한 구조로, Edit-distance가 6 이하인 경우이다.
- **Negative samples**: 무작위로 선택된 다른 구조들로, 대부분 Edit-distance가 6을 초과한다.

### 2. 통합 프레임워크: Neural Architecture Optimization (NAO)

본 방법론의 유효성을 검증하기 위해 NAO 프레임워크에 적용하였다. NAO는 Encoder $f_e$, Performance predictor $f_p$, Decoder $f_d$로 구성되며, 기본 학습 손실 함수는 다음과 같다.
$$\mathcal{L}_{train} = \lambda \mathcal{L}_p + (1-\lambda) \mathcal{L}_r = \lambda (f_p(f_e(a)) - r)^2 + (1-\lambda) \log P_d(a|f_e(a))$$
여기서 $a$는 네트워크 구조, $r$은 성능, $\mathcal{L}_p$는 성능 예측 손실, $\mathcal{L}_r$은 재구성 손실(reconstruction loss)이다.

### 3. Pretraining Scheme

사전 학습 단계에서는 성능 예측기 $f_p$를 제외한 $f_e$와 $f_d$를 다음의 손실 함수로 학습시킨다.
$$\mathcal{L}_{pretrain} = \mathcal{L}_r + \lambda_e \mathcal{L}_e$$
여기서 $\mathcal{L}_e$는 구조적 유사성을 학습하기 위한 Metric Learning 손실 함수이다. 예시로 사용된 TripletMarginLoss는 다음과 같이 정의된다.
$$\mathcal{L}_e = \max(d(f_e(a), f_e(p)) - d(f_e(a), f_e(n)) + m, 0)$$

- $a$: 앵커 구조 (anchor)
- $p$: 구조적으로 유사한 양성 샘플 (positive)
- $n$: 구조적으로 먼 음성 샘플 (negative)
- $d(\cdot)$: 임베딩 공간에서의 거리 함수
- $m$: 마진(margin) 값

이 과정을 통해 $f_e$는 유사한 구조를 가깝게, 다른 구조를 멀게 임베딩하는 능력을 갖추게 되며, $f_d$는 이 임베딩을 다시 구조로 복원하는 능력을 학습한다.

## 📊 Results

### 실험 설정

- **데이터셋**: NAS-Bench-101 (재현 가능한 NAS 벤치마크)
- **비교 대상**:
  - Baseline: $\mathcal{L}_r$만 사용한 NoMetricLoss, 완전히 무작위로 선택하는 Random.
  - Metric Learning 방법론: MarginLoss, AngularLoss, NPairsLoss, MultiSimilarityLoss, LiftedStructureLoss, TripletMarginLoss, ContrastiveLoss, GenLiftedStructureLoss.
- **지표**: 테스트 정확도(Test Acc.), 최적 성능과의 차이(Test Regret), 전체 데이터셋 내 순위(Ranking), 최적 구조 발견까지의 쿼리 횟수(#Queries).

### 주요 결과

1. **구조적 유사성의 효용성**: UMAP 시각화 결과, Metric loss를 통해 사전 학습된 임베딩 공간에서 양성 샘플들이 더 잘 군집화(clustering)되는 것이 확인되었다.
2. **NAS 성능 향상**: $\mathcal{L}_e$를 추가한 대부분의 방법이 NoMetricLoss보다 효율적으로 최적 구조를 찾아냈다.
    - **MultiSimilarityLoss**: 가장 빠르게 최적 구조에 도달하였다.
    - **MarginLoss, AngularLoss**: 평균 성능 면에서 NoMetricLoss를 일관되게 상회하였다.
3. **손실 함수별 차이**: TripletMarginLoss, ContrastiveLoss, GenLiftedStructureLoss는 예상외로 성능이 낮게 나타났는데, 이는 임베딩 공간을 지나치게 희소(sparse)하게 만들어 유효한 구조를 생성하는 데 방해가 된 것으로 추측된다.
4. **정규화 효과**: $\mathcal{L}_e$를 추가하면 재구성 손실 $\mathcal{L}_r$ 값이 상승하는 경향이 있는데, 이는 $\mathcal{L}_e$가 임베딩 공간을 구조적 유사성에 맞게 제한함으로써 $\mathcal{L}_r$에 대한 일종의 정규화(regularization) 역할을 수행하기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 연구는 NAS Controller의 사전 학습에 구조적 유사성이라는 강력한 Prior를 주입함으로써, 값비싼 성능 측정 과정 없이도 효율적인 탐색이 가능함을 보였다. 특히 Metric Learning의 다양한 손실 함수 중 어떤 것이 NAS 임베딩 공간 구축에 유리한지를 실험적으로 분석했다는 점에 의의가 있다.

다만, 일부 Metric Learning 손실 함수가 오히려 성능을 저하시키는 현상이 발견되었는데, 이에 대한 정확한 이론적 원인은 밝혀지지 않았다. 저자들은 임베딩 공간의 분포 특성(sparsity 등)이 영향을 미쳤을 것이라고 가정하고 있다. 또한, 본 실험이 NAS-Bench-101이라는 고정된 벤치마크에서 수행되었으므로, 실제 대규모의 새로운 검색 공간에서도 동일한 효과가 나타날지에 대해서는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 네트워크 구조 간의 편집 거리(Edit-distance)를 기반으로 한 자기지도학습 태스크를 제안하여, NAS Controller를 효율적으로 사전 학습시키는 방법을 제시한다. 구조적 유사성을 반영한 Metric Learning을 통해 초기 임베딩 공간을 최적화함으로써, 실제 성능 측정 횟수를 줄이면서도 더 빠르게 고성능의 네트워크 구조를 찾을 수 있음을 증명하였다. 이 연구는 향후 계산 비용이 높은 NAS의 실용성을 높이는 데 기여할 수 있을 것으로 기대된다.
