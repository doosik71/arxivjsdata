# Zero-Shot Dual Machine Translation

Lierni Sestorain, Massimiliano Ciaramita, Christian Buck, Thomas Hofmann (2018)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 신경 기계 번역(Neural Machine Translation, NMT) 시스템 구축 시 필수적인 대규모 병렬 코퍼스(Parallel Corpora)의 부족 문제이다. 특히 저자원 언어(Low-resource languages) 쌍의 경우, 고품질의 병렬 데이터를 확보하는 것이 매우 비용이 많이 들거나 사실상 불가능한 경우가 많아 번역 시스템의 가용성이 크게 제한된다.

기존의 Zero-shot 번역 방식은 다국어 모델을 통해 학습하지 않은 언어 쌍 간의 번역을 시도하지만, 지도 학습(Supervised learning) 방식에 비해 성능 격차가 크다는 한계가 있다. 또한, Dual learning 방식은 병렬 데이터 없이 단일 언어 데이터(Monolingual data)만으로 학습이 가능하지만, 초기 탐색을 가이드하기 위해 어느 정도의 초기 병렬 데이터가 필요하며, 이것이 없을 경우 무작위 번역 모델로는 강화 학습의 방향을 잡기 어렵다는 문제가 있다. 따라서 본 논문의 목표는 Zero-shot 메커니즘을 통해 Dual learning의 초기화 문제를 해결함으로써, 병렬 데이터가 전혀 없는 언어 쌍에 대해서도 지도 학습에 근접하는 번역 성능을 달성하는 것이다.

## ✨ Key Contributions

본 연구의 중심 아이디어는 다국어 NMT 아키텍처의 **Zero-shot 능력**과 강화 학습 기반의 **Dual learning**을 결합하는 것이다.

핵심 직관은 다음과 같다. 다국어 NMT 모델은 직접적으로 학습하지 않은 언어 쌍에 대해서도 어느 정도의 번역 능력을 갖춘 Zero-shot 상태가 되는데, 이 상태를 Dual learning의 시작점으로 활용한다면 기존 Dual learning이 요구하던 '초기 약한 번역 모델(Initial weak translation model)'을 위한 병렬 데이터 없이도 강화 학습을 시작할 수 있다는 것이다. 이를 통해 단일 모델 내에서 모든 번역 방향을 학습시키며, 타겟 언어의 단일 언어 코퍼스만을 활용해 번역 품질을 비약적으로 향상시킬 수 있다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 소개한다.

1. **Multilingual NMT & Zero-shot Translation**: Johnson et al. [11]은 단일 인코더-디코더 구조에서 타겟 언어 태그를 사용하여 여러 언어 쌍을 동시에 학습시키는 방식을 제안하였다. 이 방식은 학습 데이터에 없던 언어 쌍 간의 번역(Zero-shot)을 가능하게 하지만, 지도 학습 모델과의 성능 차이가 존재한다.
2. **Dual Learning**: He et al. [8]은 두 언어 간의 번역이 서로 역관계에 있다는 점을 이용해, 번역된 문장을 다시 원래 언어로 되돌리는 재구성(Reconstruction)과 타겟 언어 모델을 통한 유창성(Fluency)을 보상으로 사용하는 강화 학습 기법을 제안하였다. 하지만 이 방식은 초기 모델을 학습시키기 위한 병렬 데이터가 필수적이다.
3. **Unsupervised NMT**: Lample et al. [12, 13] 및 Artetxe et al. [2] 등은 역번역(Back-translation)과 적대적 학습, 공유 잠재 공간(Shared latent space) 등을 이용하여 병렬 데이터 없이 번역을 수행하는 연구를 진행하였다.

본 연구는 기존 Dual learning이 필요로 했던 초기 병렬 데이터를 Zero-shot NMT의 초기 상태로 대체함으로써, 완전한 비지도 학습 환경에서도 Dual learning의 이점을 누릴 수 있도록 차별화하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

본 시스템은 최소 3개 이상의 언어 $\text{X, Y, Z}$를 대상으로 한다. 여기서 $\text{Z-X}$와 $\text{Z-Y}$는 병렬 데이터가 존재하여 학습이 가능하며, $\text{X-Y}$는 병렬 데이터가 없는 타겟 언어 쌍이다. 전체 과정은 다음과 같은 단계로 진행된다.

1. **Zero-shot 기초 모델 학습**: $\text{Z-X}$ 및 $\text{Z-Y}$ 병렬 데이터를 사용하여 다국어 NMT 모델 $\theta$를 학습시킨다. 이 단계에서 모델은 $\text{X} \to \text{Y}$ 및 $\text{Y} \to \text{X}$ 번역 능력을 잠재적으로 획득한다.
2. **언어 모델(LM) 구축**: 각 언어 $\text{X, Y}$에 대해 독립적인 LSTM 기반 언어 모델을 학습시켜, 특정 문장이 해당 언어에서 얼마나 유창한지를 측정하는 확률 $P^X(\cdot), P^Y(\cdot)$를 계산할 수 있게 한다.
3. **Zero-shot Dual Training**: 단일 언어 데이터 $\text{D}^X, \text{D}^Y$를 사용하여 $\theta$를 강화 학습으로 업데이트한다.

### 학습 절차 및 보상 함수

특정 문장 $x$에 대해 다음과 같은 절차로 모델을 업데이트한다.

1. 모델 $P_\theta$를 통해 번역문 $y$를 샘플링한다: $y \sim P_\theta(\cdot|x)$.
2. **유창성 보상(Fluency Reward)**: 타겟 언어 모델을 통해 $r_1 = \log P^Y(y)$를 계산한다.
3. **재구성 보상(Reconstruction Reward)**: 샘플링된 $y$를 다시 $\text{X}$로 번역하여 원래 문장 $x$가 나올 확률을 계산한다: $r_2 = \log P_\theta(x|y)$.
4. **최종 보상(Total Reward)**: 하이퍼파라미터 $\alpha$를 사용하여 두 보상을 결합한다.
    $$R = \alpha r_1 + (1-\alpha) r_2$$
5. **모델 업데이트**: REINFORCE 알고리즘을 사용하여 보상 $R$을 최대화하는 방향으로 $\theta$를 업데이트한다.

### 수학적 세부 사항

기대 보상 $\mathbb{E}_{y|x}[R]$의 기울기(Gradient)는 다음과 같이 계산된다.

$$\nabla_\theta \mathbb{E}_{y|x}[R] = \mathbb{E}_{y|x}[R(y) \nabla_\theta \log P_\theta(y|x) + (1-\alpha) \nabla_\theta \log P_\theta(x|y)]$$

실제 구현에서는 분산을 줄이기 위해 배치(Batch) 내의 평균 보상을 빼주는 베이스라인(Baseline) 기법을 사용하였으며, 샘플링 시 Softmax Temperature를 $0.002$로 낮게 설정하여 결정론적인 성향을 강화하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: UN Parallel Corpus를 사용하였다 (영어, 프랑스어, 스페인어, 러시아어 등).
- **비교 대상**:
  - $\text{NMT-0}$: $\text{en-es, en-fr}$만 학습한 Zero-shot 베이스라인.
  - $\text{NMT-S}$: $\text{NMT-0}$에 $\text{es-fr}$ 병렬 데이터 1만 개를 추가 학습한 모델.
  - $\text{NMT-F}$: $\text{es-fr}$ 병렬 데이터 100만 개를 모두 학습한 완전 지도 학습 모델.
  - $\text{Dual-0}$: $\text{NMT-0}$에서 시작하여 단일 언어 데이터로만 Dual learning을 수행한 모델.
  - $\text{Dual-S}$: $\text{NMT-S}$에서 시작하여 Dual learning을 수행한 모델.

### 주요 결과 (스페인어-프랑스어 $\text{es} \leftrightarrow \text{fr}$)

- **비지도 학습 성능의 비약적 향상**: $\text{NMT-0}$의 BLEU 점수는 $\text{es} \to \text{fr}$ 기준 $10.02$, $\text{fr} \to \text{es}$ 기준 $6.25$로 매우 낮았다. 그러나 $\text{Dual-0}$는 각각 $35.54$, $39.00$을 기록하며 baseline 대비 최대 $32.58$ BLEU 포인트 상승하였다.
- **지도 학습과의 격차 감소**: $\text{Dual-0}$는 완전 지도 학습 모델($\text{NMT-F}$)의 성능($37.67 / 40.85$)에 단 $2.2$ BLEU 포인트 차이로 근접하였다.
- **소량의 데이터 대비 우위**: $\text{Dual-0}$는 1만 개의 병렬 데이터를 사용한 $\text{NMT-S}$보다 더 높은 성능을 보였다.

### 다국어 확장 실험 (러시아어 추가)

러시아어를 포함한 4개 언어 설정에서도 유사한 경향이 관찰되었다. 영어를 포함하지 않는 모든 Zero-shot 방향의 번역 성능이 향상되었으며, 지도 학습 모델 대비 약 $2 \sim 3$ BLEU 포인트 차이 이내로 성능을 회복하였다. 이는 본 방법론이 서로 다른 문자 체계를 가진 언어 확장 시에도 유효함을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 논문은 Zero-shot 번역의 고질적인 문제인 낮은 품질을 Dual learning의 강화 학습 프레임워크로 해결하였다. 특히 기존 Dual learning의 한계였던 '초기 모델을 위한 병렬 데이터 필요성'을 다국어 모델의 Zero-shot 능력으로 극복함으로써 완전한 비지도 학습의 가능성을 제시하였다.

### 한계 및 분석

1. **Catastrophic Forgetting (파괴적 망각)**: $\text{X-Y}$ 쌍을 강화 학습으로 최적화하는 동안, 기존에 학습되었던 $\text{Z-X}, \text{Z-Y}$ 방향의 번역 성능이 저하되는 현상이 관찰되었다. 이는 새로운 목적 함수에 최적화되면서 이전 지식을 잃어버리는 전형적인 신경망의 문제이다.
2. **Zero-shot 모델의 특성**: 정성 분석 결과, $\text{NMT-0}$ 모델은 번역 결과물에 브리지 언어(영어)를 섞어 쓰거나 소스 언어를 그대로 출력하는 경향이 있었다. $\text{Dual-0}$는 이러한 현상을 완전히 제거하고 타겟 언어로만 출력하도록 모델을 유도하였다.
3. **모델 용량과 성능**: 지도 학습 모델이 구문 기반(Phrase-based) 모델보다 성능이 낮은 이유는 학습 데이터 양의 차이뿐만 아니라, 하나의 모델이 여러 언어 방향을 동시에 학습하면서 언어당 할당되는 파라미터 용량이 감소했기 때문으로 추정된다.

## 📌 TL;DR

본 논문은 병렬 데이터가 없는 언어 쌍 간의 번역을 위해 **Zero-shot NMT와 Dual Learning을 결합한 강화 학습 프레임워크**를 제안하였다. 다국어 모델의 Zero-shot 능력을 Dual learning의 시작점으로 활용함으로써 초기 병렬 데이터 없이도 타겟 언어의 단일 언어 코퍼스만으로 학습이 가능하게 하였으며, 그 결과 비지도 학습임에도 불구하고 지도 학습 모델에 근접하는 성능을 달성하였다. 이 연구는 데이터가 부족한 저자원 언어의 기계 번역 시스템 구축에 있어 매우 효율적인 경로를 제시하며, 향후 다국어 임베딩 최적화 기술과 결합될 경우 더 큰 시너지를 낼 것으로 기대된다.
