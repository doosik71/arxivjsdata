# Lifelong Wandering: A realistic few-shot online continual learning setting

Mayank Lunayach, James Seale Smith, Zsolt Kira (2022)

## 🧩 Problem to Solve

본 논문은 로보틱스와 같은 실제 응용 분야에서 발생할 수 있는 현실적인 온라인 지속 학습(Online Continual Learning) 시나리오를 제안하고 분석한다. 기존의 Online Few-Shot Learning 연구들은 주로 단일 실내 환경에서 데이터 스트림을 통해 학습하며 인스턴스 분류(Instance Classification) 성능에 집중하였다. 그러나 실제 자율 주행 에이전트는 여러 건물과 환경을 이동하며 끊임없이 새로운 데이터를 접하게 되며, 이 과정에서 데이터 분포의 급격한 변화(Distribution Shift)가 발생한다.

특히, 새로운 지식을 습득하는 과정에서 과거에 학습한 지식을 잃어버리는 Catastrophic Forgetting(치명적 망각) 문제는 매우 중요하지만, 기존의 Few-Shot Online Learning 설정에서는 이를 명시적으로 다루지 않았다. 따라서 본 논문의 목표는 여러 실내 환경을 이동하며 객체 분류(Object Category Classification)를 수행하는 'Continual Wandering'이라는 새로운 벤치마크 설정을 도입하고, 온라인 학습 성능과 망각 사이의 트레이드오프(Trade-off)를 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Continual Wandering 설정 제안**: 에이전트가 여러 환경을 이동하며 데이터를 순차적으로 학습하는 자연스러운 온라인 지속 학습 설정을 도입하였다. 이는 단순한 데이터 스트림을 넘어 환경 간의 분포 변화와 Catastrophic Forgetting을 명시적으로 측정한다.
2. **현실적인 베이스라인 구축**: Prototypical 방법론과 End-to-End 방법론을 모두 포함하여, 제안된 도전적인 설정에 적합하도록 수정된 베이스라인 모델들을 제시하였다.
3. **실험적 분석 및 트레이드오프 발견**: 온라인 학습 성능(Online Performance)과 망각(Forgetting) 사이에 상충 관계가 존재함을 실증적으로 입증하였으며, 이를 통해 향후 연구 방향성을 제시하였다.

## 📎 Related Works

본 논문은 특히 `Wandering Within a World (WW)` 연구를 계승하고 확장한다. WW는 실내 이미지 데이터셋인 RoamingRooms를 사용하여 에이전트가 세계를 배회하는 설정을 제안하였으나, 본 논문은 다음과 같은 차별점을 가진다.

- **분류 대상의 확장**: WW가 동일 객체의 다른 시점들을 구분하는 인스턴스 분류(Instance Classification)에 집중했다면, 본 연구는 객체 카테고리 분류(Object Category Classification)를 수행한다.
- **망각의 명시적 측정**: WW에서는 정확도 하락이 단순히 클래스 수 증가로 인한 난이도 상승인지, 아니면 Catastrophic Forgetting 때문인지 구분하지 않았다. 본 논문은 이를 명시적으로 측정하는 평가 지표를 도입하였다.
- **지속 학습 방법론의 벤치마크**: 기존 WW가 Few-Shot Learning 베이스라인 위주로 비교했다면, 본 논문은 지속 학습(Continual Learning) 관점의 방법론들을 포괄적으로 벤치마크한다.

## 🛠️ Methodology

### 전체 시스템 구조 및 설정
에이전트는 한 에피소드 내에서 $N$개의 환경($E_1, \dots, E_N$)을 순차적으로 탐색한다. 각 환경은 서로 다른 건물(Building)에서 추출된 구역(예: 주방, 침실 등)으로 구성된다. 클래스는 환경에 따라 새롭게 등장하거나, 사라지거나, 유지될 수 있으며, 유지되더라도 환경이 다르면 객체의 인스턴스와 맥락(Context)이 달라져 자연스러운 분포 변화가 발생한다.

데이터는 $\text{batch size} = 1$로 처리되며, 모델은 샘플을 한 번 본 뒤 즉시 폐기하는 온라인 방식을 따른다. 입력 데이터 $x_t$는 4채널 텐서로, RGB 이미지 3채널과 현재 인스턴스를 배경에서 분리한 이진 마스크(Binary Mask) 1채널로 구성된다.

### 평가 지표
본 연구는 학습과 평가가 동시에 이루어지는 에피소드 프레임워크를 사용하며, 다음 두 가지 지표를 핵심으로 정의한다.

1. **Average Online Accuracy ($O_{avg}$)**: 에이전트가 데이터를 순차적으로 보면서 예측하는 정확도의 평균이다.
   $$O_{avg} = \frac{\sum_{i=1}^{N} O_i}{N}$$
   여기서 $O_i$는 환경 $i$에서의 온라인 정확도이며, 모델이 이전에 본 클래스($\text{seen class}$)에 대해 예측한 값과 실제 라벨의 일치 여부를 계산한다.

2. **Average Forgetting ($F_{avg}$)**: 특정 환경 $j$를 학습한 후, 이후 환경 $i$를 학습했을 때 환경 $j$에 대한 성능이 얼마나 감소했는지를 측정한다.
   $$\text{Forgetting of env } j \text{ after env } i: \text{FFF}_{i,j} = C_{j,j} - C_{i,j}$$
   $$\text{Average forgetting after env } i: \text{FF}_i = \frac{\sum_{j=1}^{i-1} \text{FFF}_{i,j}}{i-1}$$
   $$\text{Final Average Forgetting}: F_{avg} = \frac{\sum_{i=1}^{N} \text{FF}_i}{N}$$
   여기서 $C_{i,j}$는 환경 $i$를 본 후 환경 $j$에서 측정한 정확도이다.

### 베이스라인 모델 및 학습 절차
모든 모델은 $\text{ResNet-12}$를 특성 추출기(Feature Extractor)로 사용하며, 평가 시 특성 추출기는 동결(Frozen)된다.

- **OAP (Online Averaged Prototypes)**: 단순한 온라인 평균 방식으로 프로토타입을 업데이트한다. 클래스 $c$의 프로토타입 $P_{c,t}$는 다음과 같이 갱신된다.
  $$\text{count}_{c,t} = \text{count}_{c,t-1} + 1$$
  $$P_{c,t} = \frac{f_{c,t} + \text{count}_{c,t-1} \times P_{c,t-1}}{\text{count}_{c,t}}$$
- **Proto-OML**: Online-aware Meta Learning(OML)을 프로토타입 기반으로 확장한 모델이다. 각 환경이 끝날 때마다 이전에 본 모든 환경에 대해 손실 함수를 계산하여 망각을 방지한다.
  $$\mathcal{L}_{\text{Proto-OML}} = \sum_{i=2}^{N} \mathcal{L}_i$$
  최종 손실 함수는 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CPM}} + \lambda \mathcal{L}_{\text{Proto-OML}}$ 형태로 정의된다.
- **LwF (Learning without Forgetting)**: 지식 증류(Knowledge Distillation)를 사용하여 이전 작업의 지식을 보존한다. 특성 추출기를 동결하고 최종 분류 레이어만 재학습한다.
- **CPM (Contextual Prototypical Memory)**: RNN을 사용하여 시공간적 맥락(Spatiotemporal Context)을 모델링하고 프로토타입 업데이트에 반영한다.

## 📊 Results

실험은 RoamingRooms 데이터셋을 사용하였으며, 에피소드당 환경 수 $N=4$, 환경당 프레임 수 $T=100$으로 설정되었다.

### 정량적 결과
| Method | $O_{avg} (\uparrow)$ | $F_{avg} (\downarrow)$ |
| :--- | :---: | :---: |
| Upper bound | $81.25 \pm 9.66$ | - |
| Base | $66.76 \pm 5.85$ | $22.48 \pm 8.27$ |
| LwF | $56.03 \pm 6.79$ | $2.38 \pm 4.82$ |
| Proto-OML | $62.75 \pm 6.06$ | $8.41 \pm 5.04$ |
| CPM | $73.26 \pm 5.11$ | $31.11 \pm 7.65$ |
| OAP | $69.23 \pm 5.57$ | $5.91 \pm 3.39$ |

### 결과 분석
- **온라인 성능 vs 망각의 상충**: $O_{avg}$가 높은 모델은 대체로 $F_{avg}$ 또한 높게 나타났다. 예를 들어, CPM은 가장 높은 온라인 정확도를 보였으나 망각 또한 가장 심했다.
- **LwF의 한계**: LwF는 망각을 억제하는 데는 매우 효과적($F_{avg} = 2.38$)이었으나, 온라인 학습 성능이 가장 낮게 나타났다. 이는 새로운 정보를 학습하는 능력(Plasticity)을 희생하여 망각을 방지했음을 시사한다.
- **OAP의 효율성**: 단순한 평균 방식인 OAP는 CPM보다 온라인 성능은 약간 낮지만, 망각은 현저히 적어 전반적으로 균형 잡힌 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 온라인 지속 학습에서 발생하는 **Plasticity-Stability Dilemma(가소성-안정성 딜레마)**를 명확히 보여준다. 새로운 환경의 데이터를 빠르게 학습하여 성능을 높이려는 시도(Plasticity)는 필연적으로 이전 환경의 기억을 지우는 망각으로 이어지며, 반대로 망각을 극도로 억제하려는 시도(Stability)는 새로운 지식의 습득 속도를 늦춘다.

특히 흥미로운 점은, 특정 한 쪽의 지표(온라인 정확도 또는 망각 방지)에 최적화된 "극단적인" 방법론들이 통합적인 관점에서는 오히려 성능이 떨어진다는 것이다. 이는 현실적인 로봇 학습 시스템을 구축하기 위해서는 단순히 망각을 줄이거나 빠르게 배우는 것이 아니라, 두 지표 사이의 최적의 균형점을 찾는 것이 핵심임을 시사한다.

본 연구의 한계로는 특성 추출기를 동결시킨 채 분류기나 프로토타입만 업데이트했다는 점이 있으며, 향후 전체 네트워크를 효율적으로 업데이트하면서도 망각을 방지하는 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 로봇의 환경 이동 상황을 모사한 **Continual Wandering**이라는 새로운 온라인 지속 학습 벤치마크를 제안하였다. 실험 결과, 새로운 지식을 빠르게 배우는 성능(Online Accuracy)과 과거 지식을 유지하는 능력(Forgetting) 사이에 강한 트레이드오프가 존재함을 확인하였다. 단순한 프로토타입 평균 방식(OAP)이 의외로 효율적인 균형을 보여주었으며, 이는 향후 가소성과 안정성을 동시에 확보하는 지속 학습 알고리즘 연구의 필요성을 강조한다.