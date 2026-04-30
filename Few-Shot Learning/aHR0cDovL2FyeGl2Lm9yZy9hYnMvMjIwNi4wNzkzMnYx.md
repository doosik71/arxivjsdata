# Lifelong Wandering: A realistic few-shot online continual learning setting

Mayank Lunayach, James Seale Smith, Zsolt Kira (2022)

## 🧩 Problem to Solve

본 논문은 로보틱스와 같은 실제 응용 분야에서 자율 주행 에이전트가 직면하는 **Online Few-Shot Continual Learning** 문제를 해결하고자 한다. 

기존의 Online Few-Shot Learning 연구들은 주로 단일 실내 환경에서 발생하는 데이터 스트림을 통해 새로운 클래스를 학습하는 것에 집중하였다. 그러나 실제 환경에서 에이전트는 서로 다른 여러 건물이나 방을 이동하며 끊임없이 새로운 데이터를 접하게 되며, 이 과정에서 데이터의 분포가 급격히 변하는 Distribution Shift가 발생한다. 이러한 환경에서는 새로운 지식을 학습하면서 동시에 이전 지식을 잃어버리는 **Catastrophic Forgetting(파괴적 망각)** 문제가 심각하게 나타난다.

따라서 본 논문의 목표는 에이전트가 여러 환경을 돌아다니며 학습하는 보다 현실적인 설정인 'Continual Wandering'을 제안하고, 이 환경에서 온라인 학습 성능과 망각 사이의 트레이드-오프(Trade-off) 관계를 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Continual Wandering 설정 제안**: 단순히 단일 환경이 아니라, 여러 건물과 구역을 이동하며 클래스가 나타나고 사라지며, 동일 클래스라도 환경에 따라 맥락(Context)과 인스턴스가 달라지는 자연스러운 온라인 지속 학습 환경을 정의하였다.
2. **현실적인 벤치마크 구축**: RoamingRooms 데이터셋을 확장하여, 데이터가 한 번에 하나씩 제공되고 사용 후 폐기되는 Online Batch Size = 1 제약 조건을 적용하여 실제 로봇의 학습 환경을 모사하였다.
3. **다양한 베이스라인 분석 및 트레이드-오프 규명**: Prototypical methods, Regularization-based, Meta-learning-based 방법론들을 적용하여, 온라인 학습 성능(Plasticity)과 망각 방지(Stability) 사이의 상충 관계가 존재함을 실험적으로 입증하였다.

## 📎 Related Works

본 논문은 **Wandering Within a World (WW)** 연구를 기반으로 하며, 다음과 같은 차별점을 가진다.

* **태스크의 성격**: WW는 동일 객체의 서로 다른 뷰포인트를 묶는 Instance Classification에 집중한 반면, 본 논문은 객체의 카테고리를 분류하는 Object Category Classification을 다룬다.
* **망각의 측정**: WW는 시간 경과에 따른 정확도 하락을 측정하였으나, 이것이 망각 때문인지 혹은 학습해야 할 클래스 수가 늘어나 난이도가 상승했기 때문인지 명확히 구분하지 않았다. 본 논문은 이전에 방문한 환경을 다시 테스트함으로써 Catastrophic Forgetting을 명시적으로 측정한다.
* **방법론적 범위**: WW는 주로 Few-shot Learning 베이스라인과 비교하였으나, 본 논문은 Continual Learning 방법론들을 포괄적으로 벤치마킹하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 설정
에이전트는 한 에피소드 당 $N$개의 환경 $E_1, E_2, \dots, E_N$을 순차적으로 탐색한다. 각 환경 $E_i$는 $T$개의 프레임으로 구성되며, 입력 데이터 $x_t$는 RGB 이미지 3채널과 현재 인스턴스를 분리하는 Binary Mask 1채널을 포함한 총 4채널 데이터이다. 학습은 semi-supervised 방식으로 진행되어 일부 데이터에만 레이블 $y_t$가 제공된다.

### 주요 구성 요소 및 학습 절차

#### 1. Online Averaged Prototypes (OAP)
가장 단순한 형태의 프로토타입 기반 방법으로, 새로운 레이블된 샘플이 들어올 때마다 다음과 같이 단순 평균을 통해 클래스 프로토타입 $P$를 업데이트한다.
클래스 $c$에 대해 시간 $t-1$까지의 샘플 수 $\text{count}_{c,t-1}$와 프로토타입 $P_{c,t-1}$가 있을 때, 새로운 피처 $f_{c,t}$가 입력되면 다음과 같이 갱신한다.
$$\text{count}_{c,t} = \text{count}_{c,t-1} + 1$$
$$P_{c,t} = \frac{f_{c,t} + \text{count}_{c,t-1} \times P_{c,t-1}}{\text{count}_{c,t}}$$

#### 2. Proto-OML
Online-aware Meta Learning (OML)을 프로토타입 방식에 접목한 모델이다. 각 환경 $E_i$가 끝날 때마다, 현재까지 방문한 모든 환경의 데이터에 대해 다시 forward pass를 수행하여 손실 함수를 계산함으로써 망각을 방지한다.
환경 $i$ 종료 시점의 손실 $\mathcal{L}_i$는 다음과 같다.
$$\mathcal{L}_i = \frac{\sum_{t=1}^{T'} \text{CrossEntropy}(y_t, \hat{y}_t) \cdot \mathbb{1}(x_t \text{ is labelled})}{\sum_{t=1}^{T'} \mathbb{1}(x_t \text{ is labelled})}$$
여기서 $T' = i \times T$이며, 최종 손실 함수는 다음과 같이 정의된다.
$$\mathcal{L}_{total} = \mathcal{L}_{CPM} + \lambda \sum_{i=2}^{N} \mathcal{L}_i$$

#### 3. 기타 베이스라인
* **CPM (Contextual Prototypical Memory)**: RNN을 사용하여 시공간적 맥락(spatiotemporal context)을 모델링하고 이를 통해 프로토타입 업데이트 가중치를 조절한다.
* **LwF (Learning without Forgetting)**: Knowledge Distillation을 사용하여 이전 태스크의 지식을 보존하는 정규화 기반 방법이다.
* **Base**: 고정된 Feature Extractor 뒤에 단순 선형 분류기(Linear Layer)만 추가하여 학습한다.

### 평가 지표
* **Average Online Accuracy ($O_{avg}$)**: 에이전트가 각 환경을 지나며 실시간으로 예측하는 정확도의 평균이다.
* **Average Forgetting ($F_{avg}$)**: 환경 $j$를 학습한 후, 이후 환경 $i$를 학습하고 나서 다시 환경 $j$를 테스트했을 때 발생하는 정확도 하락분($C_{j,j} - C_{i,j}$)의 평균으로 측정한다.

## 📊 Results

### 실험 설정
* **데이터셋**: RoamingRooms (Matterport3D 기반)
* **모델**: ResNet-12 백본 사용 (평가 시 백본은 frozen)
* **하이퍼파라미터**: $N=4$, $T=100$, 레이블 데이터 비율 0.4, Batch size = 1

### 정량적 결과 (Table 1 요약)
| Method | $O_{avg}$ ($\uparrow$) | $F_{avg}$ ($\downarrow$) |
| :--- | :---: | :---: |
| Upper bound | 81.25 | - |
| Base | 66.76 | 22.48 |
| LwF | 56.03 | 2.38 |
| Proto-OML | 62.75 | 8.41 |
| CPM | 73.26 | 31.11 |
| OAP | 69.23 | 5.91 |

### 결과 분석
1. **CPM**은 온라인 정확도가 가장 높았으나, 망각 수준 또한 가장 심각했다. 이는 시공간적 맥락을 유지하는 것이 최신 정보 학습에는 유리하지만, 장기 기억 유지에는 불리함을 시사한다.
2. **LwF**는 망각이 거의 없었으나 온라인 정확도가 매우 낮았다. 이는 망각을 줄이기 위해 새로운 정보를 학습하는 능력(Plasticity)을 희생했음을 의미한다.
3. **OAP**는 CPM보다 정확도는 약간 낮지만 망각은 훨씬 적어, 본 설정에서는 복잡한 맥락 가중치보다 단순 평균 방식이 더 균형 잡힌 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 온라인 지속 학습에서 **Plasticity(가소성, 새로운 정보를 학습하는 능력)**와 **Stability(안정성, 기존 정보를 유지하는 능력)** 사이의 전형적인 딜레마를 보여준다.

* **강점**: 실제 로봇이 겪을 법한 데이터 스트림 환경을 정교하게 설계하여, 기존의 정적인 벤치마크가 놓쳤던 '환경 이동에 따른 분포 변화'와 '망각' 문제를 동시에 다루었다.
* **비판적 해석**: 실험 결과, 극단적으로 한쪽(정확도 또는 망각 방지)에 치우친 방법론보다 단순한 OAP와 같은 방식이 더 효율적이라는 점이 밝혀졌다. 이는 복잡한 메타 학습이나 정규화 기법이 실제 온라인 스트림 데이터의 변동성을 모두 처리하기에는 여전히 한계가 있음을 시사한다.
* **미해결 질문**: 상한선(Upper bound)과 실제 방법론들 사이에 여전히 큰 성능 격차가 존재하며, 이는 온라인 환경에서 망각을 최소화하면서도 높은 적응력을 유지할 수 있는 새로운 아키텍처나 학습 전략이 필요함을 의미한다.

## 📌 TL;DR

본 논문은 로봇의 실제 이동 경로를 모사한 **Online Few-Shot Continual Learning** 설정인 'Continual Wandering'을 제안하였다. 다양한 베이스라인을 평가한 결과, 빠른 학습 속도(Online Accuracy)를 가진 모델은 망각이 심하고, 망각을 잘 방지하는 모델은 학습 속도가 느린 트레이드-오프 관계를 확인하였다. 결론적으로 단순한 온라인 평균 방식(OAP)이 비교적 균형 잡힌 성능을 보였으며, 향후 연구는 이 두 가지 상충하는 목표를 동시에 달성하는 방향으로 나아가야 함을 제시한다.