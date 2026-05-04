# InsPro: Propagating Instance Query and Proposal for Online Video Instance Segmentation

Fei He, Haoyang Zhang, Naiyu Gao, Jian Jia, Yanhu Shan, Xin Zhao, Kaiqi Huang (2022)

## 🧩 Problem to Solve

Video Instance Segmentation (VIS)은 비디오 내 객체의 세그멘테이션(Segmentation)과 추적(Tracking)을 동시에 수행하는 과제이다. 기존의 VIS 방법론은 크게 두 가지 패러다임으로 나뉜다. 첫째는 프레임 단위로 객체를 검출한 후 별도의 tracking head나 복잡한 매칭 알고리즘을 통해 인스턴스를 연결하는 'Tracking-by-detection' 방식이며, 둘째는 비디오를 여러 클립으로 나누어 처리한 후 클립 간 인스턴스를 매칭하는 'Clip-matching' 방식이다.

이러한 기존의 명시적 인스턴스 연관(Explicit Instance Association) 방식은 시스템의 복잡도를 증가시키고 추론 속도를 저하시킨다. 무엇보다 인스턴스 예측이 각 프레임이나 클립 단위로 분리되어 수행되기 때문에, 비디오가 가진 고유한 시간적 단서(Temporal cues)를 충분히 활용하지 못한다는 치명적인 한계가 있다. 본 논문의 목표는 명시적인 연관 단계 없이, 세그멘테이션과 추적을 동시에 수행하는 단순하고 빠르며 효과적인 온라인 VIS 프레임워크를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **인스턴스 쿼리-제안 쌍(Instance Query-Proposal Pair)의 시간적 전파(Temporal Propagation)** 메커니즘을 통해 명시적인 매칭 과정 없이 암시적(Implicit)으로 인스턴스 연관을 달성하는 것이다.

주요 기여 사항은 다음과 같다.

1. **Implicit Instance Association**: 이전 프레임에서 생성된 쿼리-제안 쌍을 다음 프레임으로 전파함으로써, 동일한 쿼리에 의해 생성된 인스턴스는 자동으로 동일한 ID를 갖게 하여 추적 과정을 단순화하였다.
2. **Intra-query Attention**: 이전 $T$개 프레임의 인스턴스 특징을 저장하는 Feature Bank를 구축하고, 현재 쿼리를 보강하여 폐색(Occlusion)이나 모션 블러(Motion Blur) 상황에서도 강건한 표현력을 갖게 하였다.
3. **Temporally Consistent Matching (TCM)**: 학습 과정에서 특정 쿼리가 비디오 전체에 걸쳐 하나의 특정 객체와 일대일 대응을 유지하도록 강제하는 매칭 전략을 제안하였다.
4. **Box Deduplication Loss (BDL)**: 동일 객체에 대해 중복된 제안(Proposal)이 생성되는 것을 억제하고, 사용되지 않는 쿼리들이 새로운 객체를 탐지할 수 있는 후보가 되도록 유도하는 손실 함수를 도입하였다.

## 📎 Related Works

기존 VIS 연구는 다음과 같이 분류되며, 본 연구는 이들과 명확한 차별점을 갖는다.

- **Frame-level VIS (Online)**: MaskTrack R-CNN과 같은 방법들은 'Tracking-by-detection' 방식을 사용하여 별도의 추적 모듈이 필요하며, 이는 연산 복잡도를 높인다. 본 연구는 전파 메커니즘을 통해 이 과정을 암시적으로 통합하였다.
- **Clip-level VIS (Offline)**: VisTR이나 Mask2Former와 같은 Transformer 기반 방법들은 클립 내에서는 쿼리를 공유하여 연관을 수행하지만, 클립 간에는 여전히 수동으로 설계된 매칭 알고리즘이 필요하다. 반면 InsPro는 프레임 단위로 작동하며 비디오 전체 길이에 대해 전파가 가능하므로 온라인 환경에 적합하다.
- **Query-based Object Association**: TransTrack, MOTR 등은 쿼리 전파를 사용하지만, '기존 객체 추적용 쿼리'와 '새 객체 탐지용 쿼리'를 분리하여 관리한다. 이 경우 두 쿼리 셋을 합치기 위한 휴리스틱한 규칙이 필요하며, 낮은 점수의 쿼리가 필터링될 때 궤적이 끊기는 문제가 발생한다. InsPro는 통합된(United) 쿼리 셋을 사용하여 이러한 복잡성을 제거하고 더 우아한 방식으로 탐지와 추적을 동시에 수행한다.

## 🛠️ Methodology

### 1. Instance Query and Proposal Propagation

InsPro는 Sparse R-CNN의 구조에서 영감을 받아, 학습 가능한 인스턴스 쿼리 $q \in \mathbb{R}^{N \times C}$와 제안(Proposal) $p \in \mathbb{R}^{N \times 4}$의 쌍을 사용한다.

- **작동 원리**: 첫 프레임 $I_0$에서 초기 쌍 $(q_{init}, p_{init})$을 사용하여 결과 $r_0$를 예측하고, 업데이트된 쌍 $(q_0, p_0)$를 생성한다. 이 쌍을 다음 프레임 $I_1$으로 전파하여 $r_1$과 $(q_1, p_1)$을 생성하는 과정을 반복한다.
- **암시적 연관**: 동일한 쿼리 슬라이스(Slice)에 의해 생성된 인스턴스들은 자동으로 동일한 ID를 부여받으므로, 별도의 매칭 단계 없이 추적이 완료된다.

### 2. Segmentation Head

SegHead는 다단계 네트워크로 구성되며, 다음의 두 모듈이 핵심이다.

- **Dynamic Instance Interaction Module (DIIM)**:
  - self-attention을 통해 쿼리 간 관계를 모델링한다.
  - RoIAlign을 통해 제안된 영역의 특징을 추출한다.
  - Dynamic Convolution을 사용하여 쿼리와 RoI 특징을 결합해 객체 특징 $o_t$를 생성한다.
- **Instance Segmentation Module**:
  - 분류(Classification) 및 박스 회귀(Regression) 헤드를 통해 클래스와 위치를 예측한다.
  - **Conditional Convolution**: $o_t$로부터 가중치 $\omega_i$를 생성하고, 이를 FPN 특징 맵 $f_{mask}^t$에 적용하여 마스크 $m_{i}^t$를 생성한다.
    $$m_{i}^{t} = \text{CondConv}(f_{mask}^{t}, \omega_{i})$$

### 3. Intra-query Attention

장기적인 시간적 단서를 포착하기 위해 이전 $T$개 프레임의 특징을 저장하는 Feature Bank $fb = \{o_{t-T+1}, \dots, o_t\}$를 운영한다. 현재 쿼리 $q_i^t$는 다음과 같이 가중합 형태로 보강된다.
$$q_{i}^{t} = \frac{\sum_{n=0}^{T-1} o_{i}^{t-n} \exp(\varepsilon(o_{i}^{t-n}))}{\sum_{m=0}^{T-1} \exp(\varepsilon(o_{i}^{t-m}))} + o_{i}^{t}$$
여기서 $\varepsilon(\cdot)$는 선형 변환 함수이며, 이 메커니즘은 폐색이나 모션 블러 상황에서 객체 표현력을 높인다.

### 4. Temporally Consistent Matching (TCM)

학습 시 쿼리와 실제 객체(GT) 간의 일대일 대응을 유지하기 위한 전략이다.

- 첫 프레임에서 Hungarian algorithm을 사용하여 예측값과 GT 간의 최적 매칭을 수행한다.
- 이후 프레임에서는 **이전 프레임의 매칭 결과를 그대로 전파**한다. 즉, 특정 GT 객체가 계속 존재한다면 첫 프레임에서 매칭되었던 동일한 쿼리에 계속 할당된다. 새로운 객체가 나타날 때만 새로운 매칭을 수행한다.

### 5. Box Deduplication Loss (BDL)

동일 객체에 여러 제안 박스가 겹치는 문제를 해결하기 위해, GT 박스와 매칭되지 않은 음성(Negative) 박스들 간의 중심 거리 기반 손실 함수를 도입한다.
$$L_{dedup} = \frac{1}{k} \sum_{i=1}^{k} \max(\beta - \frac{C^2(b, \hat{b}_{i}^{neg})}{D^2(b)}, 0)$$
여기서 $C(\cdot)$는 중심 거리, $D(\cdot)$는 박스의 대각선 길이, $\beta$는 하이퍼파라미터이다. 이 손실 함수는 중복 박스를 GT로부터 밀어내어 중복을 억제하고, 남은 쿼리들이 공간적으로 고르게 분포하게 하여 새로운 객체 탐지 가능성을 높인다.

## 📊 Results

### 1. 정량적 성능 평가

YouTube-VIS 2019 및 2021 벤치마크에서 ResNet-50 백본을 사용한 결과, InsPro는 기존의 모든 온라인 VIS 방법론을 능가하는 성능을 보였다.

| 방법론 | YouTube-VIS 2019 (AP) | YouTube-VIS 2021 (AP) | FPS (RTX2080Ti) |
| :--- | :---: | :---: | :---: |
| EfficientVIS | 37.9 | 34.0 | 36 |
| CrossVIS | 36.3 | 34.2 | 25.6 |
| **InsPro** | **40.2** | **36.1** | **26.3** |
| **InsPro (with COCO)** | **43.2** | **37.6** | **26.3** |
| **InsPro-lite** | 38.7 | - | **45.7** |

- **InsPro-lite**: 키 프레임(Key frame)과 비-키 프레임을 구분하여 연산량을 줄인 버전으로, 45.7 FPS라는 매우 빠른 속도를 달성하면서도 성능 저하를 최소화하였다.

### 2. 추가 실험 및 분석

- **OVIS (Occluded VIS)**: 폐색이 심한 데이터셋에서도 Feature Calibration 같은 추가 모듈 없이 Intra-query attention과 전파 메커니즘만으로 기존 방법론보다 높은 AP(21.1)를 기록하였다.
- **ImageNet VID**: 대규모 데이터셋에서도 기존의 Viterbi 알고리즘 기반 매칭 방식보다 우수한 성능(AP 38.9)을 보였으며, 특히 BDL이 새로운 객체 탐지 성능을 크게 향상시킴을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

InsPro의 가장 큰 강점은 **'통합 쿼리(United Query)'** 설계에 있다. 기존의 MOTR 등은 추적 쿼리와 탐지 쿼리를 나누어 관리하며 복잡한 휴리스틱 규칙에 의존했지만, InsPro는 단일 쿼리 셋을 전파함으로써 구조적 단순함과 효율성을 동시에 잡았다. 또한, BDL은 단순히 중복을 제거하는 것을 넘어, 쿼리들을 공간적으로 분산시켜 새로운 객체가 나타났을 때 이를 낚아챌 수 있는 '빈 공간'을 확보해 주는 효과를 준다.

### 한계 및 비판적 해석

논문에서 언급되었듯, InsPro는 **작은 객체(Small objects)**에 대한 세그멘테이션 성능이 다소 떨어지는 경향이 있다. 이는 현재의 아키텍처가 작은 객체를 처리하기 위한 특화된 설계(예: 고해상도 피처 맵의 정밀한 활용 등)가 부족하기 때문으로 판단된다. 또한, 쿼리의 개수 $N$이 고정되어 있어, 한 프레임에 매우 많은 수의 객체가 등장할 경우 성능 저하가 발생할 가능성이 있다.

## 📌 TL;DR

본 논문은 온라인 비디오 인스턴스 세그멘테이션(VIS)을 위해 **인스턴스 쿼리와 제안(Proposal)을 시간적으로 전파하는 InsPro** 프레임워크를 제안한다. 명시적인 인스턴스 매칭 단계 없이 암시적으로 추적을 수행하며, Intra-query attention, TCM, BDL 등의 기법을 통해 정확도와 강건성을 높였다. 결과적으로 YouTube-VIS 벤치마크에서 SOTA 성능을 달성했으며, 특히 실시간성에 근접한 InsPro-lite 버전은 실제 자율주행이나 비디오 편집 서비스에 적용될 가능성이 매우 높다.
