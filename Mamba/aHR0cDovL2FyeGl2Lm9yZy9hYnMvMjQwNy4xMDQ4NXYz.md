# MM-Tracker: Motion Mamba with Margin Loss for UAV-platform Multiple Object Tracking

Mufeng Yao, Jinlong Peng, Qingdong He, Bo Peng, Hao Chen, Mingmin Chi, Chao Liu, Jon Atli Benediktsson (2025)

## 🧩 Problem to Solve

본 논문은 무인 항공기(UAV) 플랫폼에서 수행되는 다중 객체 추적(Multiple Object Tracking, MOT)의 효율적인 모션 모델링과 탐지 성능 향상을 목표로 한다. UAV-MOT는 일반적인 지상 기반 MOT와 달리 다음과 같은 고유한 어려움을 가진다.

첫째, 지상의 객체 자체의 움직임(Local motion)뿐만 아니라 UAV 카메라의 이동으로 인한 전역적 움직임(Global camera motion)이 동시에 발생하여 객체의 궤적 예측이 매우 복잡하다. 기존의 Kalman Filter 기반 방식은 선형 운동과 같은 단순한 가정을 전제로 하므로 이러한 복잡한 카메라 움직임 상황에서 정확도가 떨어진다.

둘째, 급격한 카메라 움직임은 심각한 모션 블러(Motion blur)를 유발하며, 이는 객체 탐지(Object Detection) 단계에서 탐지 정확도를 크게 떨어뜨린다. 특히 UAV-MOT 데이터셋에서 큰 움직임을 가진 객체는 그 수가 적어 학습 데이터의 불균형(Long-tailed distribution) 문제가 발생하며, 결과적으로 탐지기가 이러한 어려운 샘플을 충분히 학습하지 못하는 문제가 있다.

따라서 본 연구는 전역 및 지역 모션을 모두 고려하는 효율적인 모션 모델링과 모션 블러가 발생한 객체의 탐지 성능을 높이는 학습 전략을 통해 UAV-MOT의 성능을 개선하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 크게 두 가지 모듈의 제안으로 요약된다.

1.  **Motion Mamba Module**: 지역적인 교차 상관(Cross-correlation)과 전역적인 양방향 Mamba 블록(Bi-directional Mamba Block)을 결합하여, 계산 효율성을 유지하면서도 지역 및 전역 모션 특징을 모두 추출할 수 있는 모션 모델링 구조를 설계하였다. 특히 탐지기(Detector)의 특징 맵을 재사용함으로써 중복 계산을 줄였다.
2.  **Motion Margin Loss (MMLoss)**: 객체의 움직임 크기에 따라 결정 경계(Decision boundary)를 다르게 설정하는 손실 함수를 제안하였다. 움직임이 큰 객체일수록 더 큰 마진을 부여하여, 모델이 모션 블러가 심한 어려운 객체에 대해 더 높은 확신도(Confidence score)를 출력하도록 강제함으로써 탐지 성능을 향상시켰다.

## 📎 Related Works

### 기존 모션 모델링의 한계
전형적인 MOT는 탐지와 연관(Association) 단계로 나뉘며, 연관 단계에서 위치 정보 기반의 모션 모델링이 사용된다. 
- **Kalman Filter 기반 방식**: 파라미터 학습이 없는 비학습 알고리즘으로, 특정 운동 패턴을 가정해야 하므로 임의의 카메라 움직임이 빈번한 UAV 환경에서는 한계가 명확하다.
- **학습 기반 방식**: CNN이나 교차 상관(Cross-correlation)을 사용하지만, 주로 지역적(Local) 정보에 의존하여 전역적인 카메라 움직임을 포착하지 못하며, 특징 추출 과정에서 중복 계산이 발생하는 비효율성이 있다.

### 전역 정보 집계 방식의 비교
- **RNN/LSTM**: 전역 특징 추출이 가능하나, 순차적 추론 구조로 인해 병렬 학습이 어렵고 그래디언트 소실/폭주 문제가 발생한다.
- **Transformer**: Global Attention을 통해 전역 정보를 효과적으로 처리하지만, 계산 복잡도가 $O(n^2)$으로 매우 높아 실시간 추적이 어렵고 과적합 위험이 크다.
- **Mamba (State Space Model)**: 선형 시간 복잡도로 전역 어텐션과 유사한 효과를 내며 병렬 학습이 가능하여, 본 논문에서는 이를 통해 효율적인 전역 모션 모델링을 구현하였다.

## 🛠️ Methodology

### 전체 파이프라인
MM-Tracker는 YOLOX-S를 기반으로 하는 탐지 백본(DetBackbone)과 탐지 헤드(DetHead)를 사용한다. 백본에서 추출된 다중 스케일 특징 맵은 두 갈래로 나뉜다. 하나는 객체의 바운딩 박스와 점수를 출력하는 탐지 헤드로 가고, 다른 하나는 제안된 Motion Mamba 모듈로 들어가 모션 맵(Motion Map)을 생성한다. 최종적으로 생성된 모션 맵을 통해 객체의 다음 위치를 예측하고, 이를 현재 프레임의 탐지 결과와 공간적으로 매칭하여 궤적을 생성한다.

### Motion Mamba Module
이 모듈은 이전 프레임과 현재 프레임의 탐지 특징 맵을 입력으로 받아 모션 특징을 추출한다.

1.  **지역 모션 추출**: 두 프레임의 특징 맵 간 교차 상관(Cross-correlation)을 통해 지역적인 움직임 정보를 먼저 추출한다.
2.  **전역 모션 추출 (Motion Mamba Block)**: 전역적 상호작용을 위해 양방향 상태 공간 모델(State Space Model, SSM)을 사용한다.
    - **V-SSM (Vertical SSM)**: 특징 맵의 각 열을 따라 수직 방향으로 스캔한다.
    - **H-SSM (Horizontal SSM)**: 특징 맵의 각 행을 따라 수평 방향으로 스캔한다.
    - 두 방향의 스캔 결과와 입력 값을 더해(Shortcut connection) 전역 특징을 통합한다.
3.  **특징 융합 및 출력**: 낮은 스케일의 특징을 단계적으로 업샘플링하여 높은 스케일의 특징과 융합하며, 최종적으로 원본 이미지 크기의 $1/8$ 크기를 가진 모션 맵을 출력한다. 이 맵의 두 채널은 각각 수평 및 수직 방향의 움직임을 나타낸다.

**상태 공간 모델(SSM)의 수식:**
입력 시퀀스 $x_t$에 대해 다음과 같은 반복 계산을 수행한다.
$$h_t = \hat{A}h_{t-1} + \hat{B}x_t$$
$$y_t = Ch_t + Dx_t$$
여기서 $\hat{A}$와 $\hat{B}$는 입력 $x_t$에 따라 동적으로 결정되는 선택적 스캔(Selective scanning) 메커니즘을 따른다.
$$\hat{A} = \exp(Adt), \quad \hat{B} = Bdt, \quad dt = \text{MLP}(x_t)$$

### Motion Margin Loss (MMLoss)
모션 블러가 심한(움직임이 큰) 객체의 탐지 성능을 높이기 위해, 객체의 오프셋 $x$에 따라 동적인 마진 $D(x)$를 설정한다.

**마진 함수 $D(x)$:**
$$D(x) = s \cdot \left( \frac{1}{1 + e^{-(x-5)/s}} \right) - M$$
$$M = s \cdot \left( \frac{1}{1 + e^{-(0-5)/s}} \right)$$
여기서 $s=10$으로 설정하여 움직임이 약 30 픽셀 이상일 때 마진 값이 포화되도록 설계하였다. 이는 움직임이 0일 때는 마진이 없고, 움직임이 커질수록 마진이 증가하여 결정 경계를 넓히는 효과를 준다.

**최종 손실 함수:**
분류 브랜치의 손실 함수에 마진 $D_i$를 반영하여 다음과 같이 정의한다.
$$\text{MMLoss}(y_i, \hat{y}_i, D_i) = -y_i \cdot \log(\sigma(\hat{y}_i - D_i)) - (1-y_i) \cdot \log(1-\sigma(\hat{y}_i))$$
$\hat{y}_i$에서 $D_i$를 뺌으로써, 움직임이 큰 객체는 더 높은 예측 점수를 내야만 손실이 줄어들게 되어, 결과적으로 학습 과정에서 이러한 객체들에 대해 더 강한 확신도를 갖도록 유도한다.

### 학습 및 추론 절차
- **Ground-truth 생성**: 사전 학습된 EMD-Flow 네트워크로 광학 흐름(Optical Flow) 맵을 생성한 후, 어노테이션된 객체의 중심점 오프셋 값을 해당 바운딩 박스 영역에 덮어씌워 GT 모션 맵을 생성한다.
- **학습**: 탐지기의 회귀 브랜치는 $\text{IOU Loss}$와 $\text{L1 Loss}$로, 분류 브랜치는 $\text{MMLoss}$로 학습시킨다. 모션 모델은 $\text{L1 Loss}$를 통해 GT 모션 맵과 정렬시킨다.
- **추론**: $t$ 시점의 객체 중심 좌표에서 모션 맵의 값을 읽어 $t+1$ 시점의 예측 위치를 계산하고, 이를 실제 탐지된 박스와 매칭한다.

## 📊 Results

### 실험 설정
- **데이터셋**: VisDrone, UAVDT (UAV 뷰 기반의 다중 객체 추적 데이터셋)
- **평가 지표**: $\text{MOTA}$ (Multiple Object Tracking Accuracy), $\text{IDF1}$ (ID F1 Score)
- **비교 대상**: OC-SORT, FOLT, U2MOT, TrackSSM 등 SOTA 추적기 및 광학 흐름 네트워크(EMD-Flow).

### 주요 결과
1.  **정량적 성능**: MM-Tracker는 두 데이터셋 모두에서 SOTA 성능을 달성하였다.
    - **VisDrone**: $\text{MOTA } 44.7\%$, $\text{IDF1 } 58.3\%$
    - **UAVDT**: $\text{MOTA } 51.4\%$, $\text{IDF1 } 68.9\%$
2.  **효율성**: 모션 예측 속도가 매우 빠르다. Motion Mamba의 추론 시간은 $6.9\text{ms}$로, EMD-Flow($115\text{ms}$) 대비 약 $1/16$ 수준으로 효율적이다.
3.  **Ablation Study**:
    - **Motion Mamba 구성**: Local correlation 단독보다 V-SSM, H-SSM을 모두 결합했을 때 가장 높은 정확도를 보였다.
    - **MMLoss 효과**: Focal Loss나 LDAM Loss보다 $\text{MOTA}$ 및 $\text{IDF1}$ 지표에서 우수한 성능을 보였으며, 실제 시각화 결과 모션 블러가 심한 자전거/오토바이 등의 객체를 더 잘 탐지함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 UAV-MOT의 핵심 난제인 '전역 모션'과 '모션 블러로 인한 탐지 실패'를 구조적(Mamba) 및 학습적(MMLoss) 관점에서 동시에 해결하였다. 특히, 새로운 특징 추출기를 추가하는 대신 기존 탐지기의 특징 맵을 재사용함으로써 연산 오버헤드를 최소화하면서 성능을 높인 점이 인상적이다. 또한, 데이터셋의 불균형 문제를 단순한 리샘플링이 아니라 모션 값에 기반한 동적 마진 부여라는 방식으로 해결하여 도메인 특성을 잘 반영하였다.

### 한계 및 논의사항
- **탐지기 의존성**: 본 연구는 YOLOX-S를 기본 탐지기로 사용하였다. 만약 더 강력한 최신 탐지기(예: YOLOv10 등)를 결합한다면 성능이 더 향상될 가능성이 크다.
- **하이퍼파라미터 설정**: MMLoss의 파라미터 $s=10$ 및 오프셋 기준값 $5$ 등이 30 픽셀 정도의 블러가 발생한다는 경험적 관찰에 기반하여 설정되었다. 이는 데이터셋마다 블러의 정도가 다를 수 있으므로, 이를 자동화하거나 적응적으로 설정하는 방법이 연구될 필요가 있다.
- **모션 맵의 해상도**: 모션 맵의 크기가 원본의 $1/8$로 제한되어 있어, 매우 작은 객체의 정밀한 움직임을 포착하는 데에 한계가 있을 수 있다.

## 📌 TL;DR

MM-Tracker는 UAV 환경의 복잡한 전역 모션과 모션 블러 문제를 해결하기 위해 **Motion Mamba 모듈**과 **Motion Margin Loss**를 제안한 추적기이다. Mamba의 선형 시간 복잡도를 활용해 효율적으로 전역 모션을 모델링하고, 움직임이 큰 객체에 더 엄격한 학습 마진을 부여하여 탐지 성능을 높였다. 결과적으로 VisDrone과 UAVDT 데이터셋에서 정확도와 추론 속도 모두 SOTA를 달성하였으며, 이는 향후 실시간 UAV 감시 및 추적 시스템에 적용될 가능성이 매우 높다.