# MaskRNN: Instance Level Video Object Segmentation

Yuan-Ting Hu, Jia-Bin Huang, Alexander G. Schwing (2017)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 비디오 내에서 여러 개의 객체 인스턴스를 개별적으로 분리하고 추적하는 **인스턴스 수준의 비디오 객체 분할(Instance Level Video Object Segmentation)**이다. 비디오 객체 분할은 비디오 편집 및 압축 등의 분야에서 매우 중요하지만, 실제 환경에서는 객체의 형태 변형, 빠른 움직임, 그리고 객체 간의 상호 가림(occlusion) 현상으로 인해 구현이 매우 어렵다.

기존의 접근 방식들은 주로 단일 객체의 전경-배경 분할(foreground-background segmentation)에 집중하거나, 강체(rigid) 장면을 가정하는 기하학적 모델에 의존하는 경향이 있었다. 따라서 본 논문의 목표는 다수의 객체가 존재하는 복잡한 비디오 장면에서 temporal coherence(시간적 일관성)를 유지하며 각 인스턴스를 정확하게 분할해내는 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **재귀 신경망(Recurrent Neural Network, RNN)**을 통해 비디오의 장기적인 시간적 구조를 학습하고, **이진 분할 네트워크(Binary Segmentation Net)**와 **위치 추정 네트워크(Localization Net)**를 결합하여 객체의 위치 우선순위(location prior)를 활용하는 것이다.

구체적으로, 전체 문제를 여러 개의 이진 분할 문제로 나누어 처리하는 bottom-up 방식을 채택하였다. 각 객체 인스턴스마다 별도의 네트워크를 할당하여 마스크와 바운딩 박스를 동시에 예측하게 함으로써, 바운딩 박스를 통해 엉뚱한 영역이 분할되는 outlier를 제거하고 RNN을 통해 이전 프레임의 정보를 효율적으로 전파함으로써 시간적 일관성을 확보하였다.

## 📎 Related Works

비디오 객체 분할 연구는 크게 두 가지 흐름으로 구분된다.

1.  **그래프 기반 접근 방식(Graph-based approaches):** 3차원 시공간 그래프를 구축하여 픽셀 간의 관계를 모델링하고 라벨을 전파하는 방식이다. 이 방법은 인간의 개입 정도에 따라 인터랙티브, 반지도(semi-supervised), 비지도 학습으로 나뉜다. 하지만 3D 그래프 구축 및 탐색에 막대한 계산 비용이 소요되어 실시간 처리가 어렵고, 파라미터 설정에 민감하다는 한계가 있다.
2.  **딥러닝 기반 접근 방식(Deep learning methods):** 주로 정적 이미지 분할 데이터셋으로 사전 학습된 네트워크를 사용한다. 최근에는 광학 흐름(optical flow)이나 시맨틱 분할 정보를 활용해 프레임 간 연결성을 높이려는 시도가 있었다. 그러나 대부분의 기존 연구는 단일 객체의 전경-배경 분할에만 국한되어 있으며, 다수 객체에 대한 인스턴스 수준의 분할을 다루지 않았고, 특히 위치 정보(location prior)를 명시적으로 모델링하지 않았다는 차이점이 있다.

## 🛠️ Methodology

### 전체 시스템 구조
MaskRNN은 비디오 시퀀스 $I = \{I_1, I_2, \dots, I_T\}$가 주어졌을 때, 첫 번째 프레임의 정답 마스크 $y^*_1$을 기반으로 이후 프레임 $y_2, \dots, y_T$를 예측한다. 시스템은 각 객체 인스턴스 $i \in \{1, \dots, N\}$에 대해 독립적인 딥러닝 네트워크를 운용하며, 최종 결과는 이들의 예측값을 병합하여 산출한다.

### 주요 구성 요소 및 역할

#### 1. 이진 분할 네트워크 (Binary Segmentation Net)
각 객체에 대해 전경-배경 확률 맵 $b^i_t \in [0, 1]^{H \times W}$를 예측하며, 두 개의 스트림으로 구성된다.
- **Appearance Stream:** 현재 프레임 $I_t$와 이전 프레임의 예측 마스크를 광학 흐름으로 변형(warp)한 $\phi_{t-1, t}(y_{t-1})$를 입력으로 받는다.
- **Flow Stream:** 광학 흐름의 크기(magnitude)와 변형된 마스크 $\phi_{t-1, t}(y_{t-1})$를 입력으로 받는다.

두 스트림은 VGG-16 구조를 기반으로 하며, Max-pooling 이전의 특징 맵들을 추출하여 bilinear interpolation로 업샘플링한 뒤 선형 결합한다. 최종적으로 sigmoid 함수를 통과시켜 픽셀별 전경 확률을 계산한다. 학습 시에는 weighted binary cross entropy loss를 사용한다.

#### 2. 객체 위치 추정 네트워크 (Object Localization Net)
객체가 인접 프레임 사이에서 급격하게 움직이지 않는다는 가정을 바탕으로 bounding box regression을 수행한다.
- 변형된 마스크 $\phi_t(b^i_{t-1})$에서 바운딩 박스 제안(proposal)을 생성한다.
- Fast R-CNN의 RoI-pooling 방식을 사용하여 Appearance stream의 $\text{conv5\_3}$ 특징에서 위치 정보를 추출하고, 두 개의 fully connected layer를 거쳐 박스 위치를 회귀 예측한다.
- 예측된 바운딩 박스를 $1.25$배 확대하여 적용하며, 이 영역 밖에 존재하는 분할 결과는 outlier로 간주하여 제거한다.

#### 3. 인스턴스 수준 병합 (Multiple Instance Level Segmentation)
각 객체 $i$에 대해 계산된 이진 마스크 $b^i_t$들을 하나로 합쳐 최종 예측 $y_t$를 생성한다. 모든 픽셀 $p$에 대해 가장 높은 확률값을 가진 클래스를 할당하는 $\text{argmax}$ 연산을 수행한다.
$$y^p_t = \text{argmax}_{i \in \{0, \dots, N\}} (b^i_t)$$
(여기서 $i=0$은 배경을 의미한다.)

### 학습 절차
- **Offline Training:** 정적 이미지에서 먼저 최적화한 후, GPU 메모리 한계로 인해 7프레임 단위로 BPTT(Back-propagation Through Time)를 적용하여 RNN의 재귀 관계를 학습시킨다.
- **Online Finetuning:** 테스트 시, 제공된 첫 번째 프레임의 정답 마스크 $y^*_1$을 사용하여 네트워크를 해당 비디오의 객체 특성에 맞게 미세 조정한다. 이때 RNN은 사용되지 않고 단일 프레임에 대해서만 최적화를 수행한다.

## 📊 Results

### 실험 설정
- **데이터셋:** DAVIS-2016, DAVIS-2017, SegTrack v2를 사용하였다.
- **평가 지표:**
    - **IoU (Intersection over Union):** 예측 마스크와 정답 마스크의 겹침 정도를 측정한다.
    - **Contour Accuracy ($\mathcal{F}$):** 경계선 픽셀 간의 정밀도(Precision)와 재현율(Recall)을 사용하여 경계 묘사 능력을 측정한다.
    - **Temporal Stability ($\mathcal{T}$):** 인접 프레임 간 마스크 형태의 변화량을 측정하여 시간적 일관성을 평가한다.

### 주요 결과
- **정량적 성능:** DAVIS-2016에서는 기존 SOTA 대비 $0.6\%$ 향상된 성능을 보였으며, 인스턴스 수준 분할 작업인 DAVIS-2017과 SegTrack v2에서는 각각 $5.6\%$, $4.6\%$의 IoU 향상을 달성하며 압도적인 성능을 기록하였다.
- **Ablation Study:** 
    - Online finetuning이 성능 향상에 결정적인 역할을 함을 확인하였다.
    - Appearance stream과 Flow stream을 결합했을 때, 움직이는 객체의 경계를 더 잘 포착하여 성능이 향상되었다.
    - Bounding box를 통해 분할 영역을 제한했을 때 outlier가 효과적으로 제거되어 IoU가 상승하였다.
    - RNN을 통한 재귀적 학습이 시간적 일관성을 높여 결과적으로 성능을 개선시켰다.

## 🧠 Insights & Discussion

### 강점
MaskRNN은 단순한 프레임 단위 분할을 넘어, RNN을 통한 장기적 의존성 학습과 Bounding box를 통한 위치 제약을 동시에 적용하였다. 특히 다수 객체 분할 문제를 개별 이진 분할 문제로 치환하여 해결함으로써 복잡도를 낮추면서도 높은 정확도를 확보하였다.

### 한계 및 비판적 해석
논문에서 제시한 실패 사례(Failure cases)를 분석하면 두 가지 주요 약점이 드러난다.
첫째, **유사한 외형을 가진 객체**가 존재할 경우, 네트워크가 인스턴스를 오인하여 추적에 실패하는 경향이 있다. 이는 Appearance stream이 외형에 크게 의존하기 때문으로 해석된다.
둘째, **급격한 스케일 변화나 시점 변환(viewpoint change)**이 일어나는 경우, 위치 추정 네트워크와 마스크 warping이 이를 충분히 따라가지 못해 추적을 놓치는 문제가 발생한다.

결과적으로 본 모델은 시간적 일관성을 높였음에도 불구하고, 객체의 외형 변화가 극심한 환경에서의 강건성(robustness) 문제는 여전히 해결해야 할 과제로 남아있다.

## 📌 TL;DR

MaskRNN은 RNN과 Bounding box regression을 결합하여 비디오 내 다수 객체의 인스턴스 수준 분할을 수행하는 프레임워크이다. 각 객체별로 Appearance/Flow stream 기반의 이진 분할과 위치 추정을 동시에 수행하며, 이를 통해 시간적 일관성을 확보하고 outlier를 제거한다. DAVIS 및 SegTrack 데이터셋에서 SOTA 성능을 달성하였으며, 특히 위치 우선순위(location prior)의 도입이 비디오 객체 분할의 정확도를 높이는 데 핵심적인 역할을 함을 입증하였다.