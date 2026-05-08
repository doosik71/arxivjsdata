# LIP: Learning Instance Propagation for Video Object Segmentation

Ye Lyu, George Vosselman, Gui-Song Xia, Michael Ying Yang (2019)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 비디오 객체 분할(Video Object Segmentation, VOS)이다. VOS는 비디오 내에서 배경으로부터 전경 객체를 분할함과 동시에, 전체 프레임에 걸쳐 객체의 정체성(Identity)을 일관되게 유지하며 추적하는 것을 목표로 한다. 특히 본 연구는 첫 번째 프레임에 대해서만 정답(Ground Truth) 마스크가 제공되는 **반지도 학습(Semi-supervised)** 환경의 VOS에 집중한다.

VOS 작업이 어려운 이유는 다음과 같은 세 가지 주요 도전 과제 때문이다.

1. **외형 변화(Appearance Change):** 시간이 흐름에 따라 대상 객체나 주변 배경의 모습이 크게 변할 수 있다.
2. **포즈 및 크기 변화(Pose and Scale Variation):** 객체의 움직임으로 인해 크기가 급격히 변하거나 형태(포즈)가 바뀔 수 있다.
3. **객체 간 폐쇄(Occlusions):** 여러 객체가 겹치거나 다른 물체에 의해 가려지는 상황이 발생하여 추적이 어려워진다.

따라서 본 논문의 목표는 이러한 동적인 변화 속에서도 여러 객체를 동시에, 그리고 일관성 있게 분할하고 추적할 수 있는 단일 엔드-투-엔드(end-to-end) 학습 가능 네트워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스 분할(Instance Segmentation) 네트워크인 **Mask-RCNN**과 시각적 메모리 모듈인 **Conv-GRU**를 결합하여, 픽셀 단위가 아닌 **인스턴스 단위의 정보 전파(Instance Propagation)**를 학습하는 것이다.

주요 기여 사항은 다음과 같다.

- **Convolutional Gated Recurrent Mask-RCNN** 제안: 이미지 내의 모든 대상 객체를 동시에 분할하고 추적할 수 있는 새로운 구조를 설계하였다.
- **엔드-투-엔드 학습 가능 네트워크**: 장기적인 마스크 전파(Long-term mask propagation)와 상향식 경로 증강(Bottom-up path augmentation)이 가능한 구조를 단일 네트워크로 구현하였다.
- **인스턴스 분할 손실 기반의 학습 전략**: 별도의 후처리(Post-processing)나 합성 비디오 데이터 증강 없이, 순수하게 인스턴스 분할 손실함수만을 사용하여 모델을 성공적으로 학습시키는 전략을 제시하였다.

## 📎 Related Works

### 기존 연구 및 한계

1. **객체 검출 및 분할**: Mask-RCNN은 정적 이미지에서의 인스턴스 분할에 매우 효과적이지만, 시간적 추론(Temporal inference) 능력이 부족하여 비디오 데이터에 직접 적용하기 어렵다.
2. **RNN 및 Conv-GRU**: RNN은 시퀀스 데이터 처리에 유용하지만 기울기 소실/폭주 문제가 있으며, 이를 개선한 Conv-GRU는 공간 정보를 보존하며 비디오 예측 등에 사용되어 왔다. 기존 VOS 연구에서 Conv-GRU가 사용된 적이 있으나, 대부분 이진 세그멘테이션(Binary segmentation)에 그쳐 여러 객체를 동시에 처리하는 데 한계가 있었다.
3. **마스크 전파 기반 VOS**: VPN, MSK, RGMP 등의 방법들은 픽셀 수준의 마스크 전파를 수행한다. 그러나 이러한 방식은 인스턴스 내에서 일관된 레이블을 부여하는 능력이 부족하며, 특히 여러 인스턴스를 하나씩 순차적으로 처리해야 하는 비효율성이 존재한다.
4. **정적 이미지 기반 VOS**: OSVOS, OnAVOS 등은 정적 이미지 데이터셋에서 학습한 모델을 전이 학습(Transfer learning)하여 사용한다. 이들은 대개 정교한 결과물을 얻기 위해 시간 소모가 많은 온라인 적응(Online adaptation)이나 복잡한 후처리에 의존한다.

### LIP의 차별점

LIP는 픽셀 단위가 아닌 인스턴스 단위로 마스크를 예측하며, Conv-GRU를 통해 시각적 메모리를 구축함으로써 외형 변화와 폐쇄 문제를 해결한다. 또한, 다수의 객체를 동시에 처리할 수 있는 구조를 가져 효율성과 일관성을 동시에 확보하였다.

## 🛠️ Methodology

### 전체 시스템 구조

LIP의 전체 파이프라인은 크게 세 가지 부분으로 구성된다: **특징 추출 백본(Feature Extraction Backbone)** $\rightarrow$ **시각적 메모리 모듈(Visual Memory Module)** $\rightarrow$ **예측 헤드(Prediction Heads)**.

1. **Backbone**: ResNet101-FPN과 Group Normalization을 사용하여 이미지에서 유용한 특징을 추출한다.
2. **Conv-GRU Module**: 백본에서 추출된 특징을 입력받아, 새로운 특징을 선택적으로 기억하고 오래된 상태를 잊어버리는 시각적 메모리 역할을 수행한다.
3. **Prediction Heads**: Conv-GRU의 출력 특징을 바탕으로 다음과 같은 헤드들이 작동한다.
    - **RPN (Region Proposal Network)**: 전경 객체의 후보 영역(Proposal)을 생성한다.
    - **Bounding Box Regression Head**: 후보 영역을 정교하게 조정한다.
    - **ID Classification Head**: 각 인스턴스에 고유한 ID 레이블을 부여한다.
    - **Mask Segmentation Head**: 각 인스턴스의 픽셀 단위 마스크를 생성한다.

### Conv-GRU 및 수학적 설명

비디오의 장기 의존성(Long-term dependency)과 폐쇄 문제를 해결하기 위해 Conv-GRU를 사용한다. 특히, 시퀀스 데이터의 특성상 배치 사이즈가 작을 때 발생하는 성능 저하를 막기 위해 기존의 Bias 항을 **Group Normalization (GN)** 레이어로 대체하였다.

Conv-GRU의 연산 과정은 다음과 같다.

- 업데이트 게이트(Update gate):
$$z_t = \sigma(\text{GN}(W_{hz} * h_{t-1} + W_{xz} * x_t))$$
- 리셋 게이트(Reset gate):
$$r_t = \sigma(\text{GN}(W_{hr} * h_{t-1} + W_{xr} * x_t))$$
- 후보 은닉 상태(Candidate hidden state):
$$\hat{h}_t = \Phi(\text{GN}(W_h * (r_t \odot h_{t-1}) + W_x * x_t))$$
- 최종 은닉 상태(Final hidden state):
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \hat{h}_t$$

여기서 $x_t$는 시간 $t$에서의 입력 특징, $h_t$는 은닉 상태, $\sigma$는 시그모이드 함수, $\Phi$는 $\tanh$ 함수, $*$는 합성곱 연산, $\odot$은 요소별 곱셈을 의미한다.

### 학습 절차

학습은 오프라인 학습(Offline Training)과 온라인 학습(Online Training)의 두 단계로 진행된다.

1. **오프라인 학습 (Class-agnostic)**:
    - **Pre-train**: MS-COCO 데이터셋을 사용하여 일반적인 객체 검출 및 분할 능력을 학습시킨다. 이때 Conv-GRU의 은닉 상태는 0으로 고정한다.
    - **Fine-tune**: VOS 데이터셋(DAVIS)을 사용하여 시간적 정보 전파를 학습시킨다. 백본은 동결하고 Conv-GRU와 헤드 부분만 학습시키며, 클래스 예측 헤드를 ID 예측 헤드로 변경한다.
2. **온라인 학습 (Class-specific)**:
    - 테스트 비디오의 첫 번째 프레임 정답을 사용하여 현재 시퀀스에 존재하는 특정 객체들을 구분하도록 미세 조정(Fine-tuning)한다. 백본과 Conv-GRU는 동결하고 ID 헤드와 마스크/BBox 헤드의 마지막 레이어를 학습시킨다.

### 추론 및 제약 조건

추론 시 일관성을 유지하기 위해 두 가지 제약 조건을 적용한다.

- **최대값 제약(One maximum constraint)**: 각 인스턴스 ID에 대해 가장 높은 예측 점수를 가진 객체 하나만 선택한다.
- **위치 연속성 제약(Location continuity constraint)**: 이전 프레임에서 검출된 객체와 현재 프레임 검출 객체 간의 IoU가 너무 낮으면 해당 예측을 억제한다.
- 또한, 시간이 흐를수록 ID 예측 점수가 낮아지는 문제를 해결하기 위해 추론 중에 ID 헤드의 마지막 선형 레이어를 매우 가볍게 미세 조정하는 과정을 거친다.

## 📊 Results

### 실험 설정

- **데이터셋**: DAVIS 2016 (단일 객체 VOS), DAVIS 2017 (다중 객체 VOS).
- **평가 지표**: $J$ (Jaccard index, IoU) 및 $F$ (Boundary F-score).
- **비교 대상**: OnAVOS, OSVOS, FAVOS, OSMN 등 최신 VOS 방법론들.

### 주요 결과

1. **DAVIS 2016 (단일 객체)**:
    - LIP는 비교 대상 중 상위 4위에 랭크되었다.
    - 후처리를 제외했을 때 FAVOS나 OSVOS보다 우수한 성능을 보였다.
2. **DAVIS 2017 (다중 객체)**:
    - LIP는 다중 객체 분리 및 인스턴스 내 일관성 유지 측면에서 매우 강점을 보였으며, OnAVOS를 제외한 대부분의 방법론보다 우수한 J&F Mean 성능을 기록하였다.
3. **정성적 분석**:
    - 타 모델들이 하나의 레이블을 여러 객체에 부여하거나, 한 객체에 여러 레이블을 부여하는 오류를 범하는 반면, LIP는 각 인스턴스를 명확하게 분리하여 추적하는 모습을 보였다.

### 절제 연구 (Ablation Study)

시각적 메모리(Conv-GRU)의 유무에 따른 성능 차이를 분석하였다.

- **정적 모델(Static, $h_{t-1}=0$)**: J&F Mean 59.2%
- **동적 모델(Dynamic, LIP 전체)**: J&F Mean 61.1%
- 이를 통해 Conv-GRU를 통한 동적 메모리 전파가 급격한 외형 변화를 처리하는 데 필수적임을 확인하였다.

## 🧠 Insights & Discussion

### 강점

- **인스턴스 기반 접근**: 기존의 픽셀 기반 전파 방식과 달리 Mask-RCNN의 구조를 활용해 인스턴스 단위로 추적함으로써, 다중 객체 상황에서도 레이블의 일관성을 매우 잘 유지한다.
- **단순하고 효율적인 파이프라인**: 복잡한 후처리나 합성 데이터 없이 단일 네트워크 구조와 효율적인 학습 전략만으로 경쟁력 있는 성능을 확보하였다.

### 한계 및 논의

- **첫 프레임 의존성**: 반지도 학습 특성상 첫 프레임의 마스크가 정확하지 않을 경우 전체 시퀀스의 성능이 저하될 위험이 있다.
- **ID 헤드 감쇠**: 시간이 지남에 따라 ID 예측 점수가 낮아지는 현상이 발생하여, 이를 보완하기 위한 온라인 미세 조정 단계가 필수적이라는 점은 모델의 완전한 자율 추론 능력을 제한하는 요소가 될 수 있다.
- **계산 복잡도**: Conv-GRU를 FPN의 모든 레벨에 적용하고 온라인 미세 조정을 수행하므로, 추론 속도 면에서 최적화 여지가 남아 있을 것으로 보인다.

## 📌 TL;DR

본 논문은 **Mask-RCNN의 인스턴스 분할 능력**과 **Conv-GRU의 시각적 메모리**를 결합한 **LIP** 네트워크를 제안하여, 비디오 내 여러 객체를 동시에 일관성 있게 추적하는 방법을 제시하였다. 특히 픽셀 단위가 아닌 **인스턴스 단위의 정보 전파**를 통해 다중 객체 VOS 작업에서 탁월한 성능을 보였으며, 이는 향후 자율주행이나 로보틱스 등 실시간 다중 객체 추적이 필요한 분야에 중요한 기초 연구가 될 가능성이 높다.
