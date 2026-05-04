# Implicit Feature Refinement for Instance Segmentation

Lufan Ma, Tiancai Wang, Bin Dong, Jiangpeng Yan, Xiu Li, Xiangyu Zhang (2021)

## 🧩 Problem to Solve

본 논문은 이미지 및 비디오 인스턴스 세그멘테이션(Instance Segmentation)에서 인스턴스 특징(Instance Feature)을 정제하는 기존 방식의 한계를 해결하고자 한다. 

기존의 주류 방법론들은 최종 예측 전 단계에서 coarse한 인스턴스 특징을 정제하기 위해 여러 개의 합성곱 층(Convolutional layers)을 명시적으로 쌓아 올리는(Explicitly stacked) 방식을 사용한다. 일반적으로 Mask R-CNN과 같은 구조에서는 4개의 $3 \times 3$ 합성곱 층을 연속적으로 배치하여 수용 영역(Receptive Field)을 넓히고 시맨틱 수준을 높이려 한다. 그러나 이러한 명시적 설계는 다음과 같은 문제점을 가진다.

1. **제한된 유효 수용 영역**: 이론적인 수용 영역에 비해 실제 유효 수용 영역(Effective Receptive Field)은 훨씬 작아, 충분한 전역 정보를 활용하는 데 한계가 있다.
2. **신호 손실 및 파라미터 부담**: 층이 깊어질수록 원래의 인스턴스 신호를 잊어버리는 경향이 있으며, 성능 향상을 위해 층을 추가할수록 파라미터 수와 계산 비용이 급격히 증가한다.

따라서 본 논문의 목표는 파라미터 부담을 줄이면서도, 전역 수용 영역을 확보하여 고품질의 인스턴스 특징을 생성할 수 있는 경량화된 정제 모듈을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 특징 정제 과정을 **암시적 함수(Implicit Function)**로 모델링하여, 무한한 깊이의 네트워크를 시뮬레이션하는 **Implicit Feature Refinement (IFR)** 모듈을 제안하는 것이다.

중심적인 직관은 가중치를 공유하는(Weight-sharing) 블록을 무한히 반복해서 쌓으면, 그 출력값이 결국 하나의 평형 상태(Equilibrium state)로 수렴한다는 점에 기반한다. IFR은 이 평형 상태를 고정점 반복(Fixed-point iteration)을 통해 직접 구함으로써, 단 하나의 잔차 블록(Residual block) 파라미터만으로 무한한 깊이의 네트워크 효과를 낼 수 있도록 설계되었다.

## 📎 Related Works

### Implicit Modeling
암시적 모델링은 재귀적 피드포워드 신경망에서 은닉 상태의 고정점 방정식(Fixed-point equation)을 푸는 것을 목표로 한다. 대표적으로 Neural ODE는 ODE solver를 통해 무한 깊이의 ResNet을 시뮬레이션하며, DEQ(Deep Equilibrium Models)는 블랙박스 루트 솔버(Root-solver)를 사용하여 평형 상태를 찾는다. RAFT와 같은 광학 흐름(Optical flow) 추정 모델 또한 GRU 유닛을 통해 반복적으로 특징을 업데이트하는 방식을 취한다.

### Instance Segmentation 및 Object Detection
인스턴스 세그멘테이션은 크게 "검출 후 분할"하는 2단계 방식(Two-stage, 예: Mask R-CNN)과 직접 마스크를 예측하는 1단계 방식(One-stage, 예: SOLO)으로 나뉜다. 비디오 인스턴스 세그멘테이션(VIS)은 여기에 객체 추적(Tracking)이 추가된다. 본 논문은 이러한 다양한 프레임워크의 헤드(Head) 네트워크에서 공통적으로 사용되는 명시적 합성곱 스택을 IFR로 대체함으로써 범용적인 성능 향상을 꾀한다.

## 🛠️ Methodology

### 전체 파이프라인
IFR 모듈은 기존의 Mask head 내에 위치한 명시적 합성곱 층들을 대체하는 plug-and-play 모듈로 동작한다. 입력으로 RoIAlign을 통해 얻은 인스턴스 특징 $\mathbf{x}$와 0으로 초기화된 은닉 특징 $\mathbf{z}_0$를 받으며, 최종적으로 평형 상태의 특징 $\mathbf{z}^*$를 출력하여 마스크 예측기에 전달한다.

### Implicit Feature Refinement (IFR)
IFR은 다음과 같은 재귀적 프로세스를 무한히 반복하는 것으로 정의된다.
$$\mathbf{z}_{i+1} = F_\theta(\mathbf{z}_i; \mathbf{x})$$
여기서 $F_\theta$는 파라미터 $\theta$를 가진 변환 블록이다. $i \to \infty$일 때, $\mathbf{z}$는 다음과 같은 평형 상태 $\mathbf{z}^*$에 도달한다.
$$\mathbf{z}^* = F_\theta(\mathbf{z}^*; \mathbf{x})$$
이 식은 고정점 방정식(Fixed-point equation)으로, IFR은 이를 풀기 위해 Broyden method와 같은 루트 솔버를 사용하여 $\mathbf{z}^*$를 직접 계산한다.

### Double Residual Network
$F_\theta$의 설계가 정제 성능을 결정하므로, 본 논문은 그래디언트 소실을 방지하고 학습을 안정화하기 위해 **Double Residual Connection** 구조를 도입한다.
1. **첫 번째 연결**: 입력 특징 $\mathbf{x}$와 은닉 특징 $\mathbf{z}_i$를 먼저 합산하여 $\mathbf{R} = \mathbf{z}_i + \mathbf{x}$를 생성한다.
2. **두 번째 연결**: 표준 ResNet 블록 구조를 적용하여, $\mathbf{R}$이 두 개의 $3 \times 3$ 합성곱 층과 활성화 함수 $\sigma$(ReLU)를 거친 후 다시 $\mathbf{R}$과 더해지도록 한다.

수식으로 표현하면 다음과 같다.
$$F_\theta(\mathbf{z}_i; \mathbf{x}) = W_2(\sigma(W_1(\mathbf{R}))) + W_{skip}(\mathbf{R})$$
여기서 $W_1, W_2$는 합성곱 및 Group Normalization(GN) 층이며, $W_{skip}$은 선형 항등 매핑이다.

### Hybrid Optimization 및 학습 절차
전체 네트워크는 명시적 최적화와 암시적 최적화를 동시에 수행한다.
- **명시적 최적화**: 백본(Backbone)과 마스크 예측기(Mask predictor)는 일반적인 체인 룰(Chain rule)을 통해 역전파된다.
- **암시적 최적화**: IFR 모듈은 다음과 같은 루트 찾기 문제로 변환된다.
$$\Phi_\theta(\mathbf{z}^*; \mathbf{x}) = F_\theta(\mathbf{z}^*; \mathbf{x}) - \mathbf{z}^* = 0$$
역전파 시에는 암시적 미분(Implicit differentiation) 기법을 사용하여, 평형 상태 $\mathbf{z}^*$에서의 Jacobian 역행렬 $\mathbf{J}^{-1}_{\Phi_\theta}$을 통해 그래디언트를 계산한다.
$$\frac{\partial \mathcal{L}}{\partial (\cdot)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^*} \left( -\mathbf{J}^{-1}_{\Phi_\theta} |_{\mathbf{z}^*} \right) \frac{\partial F_\theta(\mathbf{z}^*; \mathbf{x})}{\partial (\cdot)}$$

## 📊 Results

### 실험 설정
- **데이터셋**: COCO (이미지), YouTube-VIS (비디오).
- **지표**: AP, $AP_{50}$, $AP_{75}$, $AP_S, AP_M, AP_L$.
- **비교 대상**: 명시적으로 4개의 $3 \times 3$ 합성곱을 쌓은 baseline 모델들.

### 주요 결과
1. **이미지 인스턴스 세그멘테이션 (Two-stage)**: Mask R-CNN, Cascade Mask R-CNN, Mask Scoring R-CNN 등 모든 모델에서 IFR 적용 시 성능이 향상되었다. 특히 Mask R-CNN의 경우 AP가 1.0% 향상되었으며, 큰 객체($AP_L$)에서 뚜렷한 이득을 보였다.
2. **이미지 인스턴스 세그멘테이션 (One-stage)**: CondInst, SOLO, MEInst 등 RoIAlign을 사용하지 않는 모델에서도 범용적으로 성능 향상이 확인되었다.
3. **비디오 인스턴스 세그멘테이션**: MaskTrack R-CNN과 SipMask에 적용했을 때 마스크 정확도가 약 0.9% 향상되었으며, 특히 multi-scale training 설정에서 더 큰 효과를 보였다.
4. **객체 검출 (Object Detection)**: RetinaNet, FCOS 등 1단계 검출기의 헤드에 적용하여 파라미터 수를 줄이면서도 성능을 높였다.

### 정성적 분석 및 효율성
- **수용 영역 확대**: 시각화 결과, 명시적 합성곱 기반의 특징은 coarse하고 수용 영역이 제한적인 반면, IFR로 정제된 특징은 훨씬 더 정밀하고 전역적인 정보를 포함하고 있음이 확인되었다.
- **파라미터 절감**: Mask R-CNN의 mask head 기준으로, 명시적 설계 대비 약 30.0%의 파라미터만 사용하면서도 더 높은 성능을 달성하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **효율적인 전역 정보 획득**: 단 하나의 잔차 블록 파라미터만으로 무한한 깊이의 네트워크를 시뮬레이션함으로써, 계산 비용 증가 없이 이론적으로 무한한 수용 영역을 확보할 수 있음을 증명하였다.
- **범용성**: 특정 아키텍처에 종속되지 않고, 특징 정제가 필요한 대부분의 객체 인식 프레임워크(1-stage, 2-stage, Image, Video)에 즉시 적용 가능한 plug-and-play 모듈임을 보였다.
- **수렴성 보장**: Jacobian 행렬의 spectral radius가 1보다 작음을 확인하여 평형 상태로의 수렴성을 이론적으로 뒷받침하였다.

### 한계 및 논의사항
- **솔버의 반복 횟수**: Broyden 솔버의 반복 횟수(Hyper-parameter)에 따라 성능과 추론 시간이 결정된다. 본 논문에서는 15회로 설정하였으나, 실시간성이 극도로 중요한 환경에서는 이 반복 계산이 병목이 될 가능성이 있다.
- **초기값 의존성**: 은닉 특징 $\mathbf{z}_0$를 0으로 초기화하였으나, 다른 초기화 전략이 성능에 미치는 영향에 대해서는 구체적으로 다루지 않았다.

## 📌 TL;DR

본 논문은 인스턴스 세그멘테이션의 특징 정제 과정에서 발생하는 파라미터 과다와 제한된 수용 영역 문제를 해결하기 위해, 무한 깊이 네트워크를 시뮬레이션하는 **Implicit Feature Refinement (IFR)** 모듈을 제안한다. IFR은 특징 정제를 고정점 반복 문제로 정의하고 Broyden 솔버를 통해 평형 상태의 특징을 직접 찾아내며, 이를 위해 **Double Residual Network** 구조를 도입하였다. 실험 결과, IFR은 파라미터 수를 획기적으로 줄이면서도 COCO 및 YouTube-VIS 벤치마크에서 기존 SOTA 프레임워크들의 성능을 일관되게 향상시켰으며, 특히 전역 수용 영역 확보를 통해 큰 객체의 분할 성능을 높였다. 이 연구는 향후 딥러닝 모델의 헤드 설계에서 명시적인 층 쌓기 방식의 효율적인 대안이 될 가능성이 높다.