# Resource Optimized Neural Architecture Search for 3D Medical Image Segmentation

Woong Bae, Seungho Lee, Yeha Lee, Beomhee Park, Minki Chung, and Kyu-Hwan Jung (2019)

## 🧩 Problem to Solve

본 논문은 3D 의료 영상 분할(3D Medical Image Segmentation) 분야에서 신경망 구조 탐색(Neural Architecture Search, NAS)을 효율적으로 적용하는 문제를 해결하고자 한다. 3D 의료 영상은 일반적으로 데이터의 차원이 매우 크기 때문에, 기존의 NAS 방법론들을 그대로 적용할 경우 GPU 연산 부하가 극심하고 학습 시간이 매우 오래 걸린다는 치명적인 한계가 있다.

특히, 기존의 자동화된 설계 방식들은 수동으로 설계된 최신 모델(State-of-the-art)들의 성능에 미치지 못하는 경우가 많았다. 따라서 본 연구의 목표는 적은 양의 계산 자원(단일 GPU)과 짧은 학습 시간만으로도 수동 설계 모델보다 우수한 성능을 내는 자원 최적화된 NAS 프레임워크인 RONASMIS를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 3D 의료 영상의 특성을 고려하여 연산 비용이 높은 Micro search(세부 셀 구조 탐색) 대신 **Macro search(전체적인 네트워크 구조 탐색)**에 집중하는 것이다. 

주요 기여 사항은 다음과 같다.
1. **자원 최적화된 탐색 공간 설계**: 3D 의료 영상의 비등방성(Anisotropic) 특성을 반영하여 입력 패치 크기와 다운샘플링 양을 탐색 공간에 포함시켰다.
2. **Parameter Sharing 기반의 RL 컨트롤러**: 가중치 공유(Parameter Sharing) 방식을 채택하여 최적 구조를 찾은 후 네트워크를 처음부터 다시 학습(Retraining)시켜야 하는 기존 NAS의 비효율성을 제거하였다.
3. **메모리 효율적 아키텍처 설계**: Concatenation 기반의 Skip connection 대신 Element-wise sum 기반의 연결을 사용하고, PyTorch 등의 프레임워크에서 메모리 효율이 떨어지는 Depth-wise convolution을 일반 Convolution으로 대체하여 GPU 메모리 사용량을 최소화하였다.

## 📎 Related Works

기존의 3D 의료 영상 분할에서는 U-Net이나 Deep Supervision과 같은 수동 설계 모델들이 주를 이루었으나, 하이퍼파라미터 튜닝과 구조 설계에 막대한 시간과 노력이 소요되는 문제가 있었다. 이를 해결하기 위해 자연어 처리나 2D 이미지 분야에서 NAS가 활발히 연구되었으나, 고차원 3D 데이터에 적용하기에는 연산 비용이 너무 컸다.

최근 SCNAS와 같은 3D 의료 영상용 NAS 연구가 등장하였으나, 여전히 Micro search 공간을 탐색하는 경향이 있어 많은 GPU 자원을 요구하며, 수동으로 정교하게 설계된 nnU-Net과 같은 모델의 성능을 완전히 뛰어넘는 데 어려움이 있었다. 본 논문은 이러한 Micro search의 부담을 줄이고 Macro search에 집중함으로써 효율성과 성능을 동시에 잡고자 하였다.

## 🛠️ Methodology

### 1. Resource-Optimized Search Space
본 논문은 3D 의료 영상의 특성을 반영하여 다음과 같은 탐색 공간을 구성하였다.
- **비등방성 고려**: 입력 영상의 높이($H$), 너비($W$), 깊이($D$)가 서로 다른 점을 고려하여, 입력 패치 크기를 탐색 범위에 포함하였다. 패치 크기는 다음 식에 의해 결정된다.
  - 높이 및 너비 패치 크기: $$\text{Search Space of Patch Size } H/W = \lfloor \frac{\max(H, W)}{S^4} \rfloor \times S^4 - S^4 \times \{0,1,2,3,4\}$$
  - 깊이 패치 크기: $$\text{Search Space of Patch Size Depth} = \lfloor \frac{D}{S^4} \rfloor \times S^4 - S^4 \times \{0,1,2,3,4\}$$
  (여기서 $S$는 각 스테이지의 stride 파라미터이다.)
- **구성 요소**: 풀링(Pooling) 양, 3D Convolution의 Dilation rate, 활성화 함수(Activation function), 그리고 Skip connection의 연결 지점을 탐색 범위로 설정하였다.
- **정규화**: 과적합을 방지하고 컨트롤러의 안정적인 구조 생성을 위해 일부 연산이나 연결을 비활성화하는 Drop-path regularization(Zero operation)을 도입하였다.

### 2. Base Architecture
기본 구조는 U-Net을 수정하여 사용하며, 다음과 같은 특징을 갖는다.
- **Deep Supervision**: DeepLabV3+의 $1 \times 1 \times 1$ convolution skip connection과 Deep Supervision 기법을 결합하였다.
- **Normalization**: GPU 메모리 절약을 위해 Batch Normalization 대신 Instance Normalization을 사용하였다.
- **Skip Connection**: 메모리 사용량을 줄이기 위해 Concatenation 후 $1 \times 1 \times 1$ convolution을 적용하는 방식 대신, 채널 크기를 맞춘 후 Element-wise sum을 수행하는 매칭 연산(Matching operation)을 사용하였다.

### 3. Controller 및 학습 절차
- **RL 기반 컨트롤러**: ENAS의 Parameter sharing 방식을 사용하여 RNN(LSTM) 기반의 컨트롤러를 학습시켰다. 이는 새로운 구조를 샘플링할 때마다 네트워크를 처음부터 학습시키지 않고 기존 가중치를 공유함으로써 탐색 시간을 획기적으로 단축한다.
- **보상(Reward)**: 각 에피소드마다 컨트롤러가 20개의 자식 네트워크(Child network)를 생성하며, 검증 데이터셋에 대한 환자별 Dice score를 보상으로 받아 컨트롤러를 업데이트한다.
- **손실 함수**: 자식 네트워크의 학습에는 Dice loss가 사용된다.

## 📊 Results

### 실험 설정
- **데이터셋**: Medical Segmentation Decathlon (MSD)의 Brain, Heart, Prostate 3D 영상 데이터셋을 사용하였다.
- **비교 대상**: 수동 설계의 정점인 nnU-Net과 기존 NAS 방식인 SCNAS를 기준으로 비교하였다.
- **평가 지표**: Mean Dice score를 사용하여 5-fold cross-validation으로 평가하였다.
- **자원**: 단일 RTX 2080Ti (10.8GB GPU memory)를 사용하였다.

### 주요 결과
정량적 결과는 Table 2에 제시되어 있으며, RONASMIS는 모든 태스크에서 SCNAS를 상회하고 nnU-Net과 대등하거나 더 높은 성능을 보였다.

- **Dice Score 결과**:
  - Brain Tumor: RONASMIS ($\mathbf{74.14}$) vs nnU-Net ($74.00$)
  - Heart: RONASMIS ($\mathbf{92.72}$) vs nnU-Net ($92.70$)
  - Prostate: RONASMIS ($\mathbf{75.71}$) vs nnU-Net ($74.54$)

특히 주목할 점은 nnU-Net이 다양한 데이터 증강(V.D.A), 앙상블(Ensemble), Test Time Augmentation(T.T.A), 후처리(P.P)를 모두 적용한 결과인 반면, RONASMIS는 단순한 Horizontal flip만 사용하고 이러한 추가 기법을 전혀 사용하지 않고도 더 높은 성능을 달성했다는 점이다. 또한, 추론 시 연산량이 많은 Overlapped patch-wise 방식이 아닌 One-shot inference 방식을 사용하여 속도 면에서도 이점을 가졌다.

학습 시간은 Brain 3.1일, Heart 1.39일, Prostate 0.35일로 매우 빠르게 수렴하였다.

## 🧠 Insights & Discussion

본 논문은 3D 의료 영상이라는 특수한 도메인에서 NAS를 적용하기 위해 '무엇을 포기하고 무엇에 집중해야 하는가'를 명확히 보여준다. 

**강점 및 통찰**:
- **Macro search의 효율성**: 모든 가능한 세부 연산을 탐색하는 Micro search 대신, 패치 크기, 다운샘플링, Skip connection 지점과 같은 Macro 수준의 구조적 결정이 3D 의료 영상 성능에 더 결정적인 영향을 미친다는 것을 입증하였다.
- **실질적인 자원 최적화**: 단순히 알고리즘적 개선뿐만 아니라, PyTorch의 Depth-wise convolution 메모리 효율 문제와 같은 구현 레벨의 디테일을 파악하여 실제 GPU 메모리 점유율을 낮춘 점이 인상적이다.
- **컨트롤러의 안정성**: 엔트로피(Entropy)가 감소하고 보상(Reward)이 꾸준히 증가하는 그래프를 통해 컨트롤러가 안정적으로 최적 구조를 찾아가고 있음을 증명하였다.

**한계 및 논의**:
- **데이터셋의 제한**: MSD 데이터셋만 사용하였으며, 더 다양한 의료 영상 도메인에서의 일반화 성능에 대한 검증이 추가로 필요하다.
- **One-shot Inference**: 추론 속도를 위해 One-shot 방식을 택했으나, 저자들이 언급했듯이 이는 성능 저하의 원인이 될 수 있다. 만약 Overlapped patch-wise 방식을 적용했다면 더 압도적인 성능 향상이 있었을 가능성이 크다.

## 📌 TL;DR

본 논문은 3D 의료 영상 분할을 위해 메모리 효율적인 **Macro search 중심의 NAS 프레임워크(RONASMIS)**를 제안한다. Parameter sharing과 자원 최적화된 탐색 공간 설계를 통해, 단일 GPU만으로도 매우 짧은 시간 내에 nnU-Net과 같은 SOTA 모델을 뛰어넘는 최적의 구조를 자동으로 찾아낼 수 있음을 보였다. 이 연구는 고차원 의료 영상 분석에서 AutoML을 실무적으로 적용할 수 있는 효율적인 가이드라인을 제시했다는 점에서 큰 의미가 있다.