# Recurrent Inference Machine for Medical Image Registration

Yi Zhang, Yidong Zhao, Hui Xue, Peter Kellman, Stefan Klein, Qian Tao (2024)

## 🧩 Problem to Solve

의료 영상 등록(Medical Image Registration)은 여러 영상 간의 복셀(voxel)을 정렬하여 정성적 또는 정량적 분석을 가능하게 하는 필수적인 과정이다. 기존의 접근 방식은 크게 두 가지로 나뉜다. 첫째, 전통적인 최적화 기반 방법(Optimization-based methods)은 정확도가 높지만, 비볼록 최적화(non-convex optimization)의 복잡성으로 인해 계산 시간이 매우 길어 임상 현장에서의 실시간 적용이 어렵다. 둘째, 최근의 딥러닝 기반 방법(Deep learning-based methods)은 추론 속도가 매우 빠르지만, 한 번의 추론으로 변형 필드(deformation field)를 예측하는 'One-step inference' 방식은 변형량이 클 때 정확도가 떨어지며, 일관된 예측을 위해 막대한 양의 학습 데이터가 필요하다는 단점이 있다.

본 논문의 목표는 최적화 기반 방법의 정확도와 딥러닝 기반 방법의 속도 장점을 결합하는 것이다. 특히, 학습 데이터 효율성(data efficiency)을 극대화하면서도 복잡한 변형을 정확하게 포착할 수 있는 반복적 추론 프레임워크를 제안하여, 적은 데이터만으로도 높은 성능을 내는 등록 알고리즘을 구축하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Recurrent Inference Machine (RIM)**을 의료 영상 등록 문제에 도입하여, 등록 과정을 하나의 '최적화 문제'로 정의하고 이 **최적화 규칙(update rule) 자체를 학습**하는 메타 러닝(meta-learning) 솔버를 설계한 것이다.

주요 기여 사항은 다음과 같다:
- **RIIR(Recurrent Inference Image Registration) 프레임워크 제안**: 명시적인 순방향 모델(forward model)이 없는 고차원 최적화 문제인 영상 등록을 해결하기 위해 RIM을 확장한 반복적 추론 네트워크를 제안하였다.
- **데이터 효율성 극대화**: 명시적인 유사도 손실의 그래디언트(gradient) 정보를 네트워크 입력으로 제공함으로써, 모델이 처음부터 변형을 예측하는 부담을 줄이고 최적화 경로를 학습하게 하여 매우 적은 양의 데이터(최소 5%)만으로도 경쟁력 있는 성능을 확보하였다.
- **공간적 상관관계 유지 및 메모리 상태 활용**: ConvGRU를 도입하여 이미지의 공간적 정보를 보존하고, 은닉 상태(hidden states)를 통해 최적화 과정을 추적함으로써 복잡한 등록 문제의 수렴 성능을 향상시켰다.

## 📎 Related Works

### 기존 연구 및 한계
- **One-step DL 기반 방법**: VoxelMorph와 같은 방법들은 빠른 추론이 가능하지만, 큰 변형을 예측하는 데 한계가 있으며 대규모 데이터셋이 필수적이다.
- **반복적 DL 기반 방법**: RCVM이나 GraDIRN 등은 여러 단계의 추론을 통해 정밀도를 높이려 한다. GraDIRN은 유사도 손실의 그래디언트를 사용하지만, 이를 네트워크 내부에서 처리하기보다 단순히 업데이트에 추가하는 방식을 취하며, 내부 상태(internal states)를 유지하지 않는다.
- **Meta-Learning 및 RIM**: RIM은 역문제(inverse problems)를 해결하기 위해 제안되었으며, 주로 물리적 순방향 모델이 명확한 경우(예: MRI 재구성)에 사용되었다. 하지만 영상 등록은 이러한 닫힌 형태(closed-form)의 순방향 모델이 존재하지 않아 RIM을 그대로 적용하기 어렵다.

### 차별점
RIIR은 RIM의 개념을 확장하여 명시적 순방향 모델 없이도 작동하도록 설계되었으며, 단순한 반복 예측이 아니라 **그래디언트 입력과 은닉 상태를 결합**하여 최적화 프로세스를 학습한다는 점에서 기존의 반복적 DL 방법들과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
RIIR은 입력 영상 쌍 $(I_{mov}, I_{fix})$에 대해 $T$번의 반복 단계를 거쳐 최종 변형 필드 $\phi$를 도출한다. 각 단계 $t$에서 변형 필드는 다음과 같이 업데이트된다:
$$\phi_{t+1} = \phi_t + \Delta \phi_t$$
여기서 $\phi_0$는 항등 변환(identity mapping)으로 초기화된다. $\Delta \phi_t$는 **RIIR Cell**이라는 recurrent 네트워크 $g_\theta$에 의해 계산된다.

### RIIR Cell의 입력 및 구성
RIIR Cell은 다음과 같은 요소들을 채널 방향으로 결합(concatenation)하여 입력으로 받는다:
1. 현재 변형 필드: $\phi_t$
2. 내적 손실의 그래디언트: $\nabla_{\phi_t} L_{inner}(I_{mov} \circ \phi_t, I_{fix})$
3. 워핑된 이동 영상 및 고정 영상: $I_{mov} \circ \phi_t, I_{fix}$
4. 이전 단계의 은닉 상태: $h_t$

네트워크 내부에서는 공간적 상관관계를 보존하기 위해 **ConvGRU (Convolutional Gated Recurrent Unit)**를 사용하며, 두 단계의 은닉 상태 $h_t = \{h_1^t, h_2^t\}$를 유지하여 최적화 진행 상황을 기억한다. 이는 L-BFGS와 같은 2차 최적화 알고리즘이 과거의 정보를 활용하는 것과 유사한 효과를 낸다.

### 손실 함수 및 학습 절차
학습은 '내적 손실(inner loss)'과 '외적 손실(outer loss)'의 이단계 구조로 이루어진다.

- **Inner Loss ($L_{inner}$)**: 매 반복 단계에서 현재 정렬 상태를 평가하는 유사도 측정항이다. MSE, NCC, NMI 등이 사용되며, 이 값의 그래디언트가 네트워크의 입력으로 들어간다.
- **Outer Loss ($L_{outer}$)**: 네트워크의 파라미터 $\theta$를 업데이트하기 위한 전체 목적 함수이다. 모든 반복 단계의 손실을 가중 합산하여 계산한다:
$$L_{outer}(\theta) = \sum_{t=1}^{T} w_t (L_{sim}(I_{mov} \circ \phi_t, I_{fix}) + \lambda L_{reg}(\phi_t))$$
여기서 $L_{reg}$는 변형 필드의 매끄러움(smoothness)을 강제하는 정규화 항(Diffusion, Curvature, Elastic 등)이며, $\lambda$는 가중치이다. $w_t$는 균등 가중치 또는 지수 가중치를 사용하여 각 단계의 기여도를 조절한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 
    - OASIS (뇌 MRI, 대상 간 등록)
    - NLST (폐 CT, 대상 내 호흡 주기 등록)
    - mSASHA (심장 MRI, 다중 파라미터 시계열 등록)
- **비교 대상**: VoxelMorph(one-step), GraDIRN, LapIRN, RCVM, L2O(iterative), elastix(traditional)
- **평가 지표**: Dice Score, Hausdorff Distance (HD), Target Registration Error (TRE), Jacobian Determinant (변형의 위상 보존 확인), PCA 기반 지표($D_{PCA1}, D_{PCA2}$)

### 주요 결과
1. **데이터 효율성**: RIIR은 학습 데이터가 매우 제한적인 상황(5% 사용 시)에서도 다른 딥러닝 기반 방법들보다 월등한 성능을 보였다. 특히 NLST 데이터셋의 큰 변형을 포착하는 데 있어 매우 강력한 모습을 보였다.
2. **정량적 성능**: 100% 데이터를 사용했을 때, OASIS 데이터셋에서 Dice score 0.756, NLST에서 TRE 2.21mm를 기록하며 대부분의 baseline 모델을 상회하거나 대등한 수준의 성능을 보였다.
3. **추론 특성**: 반복 횟수 $T$가 증가할수록 정확도는 향상되지만, VRAM 소비량과 추론 시간이 선형적으로 증가하는 트레이드-오프가 관찰되었다. (예: $T=4$일 때 8GB $\rightarrow T=12$일 때 24GB VRAM 사용)

## 🧠 Insights & Discussion

### 강점 및 해석
RIIR의 가장 큰 강점은 **"최적화 프로세스의 학습"**에 있다. 단순히 결과물인 변형 필드를 예측하는 것이 아니라, 유사도 그래디언트를 입력으로 받아 이를 어떻게 수정해야 할지(update rule)를 학습함으로써, 데이터가 적은 상황에서도 일반화 성능이 뛰어나다. 또한, ConvGRU를 통한 은닉 상태의 도입이 복잡한 고차원 최적화 경로를 추적하는 데 기여했음을 확인하였다.

### 한계 및 비판적 논의
1. **메모리 효율성**: Recurrent 구조의 특성상 모든 단계의 연산 그래프를 유지해야 하므로, VRAM 사용량이 매우 높다. 이는 고해상도 3D 영상 적용 시 심각한 제약이 될 수 있다.
2. **구조적 단순함**: LapIRN과 같은 다해상도(multi-resolution) 구조를 채택하지 않아, 매우 거대한 전역적 변형을 잡는 데에는 구조적인 한계가 있을 수 있다.
3. **정규화 의존성**: 실험 결과, 여전히 정규화 항의 종류와 가중치 $\lambda$ 설정에 따라 성능 편차가 발생한다. 이는 향후 적응형 정규화(adaptive regularization) 도입의 필요성을 시사한다.

## 📌 TL;DR

본 논문은 메타 러닝 프레임워크인 **Recurrent Inference Machine (RIM)**을 의료 영상 등록에 적용한 **RIIR**를 제안하였다. RIIR는 유사도 손실의 그래디언트와 ConvGRU 기반의 은닉 상태를 입력으로 사용하여 변형 필드를 반복적으로 정밀화한다. 실험을 통해 **학습 데이터가 5%만 있어도 기존 방법들보다 우수한 성능**을 낼 수 있음을 입증하며 매우 높은 데이터 효율성을 보여주었다. 이 연구는 데이터 수집이 어려운 의료 현장에서 적은 데이터로도 고정밀 정렬을 수행할 수 있는 실용적인 방향성을 제시한다.