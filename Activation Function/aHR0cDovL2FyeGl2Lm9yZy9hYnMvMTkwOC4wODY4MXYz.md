# Mish: A Self Regularized Non-Monotonic Activation Function

Diganta Misra

## 🧩 Problem to Solve

신경망에서 활성화 함수는 비선형성을 도입하고 모델의 성능 및 학습 동역학에 결정적인 역할을 합니다. 초기 활성화 함수인 Sigmoid와 TanH는 심층 신경망에서 비효율적인 것으로 드러났고, ReLU는 수렴 속도와 일반화 성능을 개선했지만 음수 입력이 0으로 고정되어 그래디언트 정보가 손실되는 "Dying ReLU" 문제를 겪었습니다. Swish는 ReLU의 단점을 개선하며 강력한 성능 향상을 보였지만, 특정 대규모 또는 복잡한 아키텍처에서는 일관성 없는 성능을 보였습니다. 본 연구는 기존 활성화 함수의 한계를 극복하고, 심층 및 복잡한 신경망 아키텍처에서 성능과 안정성을 모두 개선할 수 있는 새로운 활성화 함수를 제안하는 것을 목표로 합니다.

## ✨ Key Contributions

- **Mish 활성화 함수 제안:** $f(x) = x \tanh(\text{softplus}(x))$로 정의되는 새로운 비단조(non-monotonic), 자기-정규화(self-regularized) 활성화 함수 Mish를 제안합니다.
- **다양한 벤치마크에서 우수한 성능 입증:** ImageNet-1k 이미지 분류 및 MS-COCO 객체 탐지 등 여러 컴퓨터 비전 벤치마크와 다양한 아키텍처(ResNet, CSP-DarkNet-53, YOLOv4 등)에서 기존 활성화 함수(ReLU, Leaky ReLU, Swish) 대비 일관되게 우수한 성능을 달성했습니다.
  - 예: YOLOv4(CSP-DarkNet-53 백본)에서 Leaky ReLU 대비 MS-COCO 객체 탐지 AP$_{50}$val에서 2.1% 향상. ResNet-50에서 ReLU 대비 ImageNet-1k Top-1 정확도 약 1% 향상.
- **수학적 분석 및 정규화 효과 탐구:** Mish의 수학적 공식, 특히 1차 도함수의 $\Delta(x)$ 파라미터가 그래디언트를 부드럽게 하는 preconditioning 또는 정규화 역할을 하여 심층 신경망의 최적화를 돕는다는 직관적인 이해를 제공합니다.
- **부드러운 손실 지형 생성:** Mish가 ReLU나 Swish보다 더 부드러운 출력 및 손실 지형을 생성하여 최적화를 용이하게 하고 일반화 성능을 향상시킨다는 것을 시각적 및 실험적으로 입증했습니다.
- **안정성 및 효율성 고려:** 깊은 네트워크, 노이즈가 있는 입력, 다양한 가중치 초기화 조건에서 Mish의 안정적이고 일관된 성능을 입증하고, 계산 복잡성 증가 문제를 해결하기 위해 CUDA 기반 최적화 버전인 Mish-CUDA를 제공하여 효율성을 크게 개선했습니다.

## 📎 Related Works

- **Sigmoid & TanH:** 초기 신경망에서 사용되었으나 심층 네트워크에서는 그래디언트 소실(vanishing gradient) 문제로 비효율적임.
- **ReLU (Rectified Linear Unit) [25, 34]:** Sigmoid/TanH보다 빠른 수렴과 좋은 일반화 성능을 제공했지만, 음수 입력이 0으로 고정되는 "Dying ReLU" 문제가 존재.
- **Leaky ReLU [32], ELU [6], SELU [23]:** ReLU의 단점을 개선하기 위해 제안된 활성화 함수들.
- **Swish [37]:** $f(x) = x \text{sigmoid}(\beta x)$로 정의되며, ReLU 대비 강력한 성능 개선을 보인 활성화 함수. 부드럽고 연속적인 프로파일로 정보 전파에 유리하며, 신경망 아키텍처 검색(NAS) [52]을 통해 발견됨.
- **GELU (Gaussian Error Linear Units) [16]:** Swish와 유사한 특성(비단조성, 작은 음수 가중치 보존, 부드러운 프로파일)을 가지며 GPT-2 [36] 아키텍처에서 사용됨.
- **Preconditioning [1, 3, 29]:** 최적화 문제에서 목적 함수의 기하학적 구조를 수정하여 수렴 속도를 높이는 기법. Mish의 도함수 분석에서 유사한 효과가 추정됨.

## 🛠️ Methodology

1. **Mish 활성화 함수 공식화:**
   - Mish를 $f(x) = x \tanh(\text{softplus}(x)) = x \tanh(\ln(1+e^x))$로 수학적으로 정의합니다.
   - 이 함수는 아래로 유계($\approx-0.31$)이고 위로 비유계($\infty$)인 특성을 가집니다.
   - Swish와 유사하게 비단조성(non-monotonicity)과 자기-게이팅(self-gating) 속성을 활용합니다.
2. **수학적 및 이론적 분석:**
   - Mish의 1차 도함수를 분석하여 Swish의 1차 도함수와의 관계를 탐색하고, 특정 파라미터 $\Delta(x)$가 그래디언트의 부드러움을 증가시켜 최적화에 도움을 주는 preconditioning 효과를 제공할 수 있다는 가설을 제시합니다.
   - Mish의 부드러운 프로파일이 손실 지형을 부드럽게 만들어 최적화를 용이하게 하고 일반화 성능을 향상시킨다는 점을 시각화(5계층 랜덤 초기화 신경망의 출력 지형 및 ResNet-20의 손실 지형)를 통해 설명합니다.
3. **CIFAR-10 및 MNIST를 사용한 탐색적 실험:**
   - **초기 성능 비교:** Swish 및 다른 유사한 함수들과의 성능 비교를 위해 CIFAR-10에서 6계층 CNN 아키텍처를 사용하여 Mish의 우월성을 검증합니다.
   - **네트워크 깊이:** MNIST 데이터셋에 대해 점진적으로 깊어지는 완전 연결 네트워크를 사용하여 깊이 증가에 따른 Mish, Swish, ReLU의 테스트 정확도를 비교하여 깊은 모델에서의 안정성을 평가합니다.
   - **노이즈 강건성:** 가우시안 노이즈가 추가된 MNIST 데이터에 대해 5계층 CNN을 사용하여 노이즈 조건에서의 Mish, Swish, ReLU의 테스트 손실을 비교합니다.
   - **가중치 초기화:** 다양한 가중치 초기화 전략(Glorot, LeCun, He 등) 하에서 Mish와 Swish의 CIFAR-10에서의 성능을 비교하여 일관성을 확인합니다.
4. **다양한 벤치마크 실험:**
   - **통계적 유의미성:** SqueezeNet을 사용하여 CIFAR-10에서 23회 반복 실행으로 Mish 및 기타 활성화 함수의 평균 정확도, 평균 손실, 정확도 표준 편차를 통계적으로 비교하여 Mish의 일관된 우월성을 입증합니다.
   - **CIFAR-10 다양한 아키텍처:** ResNet, WRN, DenseNet, MobileNet, EfficientNet 등 11가지 표준 신경망 아키텍처에 Mish, Swish, ReLU를 적용하여 이미지 분류 성능을 비교합니다.
   - **ImageNet-1k 분류:** DarkNet 프레임워크를 사용하여 ResNet-18/50, PeleeNet, CSP-ResNet-50, CSP-DarkNet-53, CSP-ResNext-50 등 대규모 모델에서 Mish, Leaky ReLU, Swish의 Top-1 및 Top-5 정확도를 비교합니다. CutMix, Mosaic, Label Smoothing 등 최신 데이터 증강 기법과의 결합 성능도 평가합니다.
   - **MS-COCO 객체 탐지:** CSP-DarkNet-53 백본을 사용하는 객체 탐지 모델 및 YOLOv4 탐지기에 Mish를 적용하여 mAP@0.5, AP$_{50}$val 등 객체 탐지 성능을 ReLU 및 Leaky ReLU와 비교합니다. 다양한 데이터 증강 전략과 함께 Mish의 효과를 검증합니다.
5. **효율성 최적화:**
   - Mish의 계산 오버헤드를 줄이기 위해 CUDA 기반으로 최적화된 `Mish-CUDA` 구현을 개발하고, FP16 및 FP32 데이터 타입에서 ReLU, SoftPlus, Mish, Mish-CUDA의 순방향(Forward) 및 역방향(Backward) 패스 실행 시간을 비교하여 효율성을 검증합니다.

## 📊 Results

- **CIFAR-10 성능:** SqueezeNet에서 Mish는 가장 높은 평균 정확도(87.48%)와 낮은 평균 손실을 기록했으며, 다양한 아키텍처(ResNet, WRN, DenseNet 등)에서 ReLU 및 Swish보다 1~3%의 일관된 성능 향상을 보였습니다.
- **ImageNet-1k 성능:** ResNet-50에서 ReLU 대비 Top-1 정확도가 약 1% 향상되었으며, CSP-DarkNet-53 같은 대규모 모델에서 Leaky ReLU 대비 Top-1 정확도가 1% 이상 향상되었습니다. 특히 Swish가 특정 대규모 모델(CSP-ResNext-50)에서 성능이 크게 하락한 반면, Mish는 안정적으로 성능을 개선했습니다.
- **MS-COCO 객체 탐지:** CSP-DarkNet-53 백본에서 ReLU를 Mish로 대체하는 것만으로 mAP@0.5가 0.4% 향상되었습니다. YOLOv4 탐지기에서는 Leaky ReLU 대비 AP$_{50}$val에서 0.9%에서 2.1%까지 일관된 성능 향상을 보이며, 최신 mAP@0.5(65.7%)를 실시간(65 FPS)으로 달성했습니다.
- **안정성 및 강건성:**
  - MNIST 데이터셋에서 네트워크 깊이가 증가할 때 Mish는 Swish 및 ReLU보다 훨씬 높은 테스트 정확도를 유지하며 깊은 모델에서의 최적화 어려움에 강건함을 보였습니다.
  - 가우시안 노이즈가 있는 MNIST 입력에 대해 Mish는 ReLU 및 Swish보다 일관되게 낮은 테스트 손실을 보였습니다.
  - 다양한 가중치 초기화 전략 하에서도 Mish는 Swish보다 일관되게 우수한 성능을 나타냈습니다.
- **손실 지형:** Mish를 사용한 ResNet-20은 ReLU 및 Swish에 비해 훨씬 부드럽고 잘 조건화된 손실 지형을 가지며, 더 넓은 최소점(minima)을 제공하여 더 나은 일반화 성능을 시사했습니다.
- **효율성 (Mish-CUDA):** CUDA 기반으로 최적화된 Mish-CUDA는 Mish의 계산 오버헤드를 크게 줄여, FP32에서 SoftPlus의 PyTorch 기본 구현보다 빠르게 작동하며 실제 배포 가능성을 높였습니다.

## 🧠 Insights & Discussion

- **Mish의 장점:** Mish는 Swish의 장점인 비단조성, 작은 음수 가중치 보존, 부드러운 프로파일을 계승하면서도, 더 깊고 복잡한 신경망 아키텍처에서 Swish를 능가하는 일관된 성능과 안정성을 제공합니다. 이는 Dying ReLU 현상을 방지하고 정보 흐름을 개선하는 데 기여합니다.
- **정규화 및 최적화 효과:** Mish의 1차 도함수에 포함된 $\Delta(x)$ 파라미터는 그래디언트를 부드럽게 하는 preconditioning 역할을 하여 손실 지형을 더 매끄럽고 최적화하기 쉽게 만듭니다. 이로 인해 모델이 더 넓고 평탄한 최적점을 찾아 일반화 성능을 향상시키는 것으로 분석됩니다.
- **연속 미분 가능성:** Mish의 연속 미분 가능성은 그래디언트 기반 최적화 시 특이점을 피하고 안정적인 학습을 가능하게 합니다.
- **계산 효율성 개선:** Mish는 수학적 복잡성으로 인해 계산 비용이 높을 수 있지만, CUDA 기반 최적화인 Mish-CUDA를 통해 실제 적용 가능성을 크게 높였습니다.
- **향후 연구 방향:** Mish-CUDA의 추가적인 최적화, 다른 컴퓨터 비전 작업 및 최신 모델에서의 성능 평가, Batch Normalization 의존도를 줄이기 위한 정규화 상수 매개변수화, 그리고 $\Delta(x)$ 파라미터의 정규화 메커니즘에 대한 이론적 이해를 통해 더 원리적인 활성화 함수 설계 접근법을 모색할 수 있습니다.

## 📌 TL;DR

신경망의 활성화 함수는 성능에 핵심적이지만 기존 ReLU는 Dying ReLU 문제, Swish는 특정 대규모 모델에서 불안정성을 보였다. 본 연구는 $f(x) = x \tanh(\text{softplus}(x))$로 정의되는 새로운 자기-정규화 비단조 활성화 함수 **Mish**를 제안한다. Mish는 작은 음수 가중치를 보존하고 부드러운 미분 가능한 프로파일을 가지며, 1차 도함수의 특정 파라미터가 그래디언트를 부드럽게 하는 정규화 역할을 하여 손실 지형을 개선하고 일반화 성능을 높이는 것으로 분석된다. CIFAR-10, ImageNet-1k, MS-COCO 객체 탐지 등 다양한 컴퓨터 비전 벤치마크와 아키텍처(ResNet, YOLOv4 등)에서 Mish는 ReLU, Leaky ReLU, Swish보다 일관되게 우수한 성능을 달성했다. 계산 복잡성 문제를 해결하기 위해 CUDA 기반 최적화 버전인 Mish-CUDA도 제공하여 실제 적용 가능성을 높였다.
