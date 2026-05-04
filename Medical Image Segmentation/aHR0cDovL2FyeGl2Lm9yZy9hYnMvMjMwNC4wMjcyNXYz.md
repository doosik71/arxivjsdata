# FMG-Net and W-Net: Multigrid Inspired Deep Learning Architectures For Medical Imaging Segmentation

Adrian Celaya, Beatrice Riviere, David Fuentes (2023)

## 🧩 Problem to Solve

본 연구는 의료 영상 분할(Medical Imaging Segmentation)에서 Convolutional Neural Networks(CNNs)가 직면한 핵심적인 한계점인 미세 규모 특징(fine-scale features)의 보존과 이미지 스케일 변화에 대한 대응 능력을 해결하고자 한다. 특히 BraTS(Brain Tumor Segmentation)와 같이 크기와 모양이 매우 다양하고 복잡한 뇌종양 하위 구성 요소를 분할해야 하는 작업에서는 최신 기법들을 사용하더라도 상당한 오차가 발생하는 문제가 존재한다. 따라서 본 논문의 목표는 수치 해석 분야의 Geometric Multigrid Methods(GMMs) 원리를 CNN 아키텍처에 통합하여, 다양한 스케일의 특징을 보다 효율적으로 캡처하고 학습 효율성을 높인 FMG-Net과 W-Net 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 선형 방정식 시스템을 풀기 위한 수치적 방법론인 Geometric Multigrid Methods의 계층적 구조를 딥러닝 모델에 이식하는 것이다. 기존의 U-Net 아키텍처가 GMM의 V-cycle과 매우 유사한 구조를 가지고 있다는 점에 착안하여, V-cycle보다 더 복잡한 상호작용을 통해 빠른 수렴성과 높은 정확도를 보이는 Full Multigrid(FMG) 및 W-cycle 방식을 CNN에 적용하였다. 이를 통해 네트워크의 파라미터 수를 줄이면서도 서로 다른 해상도 그리드 간의 풍부한 특징 상호작용을 유도하여 분할 성능을 향상시키고 학습 수렴 속도를 가속화하였다.

## 📎 Related Works

기존의 CNN 아키텍처, 특히 nnU-Net이나 U-Net은 자연 이미지나 의료 영상이 다양한 해상도의 데이터를 포함하고 있다는 직관에 따라 다운샘플링과 업샘플링을 반복하는 계층적 구조를 사용한다. 이는 GMM의 기본 개념과 일치한다.

관련 연구로 He와 Xu(2019)는 MgNet을 통해 multigrid 방법과 CNN의 유사성을 탐구하고 이미지 분류 작업에서 성능을 입증하였으나, 이를 분할(segmentation) 작업으로 확장하지는 않았다. 또한 Celaya 등은 V-cycle 구조와 U-Net의 유사성을 바탕으로 파라미터 수를 획기적으로 줄인 PocketNet 패러다임을 제안하였다. PocketNet은 다운샘플링 시 채널 수를 두 배로 늘려야 한다는 기존의 통념에 의문을 제기하며 효율성을 입증하였다. 본 연구는 이러한 V-cycle 기반의 접근을 넘어, 더 복잡한 계층 구조를 가진 FMG와 W-cycle을 분할 네트워크에 도입했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

본 논문은 GMM의 FMG-cycle과 W-cycle을 모사한 두 가지 아키텍처, **FMG-Net**과 **W-Net**을 제안한다. 두 네트워크 모두 3D 의료 영상 분할을 위해 설계되었으며, 기본적으로 다운샘플링(encoder)과 업샘플링(decoder) 과정을 거치지만, 그 연결 방식(skip connections)에서 U-Net과 차이를 보인다.

### 핵심 구성 요소 및 설계 규칙

각 네트워크의 깊이(depth)는 사용된 서로 다른 해상도 그리드의 수로 정의된다. 각 Convolutional block은 두 번의 convolution 연산 후 Batch Normalization과 ReLU 비선형 함수가 뒤따르는 구조이다. 특히, 파라미터 효율성을 위해 다운샘플링 시 채널 수를 늘리지 않는 PocketNet 패러다임을 적용하였다.

스킵 연결(Skip Connection)은 다음과 같은 엄격한 규칙을 따른다:

1. **Encoder branch**: 특징을 더 거친(coarser) 그리드로 전달할 때, 해당 그리드 레벨의 모든 후속 업샘플링 연산으로 특징을 전달한다.
2. **Peak (Up $\rightarrow$ Down)**: 업샘플링 직후 다시 다운샘플링이 일어나는 '피크' 지점에서는, 해당 피크의 특징을 동일한 그리드 레벨의 다음 특징 세트로만 전달한다.

### 학습 절차 및 손실 함수

학습을 위해 Dice loss와 Cross-entropy를 결합한 손실 함수를 사용하였으며, Adam 옵티마이저를 통해 최적화를 진행하였다.

$$ \text{Loss} = \text{Dice Loss} + \text{Cross-Entropy Loss} $$

학습은 5-fold 교차 검증(cross-validation) 방식을 사용하였으며, 128x128x128 크기의 패치 사이즈와 z-score 강도 정규화를 적용하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: BraTS 2020 데이터셋 (369개의 다중 모달리티 스캔)
- **분할 대상**: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)
- **평가 지표**: Dice Coefficient (겹침 정도), 95th percentile Hausdorff Distance (HD95, 경계 거리 오차), Average Surface Distance (ASD, 표면 거리 오차)
- **비교 대상**: 표준 3D U-Net (깊이 3, 4, 5)

### 정량적 결과

실험 결과, FMG-Net과 W-Net은 모든 깊이에서 U-Net보다 전반적으로 우수한 성능을 보였다. 특히 깊이가 3일 때 FMG-Net이 가장 뛰어난 성능을 보였으며, 깊이가 4와 5로 증가함에 따라 FMG-Net과 W-Net이 서로 유사하게 높은 성능을 유지하였다.

가장 주목할 점은 파라미터 수의 획기적인 감소이다. 예를 들어 깊이 5의 경우, U-Net은 약 9,050만 개의 파라미터를 가지는 반면, FMG-Net은 약 285만 개, W-Net은 약 789만 개만으로도 더 나은 성능을 기록하였다.

### 효율성 및 수렴 속도

- **학습 속도 및 메모리**: FMG-Net은 U-Net 대비 학습 단계당 시간을 7%~20% 단축시켰으며, GPU 메모리 사용량을 약 20% 절감하였다. W-Net 역시 효율적이었으나 FMG-Net보다는 연산 비용이 높았다.
- **수렴성**: 손실 함수 곡선 분석 결과, FMG-Net과 W-Net은 U-Net보다 훨씬 적은 에포크(epoch) 내에 더 낮은 손실 값에 도달하며 빠르게 수렴하는 특성을 보였다.

## 🧠 Insights & Discussion

본 연구는 의료 영상 분할에서 네트워크의 성능이 단순히 파라미터의 양에 비례하는 것이 아니라, **다양한 해상도 그리드 간의 상호작용의 풍부함(richness of interactions)**에 달려 있음을 시사한다. GMM의 원리를 적용하여 특징 간의 상호작용을 복잡하게 설계함으로써, 적은 파라미터로도 미세한 종양 구조를 더 정확하게 포착할 수 있었다.

또한, 네트워크의 깊이가 증가한다고 해서 반드시 정확도가 선형적으로 증가하지 않는다는 점이 관찰되었다. 이는 깊이가 깊어질수록 수렴 속도는 빨라질 수 있으나 최종 정확도는 정체될 수 있다는 multigrid 이론과 일치하며, BraTS 데이터셋에 대한 기존 연구 결과와도 맥을 같이 한다.

다만, 본 연구는 U-Net만을 비교 대상으로 삼았다는 한계가 있다. 향후 nnU-Net이나 HRNet과 같은 최신 아키텍처와의 비교 연구가 필요하며, BraTS 외의 다른 의료 영상 데이터셋(예: LiTS)에서도 일반화 성능을 검증할 필요가 있다.

## 📌 TL;DR

본 논문은 수치 해석의 Geometric Multigrid Methods(FMG, W-cycle)를 딥러닝에 접목하여 **FMG-Net**과 **W-Net**이라는 새로운 3D 분할 아키텍처를 제안하였다. 이 모델들은 U-Net 대비 **훨씬 적은 파라미터**를 사용함에도 불구하고, **더 빠른 학습 수렴 속도**와 **더 높은 분할 정확도**를 달성하였다. 이는 의료 영상 분할에서 단순한 모델 크기 확장보다 효율적인 다중 스케일 특징 상호작용 설계가 더 중요함을 입증하며, 향후 고효율 의료 AI 모델 설계에 중요한 방향성을 제시한다.
