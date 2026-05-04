# BUSU-Net: An Ensemble U-Net Framework for Medical Image Segmentation

Wei Hao Khoong(2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 의료 영상 분석, 특히 망막 혈관 분할(Retinal Vessel Segmentation)의 정확도와 효율성을 높이는 것이다. 의료 영상 분할은 임상 진단을 위한 근거를 제공하고 의사의 정확한 진단을 돕는 데 매우 중요한 역할을 한다. 하지만 수동 분할 작업은 많은 시간과 비용이 소요될 뿐만 아니라, 작업자의 숙련도에 따른 인적 오류(human error)가 발생할 가능성이 크다. 따라서 이를 자동화하여 정확하고 빠르게 처리할 수 있는 딥러닝 기반의 분할 프레임워크를 구축하는 것이 본 연구의 목표이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 서로 다른 깊이를 가진 U-Net 변형 구조를 앙상블(Ensemble) 형태로 체이닝(Chaining)하는 것이다. 구체적으로, BCDU-Net 구조를 기반으로 하여 첫 번째 네트워크는 더 깊게(Big-U), 두 번째 네트워크는 상대적으로 얕게(Small-U) 설계한 **BUSU-Net**을 제안한다. 동일한 크기의 네트워크를 앙상블 하는 것보다, 비대칭적인 깊이를 가진 네트워크를 결합하는 것이 더 우수한 성능을 낼 수 있다는 직관을 바탕으로 설계되었다.

## 📎 Related Works

본 논문에서는 다음과 같은 기존 연구들을 언급하며 차별점을 제시한다.

- **U-Net 및 FCN**: 의료 영상 분할의 기초가 되는 구조로, 특히 U-Net은 적은 양의 데이터로도 높은 성능을 보여주었다.
- **LadderNet**: 여러 개의 U-Net을 체인 형태로 연결한 다중 경로 CNN이다. 인코더와 디코더의 특징 맵을 단순히 합산하지 않고 연결(Concatenate)하여 더 복잡한 특징을 기록할 수 있게 설계되었다.
- **BCDU-Net**: U-Net의 확장판으로, Dense Connection과 Bi-directional Convolutional LSTMs (BConvLSTM)를 도입하여 중복 특징 학습 문제를 완화하고 양방향 데이터 의존성을 처리하였다.
- **NAS-Unet**: 신경망 구조 탐색(Neural Architecture Search)을 통해 최적의 구조를 찾는 방식이다.

본 논문은 LadderNet의 다중 경로 구조와 BCDU-Net의 세부 구성 요소(BConvLSTM, Dense Connection)를 결합하되, 앙상블 구성 시 네트워크의 깊이를 다르게 설정함으로써 기존의 단일 네트워크나 동일 크기 앙상블보다 성능을 개선하고자 하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
BUSU-Net은 두 개의 BCDU-Net을 직렬로 연결한 앙상블 프레임워크이다. 
- **BUSU-Net**: 총 108개의 레이어로 구성되며, 앞단에 더 깊은 **Big-U** 네트워크가 위치하고 그 뒤에 표준 BCDU-Net 크기의 **Small-U** 네트워크가 이어진다.
- **LightBUSU-Net**: 자원 제한 환경을 고려한 경량 버전으로, 총 43개의 레이어로 구성된다. Big-U와 Small-U 모두 기존 BUSU-Net보다 얕게 설계되었다.

### 2. 주요 구성 요소 및 역할
- **Densely Connected Convolutions**: 연속적인 컨볼루션 레이어에서 발생하는 중복 특징 학습 문제를 방지하기 위해 사용된다.
- **Bi-directional Convolutional LSTMs (BConvLSTM)**: 일반적인 ConvLSTM이 전방향(forward) 데이터만 처리하는 것과 달리, BConvLSTM은 전방향과 후방향(backward) 모두에서 데이터 의존성을 처리하여 더 정확한 결정을 내린다.
- **Batch Normalization**: 네트워크 레이어의 입력을 표준화하여 학습의 안정성을 높이고 속도를 향상시킨다.

### 3. 학습 및 추론 절차
- **입력 처리**: DRIVE 데이터셋의 이미지 수가 적기 때문에, 20장의 훈련 이미지에서 약 190,000개의 패치를 랜덤하게 추출하여 학습(170,000개) 및 검증(19,000개)에 사용하였다.
- **학습 환경**: TensorFlow 1.12, NVIDIA Tesla V100-32GB GPU 환경에서 학습되었다.
- **손실 함수 및 목표 변수**: 원문 텍스트 내에 구체적인 손실 함수(Loss Function)나 학습률(Learning Rate)에 대한 명시적인 설명은 포함되어 있지 않다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: DRIVE (Retinal Fundus Images). 훈련 데이터 20장, 테스트 데이터 20장으로 구성된다.
- **평가 지표**: Accuracy, Sensitivity, Specificity, F1-Score, AUC (Area Under Curve)를 사용하였다.
- **비교 대상**: COSFIRE, Cross-Modality, U-Net, DeepModel, RU-Net, R2U-Net, LadderNet, BCDU-Net, Mi-UNet 등 최신 SOTA 모델들과 비교하였다.

### 2. 정량적 결과
실험 결과, **BUSU-Net**은 대부분의 지표에서 기존 SOTA 네트워크보다 우수한 성능을 보였다. 특히 F1-Score에서 가장 높은 수치를 기록하였다. 
또한, **LightBUSU-Net**은 전체 레이어 수가 훨씬 적음에도 불구하고, Sensitivity(민감도) 지표에서는 BUSU-Net을 포함한 모든 비교 모델 중 가장 뛰어난 성능을 보였다.

| Method | Accuracy | Sensitivity | Specificity | AUC | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Mi-UNet | 0.9559 | 0.8099 | 0.9772 | 0.9799 | 0.8231 |
| **LightBUSU-Net** | 0.9539 | **0.8281** | 0.9723 | 0.9781 | 0.8207 |
| **BUSU-Net** | **0.9560** | 0.8113 | 0.9771 | **0.9799** | **0.8243** |

### 3. 추가 분석
저자는 두 개의 동일한 BCDU-Net을 연결한 **LadderBCDU-Net**을 벤치마크로 설계하여 비교하였다. 실험 결과, LadderBCDU-Net은 단일 BCDU-Net보다 성능이 좋지 않았으며, 이는 앙상블 구성 시 네트워크의 깊이를 다르게 설정하는 것(비대칭 구조)이 성능 향상의 핵심임을 시사한다.

## 🧠 Insights & Discussion

### 1. 강점 및 발견
본 연구는 단순히 모델의 크기를 키우거나 동일한 모델을 여러 개 쌓는 것보다, **깊이가 다른 모델을 조합하는 비대칭 앙상블**이 의료 영상 분할에서 더 효과적일 수 있음을 입증하였다. 특히 경량 모델인 LightBUSU-Net이 특정 지표(Sensitivity)에서 매우 높은 성능을 보인 점은 실제 자원이 제한된 의료 환경에서의 적용 가능성을 보여준다.

### 2. 한계 및 미해결 질문
- **컴퓨팅 자원의 한계**: 연산 능력의 한계로 인해 두 개의 네트워크 앙상블까지만 실험이 가능했다. 더 많은 수의 네트워크를 체이닝했을 때의 성능 향상 여부는 확인되지 않았다.
- **일반화 검증 부족**: DRIVE 데이터셋 하나만 사용하여 성능을 평가했으므로, 다른 의료 영상 데이터셋에서도 동일한 효과가 나타날지는 추가 검증이 필요하다.
- **상세 설정 누락**: 학습에 사용된 구체적인 하이퍼파라미터와 손실 함수에 대한 설명이 부족하여 재현성에 어려움이 있을 수 있다.

## 📌 TL;DR

본 논문은 비대칭적 깊이를 가진 두 개의 BCDU-Net을 연결한 **BUSU-Net** 프레임워크를 제안하여 망막 혈관 분할 성능을 향상시켰다. 실험을 통해 동일한 깊이의 앙상블보다 깊이가 다른 모델을 조합하는 것이 더 우수함을 보였으며, 경량화 버전인 LightBUSU-Net 또한 경쟁력 있는 성능을 확인하였다. 이 연구는 향후 다양한 깊이의 U-Net 변형 구조를 조합하는 자동화된 앙상블 프레임워크 연구에 기여할 가능성이 크다.