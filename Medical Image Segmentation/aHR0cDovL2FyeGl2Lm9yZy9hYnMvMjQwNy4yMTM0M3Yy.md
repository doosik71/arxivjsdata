# MIST: A Simple and Scalable End-To-End 3D Medical Imaging Segmentation Framework

Adrian Celaya et al. (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Imaging Segmentation) 분야는 딥러닝의 발전으로 인해 비약적인 성장을 이루었으며, 다양한 새로운 아키텍처와 손실 함수가 제안되고 있다. 그러나 새로운 방법론을 훈련, 테스트 및 평가하기 위한 표준화된 도구의 부재로 인해, 서로 다른 연구 간의 성능을 공정하게 비교하는 것이 매우 어려운 상황이다.

특히, 일부 연구들이 기존의 강력한 기준선(baseline)인 nnU-Net보다 우수한 성능을 보였다고 주장하지만, 다른 연구에서는 이를 반박하는 등 결과의 일관성이 부족한 문제가 발생하고 있다. 이러한 불일치는 새로운 연구의 성능 주장을 객관적으로 검증하는 것을 방해하며, 의료 영상 분할 분야의 발전을 저해하는 요소가 된다. 따라서 본 논문의 목표는 일관된 훈련, 테스트 및 평가 파이프라인을 제공하여 재현 가능하고 공정한 비교가 가능한 표준화된 프레임워크인 MIST(Medical Imaging Segmentation Toolkit)를 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 기여는 의료 영상 분할의 전체 워크플로우를 포괄하는 단순하고 모듈화된 엔드-투-엔드(end-to-end) 프레임워크인 MIST를 제안한 것이다. MIST의 중심적인 설계 아이디어는 데이터 분석부터 전처리, 모델 훈련, 그리고 최종 평가에 이르기까지의 모든 단계를 표준화하여, 연구자가 아키텍처나 손실 함수와 같은 핵심 변수만을 변경하며 실험할 수 있도록 하는 것이다.

또한, MIST는 PyTorch의 DistributedDataParallel(DDP)과 NVIDIA의 Data Loading Library(DALI)를 통합하여 다중 GPU 환경에서의 확장성(scalability)을 극대화함으로써, 대규모 의료 데이터셋을 효율적으로 처리할 수 있는 컴퓨팅 능력을 제공한다.

## 📎 Related Works

의료 영상 분할 분야에서는 2015년 제안된 U-Net 이후 수많은 변형 구조가 등장하였으며, 특히 nnU-Net은 데이터셋의 특성에 따라 스스로 설정을 최적화하는 self-configuring 방식을 통해 BraTS 및 Medical Segmentation Decathlon(MSD)과 같은 벤치마크에서 state-of-the-art(SOTA) 성능을 달성하였다.

최근에는 Vision Transformer 기반의 아키텍처나 Boundary loss, Generalized Surface loss와 같은 새로운 손실 함수들이 제안되며 nnU-Net의 성능을 뛰어넘으려는 시도가 이어지고 있다. 그러나 본 논문은 이러한 개별적인 방법론의 발전에도 불구하고, 이를 검증할 표준화된 도구가 없기 때문에 발생하는 결과의 불일치를 지적하며, 개별 모델의 성능 개선보다는 실험 환경의 표준화라는 관점에서 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

MIST는 크게 데이터 분석(Data Analysis), 전처리(Preprocessing), 훈련(Training)의 세 가지 메인 파이프라인과 평가, 후처리, 추론이라는 보조 파이프라인으로 구성된다.

### 1. 데이터 분석 및 전처리 파이프라인
MIST는 BraTS 챌린지 데이터 포맷(환자별 서브 디렉토리, NIfTI 파일 형식)을 기본으로 사용하며, 다음과 같은 규칙 기반 분석 및 전처리를 수행한다.

- **Foreground Cropping**: 배경의 불필요한 영역을 제거하기 위해 33% 및 99.5% 백분위수 값으로 윈도잉(windowing)을 수행한 후, Otsu 필터를 통해 전경 임계값을 설정하여 크롭한다. 평균적으로 볼륨이 20% 이상 감소하는 경우에만 적용한다.
- **Target Spacing**: 각 축의 보셀 간격(voxel spacing)의 중앙값을 타겟으로 설정한다. 만약 최대/최소 간격 비율이 3배를 초과하는 이방성(anisotropic) 데이터인 경우, 저해상도 축의 10% 백분위수 간격을 사용한다.
- **Normalization**: MR 영상의 경우 0.5 및 99.5 백분위수 값을 사용하여 윈도잉하고 z-score 정규화를 적용한다. CT 영상의 경우, 정답 마스크(ground truth)에서 0이 아닌 영역에 대해 전역적으로 계산된 파라미터를 사용한다.
- **Distance Transform Maps (DTMs)**: 각 클래스의 경계로부터의 거리를 나타내는 DTM을 생성하여 학습에 활용할 수 있도록 한다. DTM의 값은 외부에서 양수, 경계에서 0, 내부에서 음수로 표현된다.

### 2. 훈련 파이프라인
MIST는 기본적으로 5-fold 교차 검증(cross-validation)을 수행하며, 다음과 같은 유연한 설정을 제공한다.

- **네트워크 아키텍처**: nnU-Net, U-Net, Swin UNETR, PocketNet, MedNeXt 등 6가지 아키텍처를 지원하며, Deep Supervision 및 VAE regularization과 같은 정규화 옵션을 추가할 수 있다.
- **손실 함수**: Dice with Cross Entropy(기본), clDice, Boundary loss, Hausdorff loss, Generalized Surface loss 등을 선택할 수 있다.
- **가속화 기술**: PyTorch의 DistributedDataParallel(DDP)을 통해 데이터 병렬화를 구현하고, NVIDIA DALI를 사용하여 GPU 상에서 데이터 로딩, 패치 추출 및 증강(augmentation)을 수행함으로써 연산 효율을 높였다.

### 3. 평가 및 후처리 파이프라인
훈련된 모델은 Dice Similarity Coefficient, 95th percentile Hausdorff Distance ($\text{HD}_{95}$), Surface Dice 등을 통해 평가된다. 후처리 단계에서는 작은 객체 제거, top-k 컴포넌트 유지, 형태학적 클리닝(morphological cleaning) 및 홀 채우기(hole filling) 등의 연산을 적용할 수 있다.

## 📊 Results

### 1. 분할 정확도 분석
본 연구에서는 BraTS Adult Glioma Post-Treatment Challenge 데이터셋을 사용하여 MIST의 효용성을 검증하였다. Pocket nnUNet 아키텍처와 Dice with Cross Entropy 손실 함수를 사용하였으며, 8개의 NVIDIA H100 GPU로 훈련하였다.

- **정량적 결과**: 5-fold 교차 검증 결과, 모든 분할 클래스에서 중앙값(median) Dice 점수가 $0.9$ 이상을 기록하였다. 특히 Whole Tumor(WT) 클래스의 경우 Mean Dice $0.9063$을 기록하며 높은 정확도를 보였다.
- **검증 데이터셋 결과**: 실제 챌린지 검증 데이터셋에서도 WT 클래스에 대해 Dice $0.9257$, $\text{HD}_{95}$ $6.6367\text{mm}$의 준수한 성능을 나타냈다.

### 2. 계산 성능 및 확장성
MIST의 다중 GPU 확장성을 측정하기 위해 A100과 H100 GPU에서 배치 사이즈별 에포크당 소요 시간을 측정하였다.

- **A100 GPU**: 배치 사이즈가 커질수록 확장성이 개선되는 경향을 보였다.
- **H100 GPU**: 모든 배치 사이즈에서 거의 최적의 확장성(near optimal scaling)을 보였으며, 1개 GPU에서 8개 GPU로 확장했을 때 약 6배의 속도 향상이 관찰되었다. 이는 H100이 A100보다 약 2배 빠른 연산 속도를 가진다는 NVIDIA의 보고와 일치한다.

## 🧠 Insights & Discussion

MIST는 단순한 라이브러리를 넘어 의료 영상 분할 연구의 '표준 실험실' 역할을 수행할 수 있는 가능성을 보여주었다. 특히 모듈화된 설계를 통해 새로운 아키텍처나 손실 함수를 쉽게 통합할 수 있으며, 이를 통해 연구자들은 전처리나 평가 방식의 차이에서 오는 노이즈 없이 오직 모델의 개선 사항만을 공정하게 비교할 수 있다.

강점으로는 NVIDIA DALI와 DDP의 통합을 통해 대규모 데이터셋에서도 학습 시간을 획기적으로 단축할 수 있다는 점이 꼽힌다. 이는 향후 TotalSegmentor와 같은 초대형 데이터셋을 활용한 의료 영상 파운데이션 모델(Foundation Model) 구축에 필수적인 요소가 될 것이다.

다만, 본 논문에서는 BraTS 데이터셋에 한정된 실험 결과만을 제시하였으며, 제안된 파라미터(Deep supervision, L2 regularization 등)에 대한 상세한 어블레이션 연구(ablation study)는 향후 과제로 남겨두었다. 또한, nnU-Net이나 MedNeXt와 같은 기존 프레임워크와의 직접적인 심층 비교 분석이 부족하다는 점이 한계로 지적될 수 있다.

## 📌 TL;DR

MIST는 의료 영상 분할의 전 과정(분석 $\rightarrow$ 전처리 $\rightarrow$ 학습 $\rightarrow$ 평가)을 표준화한 엔드-투-엔드 프레임워크이다. 이 연구는 모델 간의 불공정한 비교 문제를 해결하기 위해 표준화된 파이프라인을 제공하며, 특히 NVIDIA DALI와 DDP를 통한 강력한 다중 GPU 확장성을 확보하였다. BraTS 데이터셋에서 높은 정확도와 효율적인 연산 속도를 입증하였으며, 향후 의료 영상 분야의 파운데이션 모델 개발 및 공정한 벤치마킹을 위한 핵심 도구로 활용될 가능성이 높다.