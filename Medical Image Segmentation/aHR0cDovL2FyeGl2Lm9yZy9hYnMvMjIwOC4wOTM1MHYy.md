# PyMIC: A deep learning toolkit for annotation-efficient medical image segmentation

Guotai Wang et al. (2023)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝 모델의 성능을 높이기 위해서는 픽셀 단위의 정밀하고 방대한 어노테이션(Annotation) 데이터가 필수적이다. 그러나 의료 영상의 특성상 다음과 같은 문제로 인해 고품질의 라벨을 확보하는 것이 매우 어렵다.

- **전문가 의존성**: 자연어 이미지와 달리 의료 영상의 라벨링은 고도의 전문 지식을 갖춘 의료 전문가가 수행해야 하므로 인력 확보에 한계가 있다.
- **데이터의 방대함**: CT나 MRI 같은 영상은 수백 장의 2D 슬라이스로 구성된 볼륨 데이터이며, 여러 장기와 병변을 일일이 획정하는 작업은 막대한 시간과 비용이 소모된다.
- **라벨의 불완전성**: 영상의 낮은 대비(Contrast)와 작업자의 경험 차이로 인해 라벨에 노이즈가 포함되거나 부정확한 경우가 빈번하다.

기존의 오픈소스 툴킷(MONAI, nnU-Net 등)은 주로 완전 지도 학습(Fully Supervised Learning)에 초점을 맞추고 있어, 불완전한 라벨(Imperfect labels)을 활용한 학습을 구현하려면 데이터 로더, 손실 함수, 네트워크 구조 등을 처음부터 다시 설계해야 하는 구현상의 부담이 크다. 본 논문의 목표는 이러한 어노테이션 비용을 줄이기 위한 **어노테이션 효율적 학습(Annotation-efficient learning)**을 지원하는 모듈형 딥러닝 툴킷인 PyMIC를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 의료 영상 분할을 위한 **PyTorch 기반의 모듈형 라이브러리인 PyMIC**를 제안한 것이다. PyMIC의 중심 아이디어는 완전 지도 학습뿐만 아니라 준지도 학습(Semi-Supervised Learning, SSL), 약지도 학습(Weakly Supervised Learning, WSL), 그리고 노이즈 라벨 학습(Noisy Label Learning, NLL)을 위한 고급 컴포넌트들을 표준화된 모듈 형태로 제공하여 연구자의 구현 부담을 최소화하는 것이다.

특히, 복잡한 코드 수정 없이 설정 파일(Configuration file) 편집만으로 다양한 데이터셋, 네트워크, 손실 함수 및 학습 파라미터를 변경할 수 있는 유연한 구조를 설계하였다.

## 📎 Related Works

의료 영상 분석을 위한 다양한 툴킷들이 존재한다.
- **범용 프레임워크**: TensorFlow, PyTorch, Keras 등이 있으나 의료 영상 특화 기능이 부족하여 구현 노력이 많이 든다.
- **전통적/특수 목적 툴킷**: NiftyReg, Elastix(등록), NiftySeg(분할), FreeSurfer, FSL(신경 영상 분석) 등이 있으나 딥러닝을 지원하지 않는다.
- **딥러닝 기반 의료 툴킷**: NiftyNet, DLTK, MONAI, TorchIO, Pymia, nnU-Net 등이 있다. 특히 nnU-Net은 자동 설정 기능을 통해 최신 성능(SOTA)을 달성한 것으로 유명하다.

**기존 접근 방식과의 차별점**: 기존 툴킷들은 대부분 "모든 훈련 데이터에 대해 정밀한 픽셀 단위 라벨이 존재한다"는 가정하에 설계되었다. 반면 PyMIC는 라벨이 일부만 있거나(SSL), 매우 성기게 존재하거나(WSL), 혹은 잘못된 라벨이 섞여 있는(NLL) 실제 의료 데이터의 불완전성을 직접적으로 처리할 수 있는 전용 모듈(예: Batch aggregation, Regularization loss, Multi-branch networks)을 내장하고 있다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
PyMIC는 PyTorch를 기반으로 하며, 전체 파이프라인을 캡슐화하는 `SegmentationAgent` 클래스를 중심으로 구성된다. `SegmentationAgent`는 다음과 같은 핵심 컴포넌트를 연결한다.
- **Dataset & BatchGenerator**: 데이터 로딩 및 배치 생성.
- **Network**: 딥러닝 모델 아키텍처.
- **Loss & Optimizer**: 학습 목표 및 최적화 알고리즘.
- **Inferer**: 추론 절차(Sliding window, TTA 등).
- **Evaluator**: 정량적 평가 지표 계산.

### 주요 구성 요소 및 상세 설명

#### 1. 데이터 핸들링 및 전처리
의료 영상의 특수한 포맷(Nifti, DICOM 등)을 지원하기 위해 `NiftyDataset`과 대용량 데이터의 효율적 접근을 위한 `H5Dataset`을 제공한다. 특히 SSL을 위해 라벨이 있는 데이터와 없는 데이터를 하나의 배치로 구성하는 **Batch aggregation** 기능을 지원하여, 학습 시 $N_L$개의 라벨 데이터와 $N_U$개의 무라벨 데이터를 동시에 처리할 수 있게 한다.

#### 2. 네트워크 및 추론 아키텍처
- **네트워크**: 2D(UNet, AttentionUNet, NestedUNet), 3D(UNet3D, UNet3D_ScSE), 그리고 2D와 3D 컨볼루션을 결합한 2.5D 네트워크(`UNet2D5`)를 지원한다.
- **고급 구조**: SSL/WSL/NLL에서 자주 사용되는 Multi-branch network, Uncertainty estimation(Monte Carlo Dropout 등), Self-ensembling(Mean Teacher) 구조를 내장하고 있다.
- **추론(`Inferer`)**: 대용량 볼륨 데이터를 위한 Sliding window inference, 결과의 강건성을 높이는 Test Time Augmentation(TTA), 그리고 여러 체크포인트를 결합하는 Model Ensemble을 지원한다.

#### 3. 손실 함수 및 학습 절차
PyMIC는 일반적인 `DiceLoss`, `CrossEntropyLoss` 외에 불완전한 라벨을 위한 특수 손실 함수를 제공한다.
- **RegularizationLoss**: 무라벨 픽셀에 대한 Entropy-minimization, Total Variation(TV) 등의 정규화 항을 제공한다.
- **NoiseRobustLoss**: 노이즈 라벨에 강건한 Generalized Cross Entropy(GCE), Mean Absolute Value(MAE) 손실 등을 구현하였다.
- **CombinedLoss**: 여러 손실 함수를 가중치 합으로 결합하여 사용할 수 있게 한다.
  $$ \text{Total Loss} = \sum_{i} \lambda_i \mathcal{L}_i $$
- **학습 절차**: `SegmentationAgent`를 상속받은 `SSLSegAgent`, `WSLSegAgent`, `NLLSegAgent`를 통해 각각 준지도, 약지도, 노이즈 라벨 학습을 위한 특수 파이프라인(예: Co-teaching, Knowledge distillation)을 제공한다.

## 📊 Results

### 실험 설정
- **하드웨어**: NVIDIA GTX 1080 Ti 2장.
- **데이터셋 및 작업**: 
  1. **완전 지도 학습**: MyoPS 2020(심근 병변 분할), CATARACTS 2020(백내장 수술 영상 분할).
  2. **준지도 학습 (SSL)**: ACDC(심장 MRI), Left Atrial(3D MRI) 데이터셋 (라벨 데이터 10%만 사용).
  3. **약지도 학습 (WSL)**: ACDC 데이터셋 (Scribble 어노테이션 사용).
  4. **노이즈 라벨 학습 (NLL)**: JSRT(흉부 X-ray) 데이터셋 (95%의 데이터를 인위적으로 왜곡).

### 주요 결과
- **완전 지도 학습**: MyoPS 2020에서 nnU-Net 대비 우수한 성능을 보였으며, 평균 Dice 계수 0.795(LV NM) $\sim$ 0.935(RV BP)를 달성하였다.
- **준지도 학습**: 10%의 라벨만 사용한 환경에서 CPS(Cross-Pseudo Supervision) 방법이 가장 높은 성능을 보였다. 특히 3D Left Atrial 데이터셋에서 Baseline 대비 Dice 수치가 유의미하게 상승하였다.
- **약지도 학습**: Scribble 어노테이션을 사용한 실험에서 DMPLS(Dynamically Mixed Pseudo Labels Supervision)가 가장 우수한 성능(Average Dice 89.28%)을 기록하였으며, 이는 원 논문의 결과보다 향상된 수치로 PyMIC의 효율성을 입증하였다.
- **노이즈 라벨 학습**: 노이즈가 심한 JSRT 데이터셋에서 DAST(Divergence-Aware Selective Training)가 가장 높은 Dice(96.94%)를 기록하며 노이즈에 매우 강건함을 보였다.

## 🧠 Insights & Discussion

### 강점
PyMIC는 단순한 라이브러리를 넘어, 의료 영상 분할의 실제적인 난제인 '라벨 부족'과 '라벨 오염' 문제를 해결하기 위한 최신 알고리즘들을 표준화된 모듈로 제공한다는 점에서 매우 실용적이다. 특히 `SegmentationAgent`라는 추상화 계층을 통해 네트워크 구조와 학습 알고리즘을 분리함으로써, 사용자는 백본 네트워크를 2D에서 3D로 변경하더라도 학습 로직을 수정할 필요가 없는 높은 유연성을 확보하였다.

### 한계 및 비판적 해석
- **파라미터 의존성**: nnU-Net이 데이터에 맞춰 자동으로 설정을 최적화하는 것과 달리, PyMIC는 사용자가 설정 파일을 통해 최적의 하이퍼파라미터를 직접 찾아야 한다. 이는 사용자에게 일정 수준의 전문 지식을 요구한다.
- **범위의 제한**: 현재 버전(v0.3)에서는 자기지도 학습(Self-supervised learning)이나 도메인 적응(Domain adaptation) 방법론이 포함되어 있지 않다.
- **효율성 문제**: 논문에서 언급되었듯, SSL이나 WSL 파이프라인은 일반적인 지도 학습보다 학습 효율(속도)이 떨어지는 경향이 있으며, 이에 대한 최적화 방안이 구체적으로 제시되지 않았다.

## 📌 TL;DR

PyMIC는 의료 영상 분할에서 발생하는 **어노테이션 비용 문제**를 해결하기 위해 설계된 PyTorch 기반의 모듈형 툴킷이다. 완전 지도 학습은 물론, **준지도(SSL), 약지도(WSL), 노이즈 라벨 학습(NLL)**을 위한 전용 컴포넌트와 에이전트를 제공하여 연구자가 복잡한 구현 없이 최신 알고리즘을 빠르게 적용하고 실험할 수 있게 한다. 향후 의료 영상 AI 연구에서 데이터 확보의 어려움을 겪는 연구자들에게 강력한 프레임워크가 될 가능성이 높다.