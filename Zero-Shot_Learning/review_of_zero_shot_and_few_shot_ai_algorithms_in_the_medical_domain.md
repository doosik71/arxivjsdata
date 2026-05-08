# Review of Zero-Shot and Few-Shot AI Algorithms in The Medical Domain

Maged Badawi, Mohammedyahia Abushanab, Sheethal Bhat, and Andreas Maier (2024)

## 🧩 Problem to Solve

본 논문은 전통적인 머신러닝, 딥러닝 및 컴퓨터 비전 방법론이 가진 근본적인 한계인 **대규모의 정밀하게 라벨링된 데이터에 대한 의존성**과 **낮은 일반화 성능** 문제를 해결하고자 한다. 특히 의료 영상 분야에서는 타겟 객체의 크기가 매우 작고, 이미지 선명도가 낮으며, 노이즈가 심하다는 특성이 있어 대량의 데이터를 확보하고 정밀하게 라벨링하는 것이 매우 어렵다.

따라서 본 연구의 목표는 적은 양의 데이터만으로도 높은 성능을 낼 수 있는 Few-Shot Learning (FSL)과, 학습 단계에서 전혀 보지 못한 클래스를 인식할 수 있는 Zero-Shot Learning (ZSL) 기술들을 조사하고, 이를 의료 및 일반 도메인의 객체 탐지(Object Detection)에 어떻게 적용할 수 있는지 분석하는 것이다.

## ✨ Key Contributions

본 논문은 지난 3년간 발표된 Zero-shot, Few-shot 및 일반 객체 탐지 관련 연구들을 체계적으로 리뷰하고 분류한 서베이 논문이다. 핵심적인 기여는 다음과 같다.

- **기술적 분류 체계 제시**: 리뷰 대상 논문들을 사용 데이터(Medical vs Natural), 학습 방법(Supervised vs Unsupervised), 모델 타입(Discriminative vs Generative)에 따라 분류하여 표 형태로 제공함으로써 연구 흐름을 한눈에 파악할 수 있게 하였다.
- **최신 패러다임의 분석**: 최근의 객체 탐지 트렌드가 단순한 시각 정보 처리를 넘어, Vision-Language Model (VLM)을 활용하여 텍스트 임베딩과 시각적 특징을 정렬(Alignment)하는 방향으로 진화하고 있음을 분석하였다.
- **의료 도메인 특화 문제 해결책 제시**: 의료 영상의 고유한 문제(작은 객체, 노이즈 등)를 해결하기 위한 Mask mechanism, Feature fusion and extraction module 등의 기법들이 어떻게 적용되었는지 상세히 설명하였다.

## 📎 Related Works

본 논문은 다양한 관련 연구를 다루고 있으며, 특히 기존의 Supervised Learning 기반 객체 탐지 모델들이 가진 한계를 지적한다.

- **전통적 객체 탐지**: 대량의 데이터가 필요하며, 학습 데이터에 포함되지 않은 새로운 카테고리의 객체를 인식하는 능력이 매우 부족하다.
- **ZSL 및 FSL의 등장**: 이러한 한계를 극복하기 위해 시맨틱 정렬(Semantic Alignment)이나 사전 학습된 모델의 미세 조정(Fine-tuning)을 통해 데이터 효율성을 높이려는 시도들이 이어지고 있다.
- **차별점**: 본 리뷰는 단순한 알고리즘 나열이 아니라, 의료 영상이라는 특수 도메인과 일반 자연 이미지 도메인을 모두 아우르며, 특히 VLM과 같은 최신 아키텍처가 어떻게 ZSL/FSL의 성능을 끌어올렸는지에 초점을 맞춘다.

## 🛠️ Methodology

본 논문은 서베이 논문이므로 단일 방법론을 제시하지 않으며, 대신 리뷰한 주요 모델들의 핵심 메커니즘을 설명한다.

### 1. Zero-Shot Learning (ZSL) 메커니즘

ZSL의 핵심은 학습 시 보지 못한(Unseen) 클래스를 인식하기 위해 시각적 특징을 시맨틱 공간(Semantic Space)으로 매핑하는 것이다.

- **ZSD-YOLO**: YOLOv5를 수정하여 일반적인 클래스 출력 대신 CLIP 모델의 임베딩 크기와 동일한 시맨틱 출력을 생성한다. 텍스트 임베딩과 시각적 임베딩을 정렬하여 Unseen 클래스를 탐지한다.
- **GTNet (Generative Transfer Network)**: 생성적 적대 신경망(GAN)을 사용하여 Unseen 클래스의 시각적 특징을 합성함으로써 Hubness 문제(고차원 특징이 저차원 시맨틱 공간으로 매핑될 때 발생하는 집중 현상)를 해결한다.
- **BLC (Background Learnable Cascade)**: 배경과 Unseen 클래스 간의 혼동을 줄이기 위해 Background Learnable Region Proposal Network (BLRPN)를 도입하여 배경에 대한 최적의 워드 벡터를 학습한다.

### 2. Few-Shot Learning (FSL) 메커니즘

FSL은 극소수의 샘플(One-shot 등)만으로 모델을 적응시키는 것이 핵심이다.

- **LoCoOp**: CLIP 모델을 기반으로 Local Regularized Context Optimization을 적용하여, ID(In-Distribution)와 무관한 노이즈를 제거하고 OOD(Out-of-Distribution) 탐지 성능을 높인다.
- **Category Name Initialization (CNI)**: 분류 헤드를 무작위로 초기화하는 대신, 사전 학습된 언어 모델을 통해 카테고리 이름의 텍스트 임베딩으로 초기화하여 소수 샘플만으로도 빠르게 수렴하게 한다.

### 3. Regular Object Detection 및 의료 영상 특화 기법

의료 영상의 노이즈 제거와 정밀한 경계 탐지를 위한 방법론들이 소개된다.

- **MS Transformer**: 계층적 트랜스포머와 Mask mechanism을 결합하여 이미지의 노이즈를 줄이고, 윈도우 어텐션(Window attention)을 통해 작은 병변(Lesion) 영역에 더 높은 가중치를 부여한다.
- **TRMFCN**: U-Net 구조를 개선하여 Triple Residual Multiscale 구조를 도입함으로써 다양한 스케일의 특징을 추출하고 기울기 소실 문제를 해결한다.

### 4. 주요 수식 설명

논문에서 언급된 의료 영상 분석의 손실 함수는 다음과 같다.

- **MSE Loss (이미지 재구성 단계)**: 원본 픽셀과 재구성된 픽셀 간의 차이를 측정한다.
$$L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$
여기서 $N$은 총 픽셀 수, $y_i$는 실제 값, $\hat{y}_i$는 예측 값이다.

- **IoU Loss (Bounding Box 예측 단계)**: 실제 박스와 예측 박스의 겹침 정도를 측정한다.
$$\text{IoU Loss} = -\ln \frac{\text{Intersection}(\text{box}_{gt}, \text{box}_{pre})}{\text{Union}(\text{box}_{gt}, \text{box}_{pre})}$$
$\text{box}_{gt}$는 Ground-Truth 박스, $\text{box}_{pre}$는 예측 박스를 의미하며, 값이 작을수록 두 박스가 일치함을 의미한다.

## 📊 Results

본 논문은 다양한 벤치마크 데이터셋과 지표를 통해 리뷰 대상 모델들의 성능을 간접적으로 제시한다.

- **사용 데이터셋**:
  - 자연 이미지: MS-COCO, ImageNet, PASCAL VOC, Visual Genome.
  - 의료 이미지: MoNuSeg (핵 검출), DeepLesion (병변 검출), 기타 위장 폴립(Gastric Polyps) 데이터셋 및 영유아 뇌 MRI 데이터셋.
- **평가 지표**: mean Average Precision (mAP), Recall@100 (RE@100), AUROC, Precision 등이 사용되었다.
- **주요 결과**:
  - **ZSL**: VLM(CLIP, GLIP 등)을 활용한 모델들이 기존의 단순 시각 모델보다 Unseen 클래스 탐지 능력이 월등히 높음을 확인하였다.
  - **FSL**: 카테고리 이름 정보를 활용한 초기화 방식이 단순 Fine-tuning보다 적은 데이터로 더 높은 일반화 성능을 보였다.
  - **의료 영상**: Mask mechanism과 Feature Fusion 모듈을 적용한 YOLOv3 기반 모델들이 작은 폴립이나 병변 탐지에서 기존 모델보다 높은 정확도를 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 가능성

본 논문은 최근 컴퓨터 비전의 핵심 트렌드인 '시각-언어 정렬'이 어떻게 데이터 부족 문제를 해결하는지 명확히 보여준다. 특히 의료 분야에서 라벨링 비용이 매우 높다는 점을 고려할 때, VLM을 활용한 Zero-shot 기반의 핵(Nuclei) 탐지나 병변 분석은 실제 임상 적용 가능성이 매우 높은 방향이다.

### 한계 및 비판적 해석

- **개발 과정의 상세 설명 부족**: 저자들은 리뷰한 논문들에서 모델의 성능 수치는 강조되었으나, 실제 개발 과정에서 겪은 시행착오나 하이퍼파라미터 튜닝의 어려움에 대한 논의가 부족함을 지적하였다.
- **도메인 적응 문제**: 자연 이미지에서 학습된 VLM을 의료 영상에 그대로 적용할 때 발생하는 도메인 갭(Domain Gap)을 어떻게 완전히 극복할 것인가에 대한 심층적인 분석보다는, 개별 논문의 결과 제시 수준에 머물러 있다.
- **실제 환경 검증 부족**: 대부분의 결과가 벤치마크 데이터셋에 의존하고 있으며, 실제 의료 현장의 다양한 변수(장비 차이, 환자 상태 등)가 반영된 결과인지에 대해서는 명시되지 않았다.

## 📌 TL;DR

본 논문은 데이터 희소성 문제를 해결하기 위한 **Zero-shot 및 Few-shot 학습 알고리즘을 의료 및 일반 객체 탐지 관점에서 종합적으로 분석한 서베이 보고서**이다. 특히 Vision-Language Model (VLM)을 이용한 시맨틱 정렬과 생성 모델(GAN)을 통한 특징 합성이 Unseen 클래스 탐지의 핵심임을 강조하며, 의료 영상의 특수성(작은 객체, 노이즈)을 해결하기 위한 최신 아키텍처들을 분류하고 정리하였다. 이 연구는 향후 데이터 효율적인 의료 AI 모델 설계와 VLM의 도메인 특화 적응 연구에 중요한 가이드라인을 제공할 것으로 기대된다.
