# Neural Style Transfer Improves 3D Cardiovascular MR Image Segmentation on Inconsistent Data

Chunwei Ma, Zhanghexuan Ji, and Mingchen Gao (2019)

## 🧩 Problem to Solve

본 논문은 3D 심혈관 자기공명영상(Cardiovascular MR Image) 세그멘테이션에서 발생하는 **데이터 불일치(Data Inconsistency)** 문제를 해결하고자 한다. 딥러닝 기반의 세그멘테이션 모델은 뛰어난 성능을 보이지만, 실제 의료 환경에서는 촬영 장비의 파라미터, 실험 프로토콜, 환자의 특성 차이로 인해 발생하는 높은 변동성 때문에 모델의 일반화 성능이 저하되는 문제가 발생한다.

특히, 본 연구진은 학습 데이터(Training set)는 일반적으로 품질이 높은 반면, 테스트 데이터(Testing set)는 상대적으로 품질이 낮아 밝기, 해상도, 텍스처, 신호 대 잡음비(SNR) 등에서 상당한 차이가 나는 **데이터셋 시프트(Dataset Shift)** 현상을 확인하였다. 이러한 불일치는 특히 심근(Myocardium)의 신호가 배경 신호와 겹치는 현상을 야기하여, 정확한 영역 분할을 어렵게 만든다. 따라서 본 논문의 목표는 이러한 데이터 불일치 문제를 완화하여 3D 심장 전체 세그멘테이션의 정확도를 높이는 전략인 `StyleSegor`를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Neural Style Transfer(NST)**를 활용하여 테스트 데이터의 스타일을 학습 데이터의 스타일로 변환함으로써, 두 도메인 간의 이미지 특성 차이를 최소화하는 것이다.

주요 기여 사항은 다음과 같다:
1.  **스타일 전이 기반의 데이터 정규화**: 테스트 이미지의 스타일을 학습 이미지의 스타일로 변환하여 밝기, 대비, 텍스처 등의 불일치를 줄인다.
2.  **효율적인 아키텍처 채택**: Atrous Convolutional Network와 Atrous Spatial Pyramid Pooling(ASPP) 모듈을 사용하여 미세한 구조에 대한 세그멘테이션 성능을 높였다.
3.  **확률적 조정 및 앙상블 학습**: 네트워크 출력값에 대한 확률적 조정(Probabilistic Adjustment)을 적용하고, 원본 이미지와 스타일 전이 이미지의 예측 결과를 통합하는 앙상블 기법을 통해 강건성을 확보하였다.

## 📎 Related Works

기존의 3D 심혈관 MR 세그멘테이션 연구로는 3D U-Net, VoxResNet, 3D-DSN, DenseVosNet 등이 제안되었으며, 일부 앙상블 메타 러너를 통해 성능을 향상시킨 사례가 있다.

또한, 도메인 간의 차이를 줄이기 위해 다음과 같은 접근 방식들이 시도되었다:
- **Unsupervised Domain Adaptation**: 예측 마스크가 도메인 간에 유사하도록 강제하는 방식이 제안되었으나, 심근이나 혈액 풀(Blood pool)의 형태는 폐(Lungs)와 같은 단순한 구조보다 훨씬 복잡하여 적용에 한계가 있다.
- **Data Augmentation**: 학습된 변환(Learned transforms)을 통해 샘플을 생성하는 데이터 증강 기법이 제안되었으나, 본 논문의 예비 실험 결과 저품질 도메인에서 생성된 이미지를 추가하는 것만으로는 전체 성능 향상에 기여하는 바가 적었다.

`StyleSegor`는 단순히 데이터를 증강하는 것이 아니라, 테스트 시점에 입력 데이터 자체를 모델이 학습한 스타일로 변환하여 추론한다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인
`StyleSegor`의 전체 워크플로우는 크게 **스타일 전이 $\rightarrow$ 세그멘테이션 $\rightarrow$ 확률적 조정 $\rightarrow$ 앙상블** 순으로 구성된다. 세그멘테이션 네트워크로는 ResNet-101 기반의 DeepLabv3를, 스타일 전이 네트워크로는 VGG-16을 백본으로 사용한다.

### 2. 세그멘테이션 네트워크 (Modified DeepLabv3)
ResNet-101의 마지막 블록 위에 다양한 Atrous rate $r = (6, 12, 18)$를 가진 **Atrous Spatial Pyramid Pooling(ASPP)** 모듈을 구축하였다. ASPP는 $1 \times 1$ 컨볼루션 결과와 bilinear upsampling된 입력 피처 맵을 포함하여 총 5개의 레이어 피처를 결합(Concatenate)하며, 이후 3개의 $1 \times 1$ 컨볼루션 레이어를 통해 최종 로짓(Logits)을 생성한다.

### 3. Neural Style Transfer (NST)
테스트 이미지 $x$의 스타일을 타겟 학습 이미지 $y$의 스타일로 변환하여 생성 이미지 $\hat{y}$를 만든다. 이때 두 가지 손실 함수를 최적화한다.

- **Content Loss**: 생성 이미지 $\hat{y}$가 원본 이미지 $x$의 콘텐츠를 유지하도록 한다.
$$\ell_{\phi}^{\text{content}}(\hat{y}, x) = \sum_{j=1}^{J} \frac{1}{C_j H_j W_j} \|\phi_j(\hat{y}) - \phi_j(x)\|_2^2$$
- **Style Loss**: 생성 이미지 $\hat{y}$의 텍스처가 타겟 이미지 $y$와 유사하도록 Gram matrix $G$를 사용하여 계산한다.
$$G_{\phi}^j(y)_{i,k} = \frac{1}{C_j H_j W_j} \sum_{h=1}^{H_j} \sum_{w=1}^{W_j} \phi_j(y)_{h,w,i} \phi_j(y)_{h,w,k}$$
$$\ell_{\phi}^{\text{style}}(\hat{y}, y) = \sum_{j=1}^{J} \|G_{\phi}^j(\hat{y}) - G_{\phi}^j(y)\|_2^2$$
- **Total Loss**: 두 손실의 가중치 합으로 정의된다.
$$\ell_{\phi}^{\text{total}}(\hat{y}, x, y) = \alpha \ell_{\phi}^{\text{content}}(\hat{y}, x) + \beta \ell_{\phi}^{\text{style}}(\hat{y}, y)$$

**타겟 스타일 선택 과정**:
1. 1st Wasserstein metric을 사용하여 학습 및 테스트 샘플 간의 유사도를 측정한다.
2. 계층적 클러스터링(Hierarchical clustering)을 통해 스타일 라이브러리를 구축한다.
3. baseline 네트워크를 통해 각 슬라이스의 레이블 비율을 계산하고, 유클리드 거리가 가장 가까운 학습 슬라이스를 타겟 스타일로 선정한다.

### 4. 확률적 조정 및 앙상블 학습
심근과 혈액 풀의 신호가 배경 신호에 의해 묻히는 현상을 방지하기 위해, 특정 레이블 $k$의 점수를 다른 레이블들의 점수를 조건부로 하여 조정한다.
$$c(p_k) = \arg \max_{k \in \{1,2,3\}} p_k \prod_{j \neq k} \left(1 - \frac{e^{p_j}}{\sum_{q \in \{1,2,3\}} e^{p_q}}\right)$$
최종 결과는 원본 이미지와 스타일 전이 이미지 각각에 대해 $xy, yz, zx$ 세 평면에서 얻은 예측 결과들을 투표(Voting) 방식으로 통합하여 결정한다.

## 📊 Results

### 실험 설정
- **데이터셋**: MICCAI HVSMR 2016 챌린지 데이터셋 (학습 10건, 테스트 10건).
- **평가 지표**: Dice score, Average Distance Boundary (ADB), Hausdorff distance.
- **종합 점수(Overall score) 산출 식**: $\text{Dice} \times 0.5 - \text{ADB} \times 0.25 - \text{Hausdorff} \times 0.03$.

### 주요 결과
실험 결과, `StyleSegor (ensemble)` 모델이 기존의 SOTA 방법들과 baseline 모델들을 압도하는 성능을 보였다.

| Method | Myocardium Dice | Blood Pool Dice | Overall Score |
| :--- | :---: | :---: | :---: |
| Zheng et al. [13] | 0.833 | 0.939 | 0.234 |
| DeepLabv3 (baseline) | 0.648 | 0.920 | -0.214 |
| **StyleSegor (ensemble)** | **0.839** | **0.937** | **0.304** |

특히, 기존 최고 성능 대비 **종합 점수가 29.91% 향상**되었다. 심근의 Dice score는 0.839까지 올라갔으며, Hausdorff distance 역시 심근 2.832mm, 혈액 풀 4.023mm로 크게 낮아져 관심 영역(ROI)을 매우 정확하게 포착함을 입증하였다.

## 🧠 Insights & Discussion

본 논문의 결과는 의료 영상 분석에서 **학습 데이터와 테스트 데이터 사이의 스타일 불일치(Domain Shift)가 성능 저하의 핵심 원인**임을 시사한다.

**강점 및 해석**:
- Neural Style Transfer를 통해 테스트 이미지의 스타일을 학습 데이터의 분포로 강제 이동시킴으로써, 동일한 세그멘테이션 모델이 훨씬 더 쉽게 특징을 추출할 수 있게 되었다. 특히 저품질 이미지에서 억제되었던 심근 신호가 스타일 전이 후 효과적으로 복원(Elevated)됨을 확인하였다.
- 단순한 2D 컨볼루션 기반의 DeepLabv3 모델임에도 불구하고, ASPP의 넓은 수용역(Field of View)과 스타일 전이 전략 덕분에 3D 기반 모델들과 경쟁 가능한 성능을 냈다.

**한계 및 비판적 논의**:
- 스타일 전이 과정에서 일부 거짓 양성(False Positive) 예측이 발생하여 Hausdorff distance가 일시적으로 증가하는 경향이 관찰되었다. 하지만 이는 앙상블 학습을 통해 효과적으로 제거될 수 있음을 보여주었다.
- 모든 테스트 슬라이스에 대해 스타일 전이 최적화 과정(SGD)을 거쳐야 하므로, 실시간 추론 속도 면에서는 오버헤드가 발생할 수 있다. (논문에서는 한 쌍당 약 3초 소요됨을 명시함).

## 📌 TL;DR

본 논문은 3D 심혈관 MR 영상의 데이터 불일치 문제를 해결하기 위해 **Neural Style Transfer를 도입한 StyleSegor 파이프라인**을 제안하였다. 테스트 이미지를 학습 데이터의 스타일로 변환하고, 이를 ASPP 기반의 세그멘테이션 네트워크와 앙상블 기법으로 처리함으로써 **종합 점수를 29.91% 향상**시켰다. 이 연구는 서로 다른 병원이나 장비에서 수집된 데이터 간의 편차가 심한 의료 영상 분야에서 매우 실용적인 전처리 전략이 될 가능성이 높다.