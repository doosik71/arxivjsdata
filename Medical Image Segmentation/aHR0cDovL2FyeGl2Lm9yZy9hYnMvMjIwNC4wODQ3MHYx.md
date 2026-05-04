# U-Net and its variants for Medical Image Segmentation : A short review

Vinay Ummadi (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation, MIS) 분야에서 핵심적인 역할을 하는 U-Net 아키텍처와 그 변형 모델들을 분석하는 것을 목표로 한다. 의료 영상 분석은 방사선 전문의나 병리학자에게 매우 까다롭고 시간이 많이 소요되는 작업이며, 비침습적 진단을 수행하기 위해서는 관심 영역(Region of Interest, RoI)을 정확하게 분할하는 것이 필수적이다. 

따라서 본 연구의 목적은 전통적인 영상 분할 방식의 한계를 짚어보고, U-Net의 등장 이후 어떻게 딥러닝 기반의 아키텍처들이 발전해 왔는지, 그리고 각 변형 모델들이 어떤 직관을 통해 성능을 개선하려 했는지를 종합적으로 검토하는 것이다.

## ✨ Key Contributions

본 논문의 주요 기여는 의료 영상 분할의 진화 과정을 체계적으로 정리하고, 특히 U-Net을 기반으로 한 다양한 하이브리드 아키텍처들의 설계 아이디어를 분석한 점에 있다. 구체적으로는 다음과 같은 설계적 진화 방향을 제시한다.

- **구조적 보완**: 단순한 Encoder-Decoder 구조에서 Nested skip connections 및 Dense connections로의 확장(U-Net++).
- **학습 안정성 및 문맥 파악**: Residual connection과 Recurrent connection의 결합(R2U-Net).
- **선택적 집중**: Attention Gate를 통한 불필요한 정보 억제 및 RoI 집중(Attention U-Net).
- **전역적 문맥 학습**: Convolution의 국소적 한계를 극복하기 위한 Transformer의 도입(Trans U-Net).

## 📎 Related Works

논문에서는 딥러닝 이전의 전통적인 영상 분할 방식(Old school segmentation)을 다음과 같이 설명하며 그 한계를 지적한다.

- **Threshold-based segmentation**: 히스토그램 특성을 이용해 임계값을 설정하고 이진화하는 방식이다. 하지만 다중 클래스 분할이 불가능하고 강도 변화(intensity variations)에 취약하다.
- **Clustering-based segmentation**: K-means와 같이 유사한 픽셀을 그룹화하는 방식이다. 하지만 클러스터의 개수($k$)를 사전에 알고 있어야 한다는 제약이 있다.
- **Mean shift segmentation**: 윈도우의 평균을 이용해 언덕을 오르는(hill climbing) 방식으로 클러스터를 할당한다.
- **Graph Cut segmentation**: 최대 유량 최소 컷(maximum flow and minimum cut) 알고리즘 기반의 복잡한 방식이다.

이러한 전통적 방법들은 학습 데이터가 필요 없다는 장점이 있으나, 의료 영상의 특수성과 복잡성으로 인해 일반화 성능이 매우 떨어지며 실제 의료 현장에 적용하기에는 결과가 불충분하다는 한계가 있다.

## 🛠️ Methodology

본 논문은 U-Net과 그 변형 모델들의 구조적 특징을 다음과 같이 설명한다.

### 1. U-Net (2015)
U-Net은 대칭적인 Encoder-Decoder 구조를 가진다.
- **Encoder (Contracting Path)**: $3 \times 3$ 필터의 Convolution 레이어와 ReLU 활성화 함수가 반복되는 Conv block으로 구성되며, Max Pool을 통해 공간 해상도를 줄이고 특징 맵의 채널 수를 늘린다.
- **Decoder (Expanding Path)**: Transpose convolution(up-convolution)을 사용하여 해상도를 복원하며, 채널 수를 점진적으로 줄인다.
- **Skip Connection**: Encoder의 특징 맵을 Decoder의 대응되는 레이어에 직접 연결(Concatenation)하여 세밀한 공간 정보(semantic information)를 전달한다.
- **학습**: Cross-entropy loss를 사용하여 엔드-투-엔드로 학습한다.

### 2. U-Net++ (2019)
U-Net의 단순한 skip connection을 개선하여 **Nested and Dense skip connections**를 도입하였다.
- **설계 아이디어**: Encoder와 Decoder 사이의 세밀한 간극을 메우기 위해 중첩된 컨볼루션 레이어를 배치하여 그래디언트 흐름을 원활하게 하고 특징 재사용성을 높였다.
- **Deep Supervision**: 네트워크의 여러 단계에서 분할 맵을 추출하고 이를 업샘플링하여 학습에 활용함으로써, 거친(coarse) 특징부터 세밀한(fine) 특징까지 단계적으로 학습하게 한다.

### 3. R2U-Net (2018)
Residual connection과 Recurrent connection을 결합한 모델이다.
- **Recurrent Residual block**: U-Net의 표준 Conv block을 이 블록으로 대체한다. 
- **Residual connection**: 이전 레이어의 출력을 다음 레이어에 더해줌으로써 vanishing gradient 문제를 해결한다.
- **Recurrent connection**: 출력값을 다시 입력으로 피드백하는 구조를 통해 순차적 문맥(sequential context) 학습 능력을 부여한다.

### 4. Attention U-Net (2018)
Skip connection 경로에 **Attention Gate (AG)**를 추가한 하이브리드 구조이다.
- **작동 원리**: Soft attention 방식을 사용하여 RoI에 해당하는 중요한 특징은 강조하고, 불필요한 배경 정보는 억제한다. 이는 RoI의 모양과 크기가 매우 다양할 때 특히 효과적이다.

### 5. Trans U-Net (2021)
CNN의 국소적 수용장(local receptive field) 한계를 극복하기 위해 Vision Transformer(ViT)를 결합하였다.
- **구조**: CNN 기반의 특징 추출기 뒤에 Transformer 레이어를 배치하여 전역적 공간 의존성(global spatial dependencies)을 학습한다. 
- **결합**: Transformer가 인코딩한 전역 표현과 CNN의 저수준 세부 특징을 Decoder에서 결합하여 정밀한 위치 추정과 전역적 문맥 파악을 동시에 달성한다.

## 📊 Results

논문은 각 모델의 성능을 입증하기 위해 다음과 같은 정량적 지표와 결과치를 제시한다. 분할 성능 평가는 주로 Sørensen–Dice coefficient (DSC)와 Jaccard score (JS)를 사용하며, 공식은 다음과 같다.

$$DSC = \frac{2|X \cap Y|}{|X| + |Y|}, \quad JS = \frac{|X \cap Y|}{|X| + |Y|}$$

- **U-Net**: 35장의 부분 주석된 위상차 광학 현미경 이미지에서 $\text{IoU} = 0.92$를 달성하였다.
- **U-Net++**: U-Net 및 Wide U-Net 대비 평균 IoU 점수가 각각 $3.9$점, $3.4$점 향상되었다.
- **R2U-Net**: 피부암 병변 분할에서 Dice score $0.86$을 기록하여 standard U-Net($0.84$)보다 우수한 성능을 보였다.
- **Attention U-Net**: CT82 데이터셋의 췌장 분할 작업에서 Dice score $81.48 \pm 6.23$을 달성하였다.
- **Trans U-Net**: MICCAI 복부 라벨링 챌린지의 CT 스캔 데이터에서 평균 $\text{DSC} = 77.48$을 기록하여, standard U-Net($74.68$)과 Attention U-Net보다 뛰어난 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 U-Net 계열의 발전을 분석하며 다음과 같은 통찰을 제시한다.

**1. 아키텍처의 효율성과 복잡도**
U-Net은 적은 양의 데이터로도 높은 성능을 내는 매우 효율적인 모델이다. U-Net++, R2U-Net, Attention U-Net과 같은 변형 모델들이 성능을 일부 향상시키기는 했으나, 증가하는 모델 복잡도와 계산 비용에 비해 성능 향상 폭은 다소 제한적(marginal)이다. 반면 Trans U-Net은 Transformer를 통해 전역적 문맥을 학습함으로써 복잡도 대비 합리적인 성능 향상을 이루어냈다.

**2. 현재의 한계점**
현시점에서도 의료 영상 분할에는 다음과 같은 난제가 존재한다.
- 데이터 부족 및 라벨링 비용 문제.
- 의료 영상 모달리티의 다양성과 장비/병원별 데이터 편향(Data bias).
- 전문가(의사)와 연구자 간의 피드백 루프 부족 및 라벨링의 불일치(Noisy labels).

**3. 향후 연구 방향**
작가는 단순한 수동 아키텍처 설계보다는 다음과 같은 자동화 및 최적화 접근법이 필요하다고 주장한다.
- **AutoML 및 NAS**: 최적의 신경망 구조, 하이퍼파라미터, 손실 함수를 자동으로 탐색하는 기법.
- **약지도 학습(Weakly-supervised)** 및 **자기지도 학습(Self-supervised)**: 라벨이 부족하거나 불완전한 데이터를 처리하기 위한 전략.
- **모델 해석 가능성(Model Interpretation)**: 의료 분야의 특성상 모델의 판단 근거를 이해하는 것이 필수적이다.

## 📌 TL;DR

본 논문은 의료 영상 분할의 표준이 된 U-Net부터 최신 Trans U-Net까지의 진화 과정을 리뷰한다. 단순한 $\text{Conv-ReLU}$ 구조에서 시작해 $\text{Residual}$, $\text{Attention}$, $\text{Transformer}$와 같은 최신 기법들이 어떻게 통합되어 분할 정밀도를 높였는지 분석한다. 특히, 최근의 Transformer 도입이 전역적 문맥 파악이라는 돌파구를 마련했음을 강조하며, 향후에는 데이터 부족 문제를 해결하기 위한 자기지도 학습과 AutoML 기반의 아키텍처 탐색이 핵심 연구 방향이 될 것임을 시사한다.