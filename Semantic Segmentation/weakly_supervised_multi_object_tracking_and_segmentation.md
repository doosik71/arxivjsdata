# Weakly Supervised Multi-Object Tracking and Segmentation

Idoia Ruiz, Lorenzo Porzi, Samuel Rota Bullo, Peter Kontschieder, Joan Serrat (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Multi-Object Tracking and Segmentation (MOTS) 태스크에서 발생하는 과도한 어노테이션 비용 문제이다. MOTS는 비디오 시퀀스 내 객체들에 대해 탐지, 분류, 추적뿐만 아니라 픽셀 단위의 마스크(Mask)를 예측해야 하는 고난도 작업이다.

기존의 MOTS 접근 방식은 모든 객체 인스턴스에 대해 정밀한 픽셀 수준의 세그멘테이션 마스크를 수동으로 작성해야 하며, 이는 데이터셋 구축 시 막대한 시간과 비용을 발생시킨다. 따라서 본 연구의 목표는 인스턴스 세그멘테이션을 위한 마스크 어노테이션 없이, 오직 Bounding Box, 클래스 정보, 그리고 객체 ID(Identity)만을 이용하여 MOTS를 수행하는 Weakly Supervised MOTS 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Multi-task Learning을 통해 서로 다른 태스크 간의 시너지 효과를 이용하는 것이다. 즉, 지도 학습(Supervised)이 가능한 분류(Classification)와 추적(Tracking) 태스크가 비지도 학습(Unsupervised) 상태인 인스턴스 세그멘테이션 태스크의 학습을 가이드하도록 설계하였다.

주요 기여 사항은 다음과 같다:

- **Weakly Supervised MOTS 문제 정의**: 마스크 어노테이션 없이 MOTS를 해결하려는 첫 번째 시도로서, 이 문제를 정의하고 해결 방법을 제시하였다.
- **시너지 학습 전략 설계**: Mask R-CNN 기반 아키텍처를 수정하여, 분류 브랜치와 추적 브랜치가 세그멘테이션 브랜치를 능동적으로 지원하는 구조를 제안하였다.
- **약한 감독 신호 생성**: Grad-CAM을 활용하여 전경(Foreground) 위치 정보를 추출하고, RGB 이미지 레벨의 정보를 통해 객체의 경계선을 정교화하는 손실 함수를 도입하였다.
- **성능 검증**: KITTI MOTS 벤치마크에서 완전 지도 학습(Fully Supervised) 방식과 비교했을 때, 세그멘테이션 품질 지표인 MOTSP 기준 성능 하락폭을 자동차 12%, 보행자 12.7% 수준으로 억제하며 가능성을 입증하였다.

## 📎 Related Works

### Multi-Object Tracking and Segmentation (MOTS)

기존의 MOTS 연구들은 주로 Fully Supervised 설정에서 진행되었다. 대표적으로 Mask R-CNN에 추적 브랜치를 추가하여 임베딩을 학습하고 객체를 매칭하는 방식이 사용되었다. 일부 연구에서는 Panoptic Segmentation과 추적을 결합하거나, 추적 정보가 3D 재구성(Reconstruction)을 돕는 상호 보완적 구조를 제안하였으나, 모두 정밀한 마스크 어노테이션이 필요하다는 한계가 있다.

### Weakly Supervised Segmentation

시맨틱 세그멘테이션 분야에서는 초기 약한 추정치(Weak estimate)를 예측한 후, Conditional Random Fields (CRF)와 같은 후처리 기법을 통해 경계선을 정교화하는 전략이 널리 사용된다. 일부 연구는 분류 네트워크의 활성화 맵(Activation map)을 초기 마스크로 활용하기도 한다. 본 논문은 이러한 아이디어를 MOTS의 인스턴스 레벨로 확장하여 적용하였다.

### Video Object Segmentation (VOS)

VOS는 모든 두드러진(Salient) 객체를 추적하고 세그멘테이션하는 작업으로 MOTS와 유사하다. 그러나 VOS는 특정 클래스에 국한되지 않으며, MOTS처럼 복잡한 폐색(Occlusion)이나 객체의 재등장과 같은 하드한 시나리오를 충분히 다루지 않는 경향이 있다.

## 🛠️ Methodology

### 전체 시스템 구조

본 연구는 MOTSNet 아키텍처를 기반으로 한다. ResNet-50 백본과 Feature Pyramid Network (FPN)를 통해 특징을 추출하고, Region Proposal Head (RPH)가 후보 영역을 제안한다. 이후 Region Segmentation Head (RSH)에서 분류, 탐지, 세그멘테이션을 수행하며, Tracking Head (TH)에서 임베딩 벡터를 생성한다.

특히, Grad-CAM을 통한 위치 정보 추출을 위해 분류 및 탐지 브랜치에 $1 \times 1$ Convolutional layer 두 개를 추가하여 ROI proposal별로 활성화 맵을 계산할 수 있도록 수정하였다.

### 학습 목표 및 손실 함수

전체 손실 함수 $L$은 다음과 같이 정의된다:
$$L = L_T + \lambda(L_{RPH} + L_{RSH})$$
여기서 $L_T$는 추적 손실, $L_{RPH}$는 영역 제안 손실, $L_{RSH}$는 영역 세그멘테이션 손실이다. 본 논문의 핵심은 마스크 정답이 없는 상태에서 $L_{RSH}$ 내부의 마스크 손실 $L_{msk}^{RSH}$를 다음과 같이 대체하는 것이다:
$$L_{msk}^{RSH} = L_{loc} + \lambda_{CRF}L_{CRF}$$

#### 1. Foreground Localization Loss ($L_{loc}$)

분류 브랜치에서 Grad-CAM을 사용하여 전경 위치 정보를 추출한다.

- **Pseudo-label 생성**: Ground Truth Bounding Box 내부의 픽셀 중 Grad-CAM 값이 임계값 $\mu_A$보다 높으면 전경(1), 박스 외부면 배경(0), 박스 내부지만 임계값보다 낮으면 무시($\emptyset$)로 설정하여 Pseudo-label $Y_{\check{r}}$을 생성한다.
- **손실 함수**: 예측된 마스크 $S_{\check{r}}$과 Pseudo-label $Y_{\check{r}}$ 사이의 Cross Entropy 손실을 계산한다.
$$L_{loc}(Y_{\check{r}}, S_{\check{r}}) = -\frac{1}{|P_{\check{r}}^Y|} \sum_{(i,j) \in P_{\check{r}}^Y} [Y_{\check{r}ij} \log S_{\check{r}ij} + (1 - Y_{\check{r}ij}) \log(1 - S_{\check{r}ij})]$$

#### 2. CRF Loss ($L_{CRF}$)

객체의 경계선을 정교하게 다듬기 위해 RGB 이미지의 픽셀 친화도(Affinity)를 이용하는 CRF 정규화 손실을 도입한다.
$$L_{CRF}(S_{\check{r}}) = \sum_k S'_{\check{rk}} W(1 - S_{\check{rk}})$$
여기서 $W$는 RGB 색상 및 위치 정보를 기반으로 한 친화도 행렬이다. 이를 통해 딥러닝 모델이 이미지의 실제 엣지 정보를 반영하여 마스크를 예측하도록 유도한다.

#### 3. Tracking Loss ($L_T$)

추적 헤드는 Mask-pooling 연산을 통해 예측된 마스크의 전경 부분만을 사용하여 임베딩 벡터를 생성한다.

- **Hard-triplet Loss**: 동일 객체의 임베딩은 가깝게, 다른 객체의 임베딩은 멀게 학습시킨다.
- **간접적 감독**: 추적 손실 $L_T$는 마스크-풀링 결과에 의존하므로, 추적 성능을 높이려는 학습 과정이 간접적으로 세그멘테이션 브랜치의 품질을 높이는 감독 신호로 작용한다.

### Grad-CAM의 변형 적용

기존 Grad-CAM은 ReLU를 적용하여 양의 영향력을 가진 특징만 고려하지만, 본 논문은 가중치의 절대값($|\alpha_{ck}|$)을 사용하여 음의 방향으로 큰 영향력을 가진 특징까지 모두 활용한다. 실험 결과, 이 방식이 더 완전하고 유용한 초기 마스크 힌트를 제공함을 확인하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: KITTI MOTS (자동차 및 보행자 클래스).
- **지표**: MOTSA (정확도), sMOTSA (엄격한 정확도), MOTSP (세그멘테이션 순수 품질).
- **비교 대상**: Fully Supervised 설정의 동일 모델 및 SOTA 모델(MOTSNet).

### 정량적 결과

결과는 표 2에 제시되어 있으며, 주요 내용은 다음과 같다:

- **Supervised vs Weakly Supervised**: 마스크 어노테이션 없이 학습했을 때, 세그멘테이션 품질 지표인 MOTSP의 성능 하락은 자동차에서 12.0%, 보행자에서 12.7%에 불과하였다.
- **클래스별 차이**: 보행자의 경우 자동차보다 성능 하락폭(MOTSA 기준)이 더 크게 나타났다. 이는 보행자가 크기가 작고 형태가 불규칙하며, Grad-CAM이 다리 주변 배경을 전경으로 오인하는 경우가 많기 때문이다.

### Ablation Study

각 손실 함수의 기여도를 분석한 결과:

- $L_{CRF}$를 제거했을 때 모든 지표에서 성능이 크게 하락하여, RGB 정보를 이용한 경계 정교화가 필수적임을 확인하였다.
- $L_T$ (추적 손실)는 특히 보행자 클래스의 세그멘테이션 성능 향상에 도움을 주었으나, 자동차 클래스에서는 영향이 적었다.

## 🧠 Insights & Discussion

본 논문은 데이터 어노테이션 비용이 매우 높은 MOTS 태스크에서, 지도 학습이 가능한 다른 태스크(분류, 추적)를 이용해 비지도 학습 태스크(세그멘테이션)를 성공적으로 가이드할 수 있음을 보여주었다. 특히 Mask R-CNN의 다중 작업 구조를 활용하여 각 브랜치가 서로를 보완하는 시너지 구조를 설계한 점이 돋보인다.

**강점 및 한계**:

- **강점**: 마스크 정답 없이도 상당 수준의 세그멘테이션 성능을 확보하였으며, Grad-CAM의 변형을 통해 더 나은 Pseudo-label을 생성하는 기법을 제시하였다.
- **한계**: 보행자와 같이 크기가 작고 복잡한 형태의 객체에 대해서는 여전히 정밀한 경계 추출에 어려움이 있다. 또한, Grad-CAM 기반의 약한 감독 신호가 항상 정확한 전경 위치를 보장하지 못한다는 가정이 존재한다.

**비판적 해석**:
본 연구는 MOTSP 지표에서는 낮은 성능 하락을 보였으나, MOTSA나 sMOTSA와 같이 탐지/추적 성능이 결합된 지표에서는 하락폭이 더 크게 나타난다. 이는 세그멘테이션 품질 자체가 낮아져서 탐지 및 매칭 단계의 신뢰도가 떨어졌음을 의미한다. 따라서 향후 연구에서는 더 정밀한 Pseudo-label 생성 기법이나 self-training 방식의 도입이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 마스크 어노테이션이 전혀 없는 **Weakly Supervised MOTS** 문제를 정의하고, 분류 및 추적 태스크가 세그멘테이션을 가이드하는 **시너지 학습 전략**을 제안하였다. Grad-CAM을 통한 전경 위치 추정($L_{loc}$)과 RGB 친화도 기반의 경계 정교화($L_{CRF}$), 그리고 추적 임베딩 학습($L_T$)을 결합하여, Fully Supervised 방식 대비 세그멘테이션 품질(MOTSP) 손실을 13% 이내로 줄이는 성과를 거두었다. 이는 향후 대규모 MOTS 데이터셋 구축 비용을 획기적으로 줄일 수 있는 가능성을 제시한다.
