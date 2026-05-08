# Refine and Represent: Region-to-Object Representation Learning

Akash Gokul, Konstantinos Kallidromitis, Shufan Li, Yusuke Kato, Kazuki Kozuka, Trevor Darrell, Colorado J Reed (2022)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전의 자기지도학습(Self-Supervised Learning, SSL) 기반 사전 학습(Pretraining) 모델이 겪고 있는 한계점을 해결하고자 한다. 기존의 객체 중심 사전 학습(Object-centric pretraining) 방법들은 이미지 내에서 객체를 식별하기 위해 정적인 오프더쉘 세그멘테이션 휴리스틱(Off-the-shelf segmentation heuristics)에 의존한다. 하지만 이러한 방식은 사전 정의된 마스크의 정확도에 모델의 성능이 종속되는 문제가 있으며, 고정된 마스크를 사용하기 때문에 학습 과정에서 객체 영역을 유연하게 발견하거나 정교화할 수 없다는 한계가 있다.

특히, 픽셀 수준의 예측이 필요한 객체 탐지(Object Detection), 인스턴스 분할(Instance Segmentation), 시맨틱 분할(Semantic Segmentation)과 같은 밀집 예측(Dense prediction) 작업에서는 단순한 이미지 수준의 표현 학습보다 객체 중심의 특징 학습이 필수적이다. 따라서 본 연구의 목표는 외부의 고정된 휴리스틱에 의존하지 않고, 학습 과정에서 스스로 객체 중심의 영역을 발견(Discovery)하고 이를 통해 표현 학습(Representation Learning)을 수행하는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **마스크 예측(Mask Prediction)**과 **표현 학습(Representation Learning)** 사이를 반복적으로 오가며(Oscillating), 학습이 진행됨에 따라 점진적으로 정교한 객체 영역을 찾아내는 것이다.

가장 중심적인 설계는 **Region-to-Object Curriculum**의 도입이다. 이는 학습 초기에는 많은 수의 작은 영역(Region)을 학습하고, 학습이 진행될수록 점차 적은 수의 큰 객체 중심 영역(Object)을 학습하도록 유도하는 전략이다. 이를 통해 모델은 국소적인 특징(Local features)부터 시작하여 점진적으로 시맨틱하게 의미 있는 객체 단위의 특징을 학습할 수 있게 된다.

## 📎 Related Works

기존의 자기지도학습은 크게 세 가지 방향으로 발전해 왔다.

1. **이미지 수준 사전 학습(Image-Level SSL):** SimCLR나 BYOL과 같이 이미지 전체의 불변성(Invariance)을 학습하는 방식이다. 이는 분류 작업에는 효과적이지만, 픽셀 수준의 로컬리티(Locality)가 중요한 밀집 예측 작업에는 최적이지 않다.
2. **영역 기반 사전 학습(Region-Based SSL):** DenseCL나 ReSim과 같이 이미지 패치 간의 일관성을 학습하는 방식이다. 모든 영역에서 학습이 가능하다는 장점이 있으나, 객체의 시맨틱한 경계를 고려하지 않는다.
3. **객체 중심 사전 학습(Object-Centric SSL):** DetCon과 같이 세그멘테이션 휴리스틱을 사용해 객체 영역을 추출하고 해당 영역의 특징을 학습하는 방식이다. 성능은 뛰어나지만, 앞서 언급한 것처럼 고정된 마스크의 정확도라는 제약 조건에 묶여 있다.

R2O는 영역 기반 학습의 '유연한 로컬 특징 학습'과 객체 중심 학습의 '시맨틱한 객체 표현'이라는 두 가지 장점을 통합하여, 고정된 마스크 없이도 객체 중심의 표현을 학습할 수 있도록 설계되었다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

R2O는 크게 두 가지 단계가 상호작용하는 이단계 최적화(Bilevel optimization) 문제로 정의된다.

- **상위 수준(Upper-level):** 현재 학습된 인코더의 특징을 사용하여 영역 기반 우선순위(Region-level prior)를 객체 중심 마스크 $M$으로 변환하는 과정이다.
- **하위 수준(Lower-level):** 생성된 마스크 $M$을 기반으로 마스크 내부의 특징들이 서로 유사해지도록 인코더 $\theta$를 최적화하는 과정이다.

### 상세 구성 요소 및 절차

**1. 마스크 예측 (Mask Prediction)**
모델은 SLIC(Simple Linear Iterative Clustering)과 같은 간단한 영역 기반 우선순위를 사용하여 이미지를 작은 슈퍼픽셀(Super-pixels) 단위로 나눈다. 각 영역의 특징을 추출하여 $K$-means 클러스터링을 수행함으로써 객체 중심의 세그멘테이션 마스크 $M$을 생성한다. 이때 마스크 예측의 목적 함수는 다음과 같은 클러스터링 비용 함수로 정의된다.
$$L_{\text{mask}}(M; f_\theta(X), R) = \frac{1}{K} \sum_{k=1}^K \frac{1}{|M_k|} \sum_{p \in M_k} ||p - \mu_k||^2$$
여기서 $p$는 영역별 임베딩이며, $\mu_k$는 해당 클러스터의 중심점이다.

**2. 객체 중심 표현 학습 (Representation Learning)**
BYOL(Bootstrap Your Own Latent) 아키텍처를 기반으로 하며, Online Network와 Target Network(EMA 업데이트)의 Siamese 구조를 가진다. 기존 BYOL이 이미지 전체를 평균 내는 Global-pooling을 사용했다면, R2O는 위에서 예측된 마스크를 사용하는 **Mask-pooling**을 적용한다.

두 가지 뷰($x_1, x_2$)에 대해 각각 마스크 $m_1, m_2$를 적용하여 특징 $z_{\theta,1}, z_{\xi,2}$를 추출하고, L2 손실 함수를 통해 두 표현의 유사도를 극대화한다.
$$L_{\text{BYOL}}(z_\theta, z_\xi) = 2 - 2 \cdot \frac{q_\theta(z_\theta) \cdot z_\xi}{||q_\theta(z_\theta)||_2 \cdot ||z_\xi||_2}$$
최종 손실 함수 $L_{\text{repr}}$은 두 뷰를 서로 교차하여 적용한 두 손실의 합으로 계산된다.

**3. Region-to-Object Curriculum**
학습 과정에서 $K$-means의 클러스터 개수 $K$를 조절하는 스케줄러를 적용한다.

- **학습 초기:** $K=128$과 같이 높은 값을 설정하여 모델이 매우 작은 영역들의 특징을 학습하게 한다(Region-level pretraining).
- **학습 후기:** $K=4$와 같이 낮은 값으로 점진적으로 줄여, 모델이 강제로 더 큰 의미 단위의 객체 영역을 묶어 학습하게 한다(Object-centric pretraining).

## 📊 Results

### 실험 설정

- **데이터셋:** ImageNet-1K(사전 학습), MS COCO(객체 탐지 및 인스턴스 분할), PASCAL VOC 및 Cityscapes(시맨틱 분할), CUB-200-2011(비지도 객체 분할).
- **모델 아키텍처:** ResNet-50 backbone 사용.
- **평가 지표:** $AP_{bb}$ (Bounding box), $AP_{mk}$ (Mask), mIOU (Mean Intersection over Union).

### 주요 결과

1. **전이 학습 성능 (Transfer Learning):** ImageNet으로 사전 학습 후 COCO 데이터셋에서 $2\times$ 스케줄 기준 $AP_{bb}$에서 +0.9, $AP_{mk}$에서 +0.4의 향상을 보이며 SOTA 성능을 달성하였다. 또한 PASCAL VOC(+1.3 mIOU)와 Cityscapes(+0.3 mIOU) 시맨틱 분할에서도 기존 방법론들을 능가하였다.
2. **장면 중심 데이터 학습 (COCO Transfer):** 객체 중심의 ImageNet이 아닌, 여러 객체가 섞인 COCO 데이터셋으로 사전 학습했을 때도 PASCAL VOC(+1.9 mIOU)와 Cityscapes(+2.4 mIOU)에서 뛰어난 성능을 보여, 객체 발견 능력의 일반화 가능성을 입증하였다.
3. **비지도 객체 분할 (Unsupervised Segmentation):** 추가 학습 없이 ImageNet 사전 학습 모델에 $K=5$ 클러스터링만 적용했을 때, CUB-200-2011 데이터셋에서 71.6 mIOU를 기록하며 기존 비지도 분할 전용 모델들을 앞질렀다.

## 🧠 Insights & Discussion

### 강점 및 해석

R2O의 성공 요인은 **'세그멘테이션 병목(Segmentation Bottleneck)'**의 도입에 있다. $K$값을 점진적으로 줄임으로써, 모델은 초기에는 세밀한 텍스처와 색상 정보를 학습하고, 후기에는 이를 통합하여 시맨틱한 객체 단위를 구성해야만 하는 제약 조건에 놓이게 된다. 결과적으로 외부 휴리스틱 없이도 인코더 자체가 객체의 경계를 이해하는 표현을 학습하게 된 것이다.

### 한계 및 비판적 논의

논문에서도 명시되었듯, R2O가 생성하는 마스크가 항상 완벽한 '객체'를 의미하는 것은 아니다. 비지도 학습의 특성상 색상이나 텍스처가 유사한 배경과 객체를 하나의 그룹으로 묶거나, 색상 대비가 심한 객체의 일부를 분리하는 등의 실패 사례가 존재한다. 이는 모델이 시맨틱한 의미(Semantic grounding)보다는 시각적 유사성에 기반하여 영역을 묶기 때문에 발생하는 한계이다. 하지만 이러한 불완전한 마스크를 사용함에도 불구하고 전이 학습 성능이 높게 나타난다는 점은, 정확한 마스크 그 자체보다 '마스크를 예측하고 정교화하는 과정'이 유용한 특징 학습에 더 큰 기여를 함을 시사한다.

## 📌 TL;DR

본 논문은 고정된 세그멘테이션 휴리스틱 없이 스스로 객체 영역을 발견하고 학습하는 **R2O(Region-to-Object)** 프레임워크를 제안한다. 작은 영역에서 큰 객체 영역으로 학습 범위를 넓히는 **Region-to-Object Curriculum**을 통해, 국소 특징과 객체 중심 특징을 모두 효과적으로 학습하였다. 그 결과 객체 탐지, 인스턴스 분할 및 시맨틱 분할 등 다양한 밀집 예측 작업에서 SOTA 성능을 달성하였으며, 특히 비지도 객체 분할에서도 탁월한 능력을 보여 향후 범용적인 시각 표현 학습 연구에 중요한 가능성을 제시하였다.
