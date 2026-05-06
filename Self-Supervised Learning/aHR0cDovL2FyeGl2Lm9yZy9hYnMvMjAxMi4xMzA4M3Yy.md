# Self-Supervised Representation Learning for Astronomical Images

Md Abul Hayat, George Stein, Peter Harrington, Zarija Lukić, and Mustafa Mustafa (2021)

## 🧩 Problem to Solve

천문학 분야의 스카이 서베이(Sky Surveys)는 방대한 양의 데이터를 생성하지만, 이를 분석하기 위한 고품질의 레이블링된 데이터(Labeled data)를 확보하는 것은 매우 어렵다. 특히 Galaxy Zoo와 같은 시민 과학(Citizen science) 프로젝트를 통해 레이블을 생성하는 방식은 인간의 휴리스틱한 이해력에 의존하므로, 향후 Vera Rubin Observatory나 Euclid와 같이 데이터 규모가 기하급수적으로 증가하는 차세대 관측 시스템에서는 더 이상 적용 가능하지 않다.

기존의 지도 학습(Supervised learning) 기반 딥러닝 모델들은 레이블의 양과 질에 크게 의존하며, 이는 학습 데이터의 선택 편향(Selection bias) 문제를 야기한다. 예를 들어, 분광학적 적색편이(Spectroscopic redshift) 레이블이 있는 은하들은 주로 밝고 근거리에 위치한 객체들에 치중되어 있어, 전체 은하 집단을 대표하지 못한다. 따라서 본 논문의 목표는 레이블 없이도 천체 이미지의 의미론적 특징을 추출할 수 있는 Self-supervised representation learning 방법을 제안하고, 이를 통해 은하 형태 분류(Morphology classification)와 광도 적색편이 추정(Photometric redshift estimation)과 같은 하위 작업(Downstream tasks)에서 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 Contrastive Learning 프레임워크를 천문학 이미지에 적용하여, 레이블 없이도 이미지의 본질적인 특징을 보존하는 저차원 표현(Representation)을 학습하는 것이다. 특히, 일반적인 컴퓨터 비전 분야의 증강 기법 대신 천문학적 도메인 지식을 반영한 데이터 증강(Data Augmentation) 전략을 설계하여 모델이 천체 이미지의 특성에 맞는 불변성(Invariance)을 학습하도록 유도하였다. 이를 통해 매우 적은 양의 레이블만으로도 지도 학습 모델을 능가하는 성능을 낼 수 있음을 입증하였다.

## 📎 Related Works

기존의 천문학 이미지 분석은 주로 전문가의 수동 검사나 Galaxy Zoo와 같은 크라우드 소싱 기반의 분류에 의존하였다. 최근에는 CNN 기반의 지도 학습 방법들이 은하 형태 분류 및 적색편이 추정에 적용되어 성과를 거두었으나, 앞서 언급한 레이블 부족 문제와 선택 편향 문제라는 한계가 있었다.

비지도 학습(Unsupervised learning) 방법론들이 은하 형태 분석이나 이상 탐지(Anomaly detection)에 적용된 사례가 있으나, 역사적으로 비지도 학습으로 얻은 표현력은 지도 학습에 비해 낮았다. 하지만 최근 컴퓨터 비전 분야에서 SimCLR나 MoCo와 같은 Self-supervised learning(SSL) 기술이 비약적으로 발전하며 지도 학습과의 격차를 줄였으며, 본 논문은 이러한 최신 SSL 기법을 천문학 도메인에 맞게 최적화하여 적용함으로써 기존 접근 방식의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 논문은 크게 두 단계의 파이프라인으로 구성된다. 첫 번째 단계는 대규모의 레이블 없는 SDSS(Sloan Digital Sky Survey) 이미지 데이터셋을 사용하여 Contrastive Learning을 통해 범용적인 이미지 표현(Representation)을 학습하는 Pre-training 단계이다. 두 번째 단계는 학습된 표현을 고정(Frozen)하거나 미세 조정(Fine-tuning)하여 구체적인 과학적 태스크(형태 분류, 적색편이 추정)를 수행하는 Downstream task 단계이다.

### 2. 천문학 특화 데이터 증강 (Data Augmentations)

모델이 천체 이미지의 불변적인 특징을 학습하도록 하기 위해 다음과 같은 도메인 특화 증강 기법을 제안한다.

- **Galactic extinction**: 은하수 내의 먼지에 의한 붉어짐 현상을 모델링하기 위해 인위적인 reddening 값을 샘플링하여 적용한다.
- **Point Spread Function (PSF)**: 관측 장비 및 환경에 따른 PSF의 불일치성을 해결하기 위해 파장 의존적인 가우시안 스무딩(Gaussian smoothing)을 적용한다.
- **Rotation**: 은하의 겉보기 방향에 상관없이 동일한 객체로 인식하도록 $U(0, 2\pi)$ 범위의 무작위 회전을 적용한다.
- **Random jitter & crop**: 이미지 중심 위치의 미세한 차이에 강건하도록 이미지를 약간 이동(Jitter)시킨 후 중심부를 크롭한다.
- **Gaussian noise**: 기기 노이즈에 대응하기 위해 각 채널별로 계산된 MAD(Median Absolute Deviation)를 기준으로 가우시안 노이즈를 추가한다.

### 3. 학습 목표 및 손실 함수

본 모델은 동일한 이미지에서 파생된 두 뷰(View) $\mathbf{x}^q, \mathbf{x}^k_+$는 서로 가깝게, 서로 다른 이미지에서 파생된 뷰 $\mathbf{x}^q, \mathbf{x}^k_-$는 서로 멀게 배치하도록 학습한다. 이를 위해 InfoNCE 손실 함수를 사용한다.

$$L_{q,k+, \{k-\}} = -\log \left( \frac{\exp(\text{sim}(\mathbf{z}^q, \mathbf{z}^k_+))}{\exp(\text{sim}(\mathbf{z}^q, \mathbf{z}^k_+)) + \sum_{k-} \exp(\text{sim}(\mathbf{z}^q, \mathbf{z}^k_-))} \right)$$

여기서 $\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\tau \|\mathbf{a}\| \|\mathbf{b}\|}$는 온도 하이퍼파라미터 $\tau$로 정규화된 코사인 유사도이다.

### 4. 네트워크 아키텍처 및 학습 절차

- **Encoder**: ResNet50을 기반으로 하며, $64 \times 64$ 크기의 입력 이미지에 맞게 첫 번째 Convolution 레이어의 stride를 1로 수정하고 MaxPool 레이어를 제거하여 활성화 맵의 해상도를 유지하였다. 입력 채널은 $ugriz$ 5개 밴드에 맞춰 5채널로 설정하였다.
- **Projection Head**: 2048차원의 인코더 출력 $\mathbf{z}$를 128차원의 공간으로 매핑하는 2층 MLP를 사용하며, Pre-training 완료 후 이 헤드는 제거된다.
- **Momentum Encoder**: Negative sample의 일관성을 유지하기 위해 메인 인코더의 가중치를 지수 이동 평균(EMA) 방식으로 업데이트하는 모멘텀 인코더와 62k 크기의 큐(Queue)를 운용한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 약 120만 개의 SDSS 은하 이미지 ($64 \times 64$ 픽셀, 5개 밴드).
- **태스크 1 (은하 형태 분류)**: Galaxy Zoo 2(GZ2) 레이블을 사용하여 이진 분류 수행.
- **태스크 2 (광도 적색편이 추정)**: 분광학적 적색편이(spec-z) 레이블을 사용하여 $0 \le z \le 0.4$ 범위의 180개 빈(bin)에 대한 분류 문제로 정의.
- **평가 지표**: 분류 작업에서는 Accuracy, AUC, $\eta$(Outlier percentage)를 사용하였고, 적색편이 추정에서는 $\sigma_{MAD}$와 $\eta$를 사용하였다.

### 2. 주요 결과

- **표현 공간 시각화**: UMAP을 통한 시각화 결과, SSL로 학습된 표현 공간이 은하의 형태적 특징과 적색편이에 따라 의미론적으로 잘 군집화되어 있음을 확인하였다. 특히 픽셀 값 기반의 UMAP과 달리, SSL 표현은 회전이나 노이즈 증강에 대해 매우 강건한 불변성을 보였다.
- **은하 형태 분류**:
  - 제한된 레이블 환경에서 SSL 기반의 선형 분류기(Linear classifier)와 미세 조정 모델이 지도 학습 모델을 압도하였다.
  - 미세 조정된 SSL 모델은 지도 학습 모델과 동일한 성능을 내기 위해 필요한 레이블의 양을 약 16배 정도 줄일 수 있었다.
- **광도 적색편이 추정**:
  - 모든 학습 데이터 비율에서 미세 조정된 SSL 모델이 지도 학습 베이스라인보다 낮은 $\sigma_{MAD}$와 $\eta$를 기록하였다.
  - SSL Pre-training을 통해 얻은 성능 이득은 지도 학습 시 데이터를 2~4배 더 많이 사용한 것과 맞먹는 효과를 냈으며, 결과적으로 CNN 기반 photo-z 예측에서 새로운 State-of-the-art(SOTA) 성능을 달성하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 의의

본 연구는 천문학 이미지라는 특수한 도메인에서 Self-supervised learning이 매우 효과적임을 입증하였다. 특히 레이블이 부족한 상황에서 SSL이 강력한 Prior를 제공하여 모델의 일반화 성능을 높인다는 점이 핵심이다. 또한, 단순한 선형 분류기만으로도 높은 성능을 낼 수 있다는 점은 머신러닝 전문가가 아니더라도 학습된 표현을 통해 쉽게 과학적 분석을 수행할 수 있음을 의미한다.

### 2. 한계 및 비판적 해석

- **데이터 증강의 의존성**: 실험 결과 PSF 스무딩 증강은 오히려 성능을 떨어뜨리는 경향이 있었으며, 이는 CNN의 풀링 레이어가 이미 어느 정도의 스무어링에 강건하기 때문으로 해석된다. 즉, 도메인 특화 증강을 설계할 때 네트워크 아키텍처와의 상호작용을 세밀하게 고려해야 한다.
- **선택 편향 문제**: 논문에서는 SSL이 레이블링된 데이터의 선택 편향을 완화할 수 있다고 주장한다. 실제로 레이블 없는 방대한 데이터를 통해 전체 데이터 분포를 먼저 학습하므로, 특정 부분집합에 치우친 지도 학습보다 더 강건한 모델을 만들 가능성이 크다.
- **미해결 질문**: 은하의 크기(Size)와 밝기(Magnitude) 정보가 적색편이 예측에 중요한 힌트가 됨에도 불구하고, 현재 모델은 크기 변화에 대한 증강을 포함하지 않았다. 향후 더 높은 적색편이 영역을 다루기 위해서는 크기/밝기 증강에 대한 추가 연구가 필요하다.

## 📌 TL;DR

본 논문은 레이블이 부족한 천문학 이미지 분석 문제를 해결하기 위해 **Contrastive Self-Supervised Learning**을 제안하였다. 천문학적 특성을 반영한 데이터 증강 기법과 ResNet50 기반의 SSL 프레임워크를 통해, **지도 학습 대비 2~16배 적은 레이블만으로도 동일하거나 더 뛰어난 성능**을 달성하였다. 특히 은하 형태 분류와 적색편이 추정 태스크에서 SOTA 수준의 성능을 보였으며, 이는 향후 거대 천체 관측 데이터셋을 위한 '파운데이션 모델' 구축의 가능성을 시사한다.
