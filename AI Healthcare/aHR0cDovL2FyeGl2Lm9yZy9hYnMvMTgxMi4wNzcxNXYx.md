# A Tour of Unsupervised Deep Learning for Medical Image Analysis

Khalid Raza and Nripendra Kumar Singh (2018)

## 🧩 Problem to Solve

현대 의료 진단 및 치료 과정에서 고차원(high-dimensional)의 이질적인(heterogeneous) 의료 영상 데이터를 해석하는 것은 헬스케어 혁신의 핵심적인 과제이다. 전통적으로 의료 영상은 방사선 전문의나 의사와 같은 인간 전문가에 의해 해석되어 왔으나, 병리학적 변동성이 크고 전문가의 피로도가 높다는 한계가 있다.

이를 해결하기 위해 컴퓨터 보조 진단(Computer-Assisted Diagnosis, CAD) 시스템이 도입되었으며, 특히 딥러닝 기술이 주목받고 있다. 하지만 지도 학습(Supervised Learning) 기반의 접근 방식은 다음과 같은 세 가지 주요 문제점을 가진다. 첫째, 클래스 레이블을 생성하기 위한 막대한 수동 작업(manual effort)이 필요하다. 둘째, 레이블링 과정에서 발생하는 인간의 편향(bias)이 알고리즘에 전이되어 예외 케이스를 처리하는 능력을 제한한다. 셋째, 타겟 함수(target function)의 확장성(scalability)이 떨어진다.

따라서 본 논문의 목표는 외부의 편향 없이 데이터 자체에서 통찰을 직접 도출하고 데이터를 그룹화하여 데이터 기반 의사결정을 가능하게 하는 비지도 학습(Unsupervised Learning) 모델들을 체계적으로 분석하고, 의료 영상 분석 분야에서의 적용 사례와 향후 과제를 제시하는 것이다.

## ✨ Key Contributions

본 논문은 의료 영상 분석에 적용 가능한 비지도 딥러닝 모델의 전반적인 지형을 조사하고 이를 체계적으로 정리하였다. 중심적인 기여 사항은 다음과 같다.

1.  **비지도 학습 작업의 분류 체계(Taxonomy) 정립**: 비지도 학습을 밀도 추정(Density estimation), 차원 축소(Dimensionality reduction), 클러스터링(Clustering)의 세 가지 주요 작업으로 구분하여 설명한다.
2.  **비지도 딥러닝 모델의 상세 분석**: Autoencoder와 그 변형 모델들, Restricted Boltzmann Machines (RBM), Deep Belief Networks (DBN), Deep Boltzmann Machines (DBM), 그리고 Generative Adversarial Networks (GAN)의 구조와 원리를 상세히 분석한다.
3.  **의료 영상 분석 적용 사례의 매핑**: 각 모델이 실제로 어떤 의료 영상(MRI, CT, PET, X-ray 등)과 어떤 작업(분류, 세그멘테이션, 잡음 제거 등)에 사용되었는지를 구체적인 문헌 사례를 통해 제시한다.
4.  **실무적 리소스 제공**: 연구자들이 즉시 활용할 수 있는 비지도 학습 소프트웨어 도구/패키지 리스트와 의료 영상 벤치마크 데이터셋 목록을 제공한다.

## 📎 Related Works

논문은 의료 영상 분석의 패러다임이 수동 해석에서 CAD로 전환되고 있음을 언급하며, 기존의 지도 학습 기반 딥러닝(CNN, RNN, FFNN 등)의 한계를 지적한다. 지도 학습은 레이블링된 데이터에 과도하게 의존하며, 이는 의료 데이터의 특성상 확보하기 어렵거나 편향될 가능성이 높다는 점이 기존 접근 방식의 가장 큰 제약이다.

반면, 비지도 학습은 레이블이 없는 데이터에서 잠재적인 특징(hidden features)을 학습하므로, 지도 학습의 전처리 단계로 활용되어 분류기의 일반화 성능을 높이거나, 데이터 압축, 잡음 제거, 초해상도(super resolution) 등 다양한 보조 작업에 활용될 수 있다는 차별점을 가진다.

## 🛠️ Methodology

본 논문은 특정 새로운 알고리즘을 제안하는 것이 아니라, 기존의 비지도 학습 모델들을 체계적으로 리뷰하는 형식을 취하고 있다. 분석 대상이 된 주요 모델들의 핵심 방법론은 다음과 같다.

### 1. Autoencoders (AE) 및 변형 모델
Autoencoder는 입력을 잠재 표현(latent representation)으로 압축하는 Encoder $f_\theta$와 이를 다시 원래 입력으로 복원하는 Decoder $g_\theta$로 구성된다. 학습 목표는 입력과 출력 사이의 재구성 오차(reconstruction error)를 최소화하는 것이다.

- **Stacked Autoencoders (SAE)**: 여러 개의 AE를 층층이 쌓아 올린 구조로, Greedy layer-wise training을 통해 깊은 네트워크의 표현력을 확보한다.
- **Denoising Autoencoder (DAE)**: 입력 데이터에 의도적으로 잡음을 섞은 $\tilde{x}$를 입력하여 깨끗한 원본 $x$를 복원하게 함으로써, 단순한 항등 함수 학습을 방지하고 더 강건한 특징을 추출한다.
- **Sparse Autoencoder**: 은닉층의 뉴런 중 일부만 활성화되도록 제약을 가한다. 이를 위해 KL Divergence를 이용한 페널티 항을 손실 함수에 추가하여 뉴런의 평균 활성도를 $\rho$ (예: 0.05)에 가깝게 유지한다.
- **Convolutional Autoencoder (CAE)**: Fully connected layer 대신 Convolutional layer를 사용하여 이미지의 지역적 구조(local structure)를 보존하며 특징을 학습한다.
- **Variational Autoencoder (VAE)**: 잠재 변수를 특정 값이 아닌 확률 분포(주로 가우시안 분포)로 학습하는 생성 모델이다. SGVB(Stochastic Gradient Variational Bayes) 등을 통해 최적화한다.
- **Contractive Autoencoder**: 입력의 미세한 변화에 강건하도록 Jacobian 행렬의 Frobenius norm을 정규화 항으로 추가하여 표현의 안정성을 높인다.

### 2. Restricted Boltzmann Machines (RBM)
RBM은 가시층(visible layer)과 은닉층(hidden layer)으로 구성된 무방향 그래프 모델(undirected graphical model)이다. 층 내 뉴런 간의 연결은 없으며, 오직 가시층과 은닉층 사이의 양방향 연결만 존재하여 데이터의 확률 분포를 학습하는 생성 모델로 작동한다.

### 3. Deep Belief Networks (DBN)
DBN은 여러 개의 RBM을 계층적으로 쌓아 올린 구조이다. 하위 층에서 저수준 특징을 학습하고, 이를 바탕으로 상위 층에서 고수준 특징을 학습하는 Greedy layer-wise unsupervised training 방식을 사용한다.

### 4. Deep Boltzmann Machines (DBM)
DBM은 DBN과 유사하게 RBM을 쌓은 형태지만, 모든 층이 무방향 연결을 가지는 생성 모델이다. 상위 층과 하위 층의 정보를 동시에 결합하여 DBN보다 더 강력한 표현력을 가진다.

### 5. Generative Adversarial Networks (GAN)
GAN은 생성자(Generator, $G$)와 판별자(Discriminator, $D$)라는 두 네트워크가 서로 경쟁하는 구조이다. $G$는 실제 데이터와 유사한 가짜 데이터를 생성하려 하고, $D$는 입력 데이터가 실제인지 가짜인지 판별한다. 이 과정은 다음과 같은 Mini-max 게임으로 정의된다.

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

## 📊 Results

본 논문은 개별 실험 결과보다는 기존 문헌들을 분석하여 모델별 적용 사례를 정리한 결과(Table 2, 3, 4, 5, 6)를 제시한다.

- **Autoencoder 계열**: 알츠하이머(AD) 및 경도인지장애(MCI) 분류(MRI/PET), 디지털 병리 이미지의 핵(nucleus) 검출, 유방암 밀도 분류 등에 광범위하게 사용되었다. 특히 SAE는 3D CNN의 사전 학습(pre-training) 도구로 활용되어 성능을 높였다.
- **RBM**: 알츠하이머 변이 탐지, 다발성 경화증 병변 세그멘테이션, 유방암 덩어리 검출 등에 적용되었으며, Random Forest와 결합하여 분류 정확도를 높인 사례가 보고되었다.
- **DBN**: fMRI 영상의 특징 추출, 자폐 스펙트럼 장애 분류, 심장 좌심실 세그멘테이션 등에 활용되었다.
- **DBM**: 심장 운동 추적(heart motion tracking), 의료 영상 검색(image retrieval) 등에 적용되어 강건한 표현력을 입증하였다.
- **GAN**: 망막 이미지 합성, 흉부 X-ray의 사실적인 이미지 생성, PET 데이터 합성 등 주로 데이터 증강(data augmentation) 및 시뮬레이션 용도로 사용되었다.

## 🧠 Insights & Discussion

### 강점 및 기회
비지도 학습은 데이터 자체의 통찰을 직접 도출하므로 레이블링 비용을 획기적으로 줄일 수 있으며, 지도 학습의 전처리 단계로서 분류기의 일반화 성능을 향상시키는 'Holy Grail'의 역할을 할 수 있다. 또한, 최근의 GPU 및 클라우드 컴퓨팅 인프라의 발전은 복잡한 비지도 딥러닝 모델의 대규모 데이터 처리를 가능하게 하고 있다.

### 한계 및 도전 과제
1.  **평가의 어려움**: 정답 레이블이 없기 때문에 알고리즘이 실제로 유용한 특징을 학습했는지 정량적으로 평가하기 매우 어렵다.
2.  **모델 및 하드웨어 선택**: 데이터의 특성에 따라 최적의 알고리즘과 하드웨어 요구사항이 크게 달라지므로 선택 과정이 까다롭다.
3.  **블랙박스(Black-box) 문제**: 딥러닝 모델의 의사결정 과정이 불투명하여, 신뢰성이 중요한 의료 현장에서 의료 전문가들이 이를 수용하는 데 한계가 있다.
4.  **데이터의 이질성**: 표준화되지 않은 다양한 획득 프로토콜을 가진 이질적인 의료 데이터에서도 작동하는 범용적인 알고리즘 개발이 필요하다.

### 비판적 해석
본 논문은 광범위한 모델을 체계적으로 정리하였으나, 개별 모델들의 성능을 직접적으로 비교한 벤치마크 결과가 부족하다. 단순히 "적용되었다"는 나열식 구성보다는, 특정 작업(예: 세그멘테이션)에서 어떤 비지도 모델이 왜 더 우수한지를 분석하는 심층적인 비교 연구가 병행되었다면 더 높은 학술적 가치를 가졌을 것으로 판단된다.

## 📌 TL;DR

본 논문은 의료 영상 분석에서 레이블링 비용과 편향 문제를 해결하기 위해 비지도 딥러닝(UDL)의 역할과 모델들을 종합적으로 분석한 리뷰 논문이다. Autoencoder, RBM, DBN, DBM, GAN 등의 모델이 어떻게 의료 영상의 특징 추출, 분류, 세그멘테이션, 이미지 합성에 활용되는지를 체계적으로 정리하였으며, 향후 의료 AI가 나아가야 할 방향으로 모델의 해석 가능성(interpretability) 확보와 이질적 데이터에 대한 강건성 향상을 제시한다. 이 연구는 레이블이 부족한 의료 환경에서 비지도 학습이 단순한 전처리를 넘어 핵심적인 진단 도구로 발전할 가능성을 시사한다.