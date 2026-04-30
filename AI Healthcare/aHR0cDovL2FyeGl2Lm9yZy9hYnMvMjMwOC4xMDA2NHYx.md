# Efficient Representation Learning for Healthcare with Cross-Architectural Self-Supervision

Pranav Singh, Jacopo Cirrone (2023)

## 🧩 Problem to Solve

본 연구는 의료 및 생의학 분야에서 딥러닝의 성능을 높이기 위해 필수적인 표현 학습(Representation Learning), 특히 자기지도학습(Self-Supervised Learning, SSL)을 적용할 때 발생하는 극심한 계산 비용과 데이터 제약 문제를 해결하고자 한다. 

일반적으로 최신 SSL 기법들은 매우 큰 배치 사이즈(Batch Size)와 많은 수의 사전 학습 에포크(Pre-training Epochs)를 요구한다. 그러나 실제 임상 환경에서는 다음과 같은 이유로 이러한 요구 사항을 충족하기 어렵다:
1. **데이터 부족**: 의료 데이터는 전문가의 레이블링 비용이 높고, 환자 개인정보 보호 및 질병의 희귀성으로 인해 수집 가능한 데이터 양 자체가 매우 적다.
2. **계산 자원 제한**: 최신 SSL 모델을 학습시키기 위해서는 여러 대의 GPU 서버와 수일의 시간이 소요되며, 이는 일반적인 의료 현장의 컴퓨팅 환경에서 실행 불가능한 수준이다.
3. **성능 저하**: 기존 SOTA(State-of-the-art) SSL 방식들은 배치 사이즈를 줄이거나 학습 시간을 단축할 경우 성능이 급격하게 하락하는 경향이 있다.

따라서 본 논문의 목표는 적은 양의 데이터와 제한된 계산 자원에서도 효율적으로 작동하며, 작은 배치 사이즈와 짧은 학습 시간에도 강건한(Robust) 의료 영상 표현 학습 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **구조적 불변성(Architecture Invariance)**을 활용하는 것이다. 기존의 시아미즈(Siamese) SSL 방식들이 동일한 구조의 네트워크에 서로 다른 데이터 증강(Augmentation)을 적용하여 긍정 쌍(Positive Pair)을 만들었다면, 본 연구에서 제안하는 **CASS(Cross-Architectural Self-Supervision)**는 서로 다른 두 가지 아키텍처인 **CNN(Convolutional Neural Network)**과 **Transformer**를 병렬로 배치하여 긍정 쌍을 생성한다.

이 설계의 중심 직관은 다음과 같다:
- **CNN**은 평행 이동 등가성(Translation Equivariance)과 지역성(Locality)이라는 강력한 귀납적 편향(Inductive Bias)을 가지고 있어 세부적인 지역 특징을 잘 포착한다.
- **Transformer**는 전역적 문맥(Global Context)을 파악하는 능력이 뛰어나지만, CNN과 같은 내재적 편향이 부족하다.
- 따라서 동일한 이미지에 대해 두 아키텍처가 추출하는 표현은 서로 다르며, 이 둘의 유사도를 최적화함으로써 Transformer는 CNN의 지역적 특성을, CNN은 Transformer의 전역적 특성을 서로 학습하게 하여 더 풍부한 표현을 얻을 수 있다.

## 📎 Related Works

### 1. 이미지 분석을 위한 신경망 구조
- **CNN**: 지역성과 평행 이동 등가성을 통해 효율적으로 시각적 개념을 학습하지만, 전역적 문맥 파악에 한계가 있다.
- **Vision Transformer (ViT)**: 셀프 어텐션(Self-attention) 메커니즘을 통해 이미지 전체의 문맥을 파악할 수 있으나, 학습을 위해 막대한 양의 데이터가 필요하며 CNN의 내재적 편향이 부족하다.

### 2. 교차 구조 기법 (Cross-architecture Techniques)
- **하이브리드 방식**: CNN과 Transformer를 하나의 모델 내에 통합하는 방식(예: ConViT, ConvNext)이다.
- **시아미즈 방식**: 아키텍처 변경 없이 두 모델을 쌍으로 배치하여 상호 학습시키는 방식이다. 본 연구의 CASS는 이 방향성을 따르되, 이를 SSL 영역으로 확장하였다.

### 3. 자기지도학습 (Self-Supervised Learning)
- **대조 학습 (Contrastive Learning)**: 긍정 쌍의 거리는 좁히고 부정 쌍의 거리는 넓히는 방식이다. BYOL과 DINO는 부정 쌍 없이도 붕괴(Collapse)를 막는 기법을 도입하여 효율성을 높였다.
- **재구성 기반 학습 (Reconstruction-based)**: 이미지의 일부를 마스킹하고 이를 복원하는 방식(예: MAE)이다.
- **한계**: 이러한 기법들은 대부분 ImageNet과 같은 거대 데이터셋에서 검증되었으며, 의료 영상과 같이 클래스 불균형이 심하고 데이터 양이 적은 환경에서는 배치 사이즈 의존성이 높아 성능이 급격히 저하되는 문제가 있다.

## 🛠️ Methodology

### 전체 시스템 구조
CASS는 하나의 입력 이미지 $X$에 대해 단 한 번의 데이터 증강을 적용하여 $X'$를 생성하고, 이를 CNN(ResNet-50)과 Transformer(ViT-B/16)라는 두 개의 서로 다른 브랜치에 동시에 통과시킨다. 두 네트워크에서 출력된 로짓(Logits)을 비교하여 유사도를 최적화하는 구조이다.

### 학습 목표 및 손실 함수
CASS는 BYOL과 유사하게 코사인 유사도 손실 함수(Cosine Similarity Loss)를 사용하여 두 아키텍처의 출력 벡터가 서로 일치하도록 학습한다.

$$loss = 2 - 2 \times F(R) \times F(T)$$

여기서 $R$과 $T$는 각각 CNN과 Transformer에서 추출된 임베딩이며, $F(x)$는 다음과 같이 L2 정규화를 수행하는 함수이다.

$$F(x) = \frac{x}{\max(\|x\|_2, \epsilon)}$$

### CASS의 특징 및 추론 절차
1. **붕괴 방지**: 기존 SSL은 붕괴를 막기 위해 모멘텀 인코더, 스톱 그라디언트, 또는 센터링(Centering) 같은 복잡한 기법을 사용한다. 그러나 CASS는 CNN과 Transformer가 본질적으로 서로 다른 표현을 생성한다는 점을 이용하여, 추가적인 수학적 장치 없이도 자연스럽게 붕괴를 방지한다.
2. **계산 효율성**: 
    - 데이터 증강을 한 번만 적용하므로 연산량이 줄어든다.
    - Teacher-Student 네트워크 간의 파라미터 전이(Lagging parameter update) 과정이 없어 계산 오버헤드가 적다.
3. **학습 절차**: Adam 옵티마이저와 코사인 스케줄러를 사용하며, 가중치 평균화 기법인 SWA(Stochastic Weight Averaging)를 적용한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 
    - Autoimmune (198장, 초소형 데이터셋)
    - Dermofit (1,300장, 피부 질환)
    - Brain tumor MRI (7,022장, 뇌종양)
    - ISIC 2019 (25,331장, 피부 병변, 클래스 불균형 심함)
- **비교 대상**: DINO, BYOL, MAE, 그리고 단순 전이 학습(Transfer Learning).
- **평가 지표**: F1 Score (Autoimmune, Dermofit, Brain MRI), Balanced Multi-class Accuracy/Recall (ISIC 2019).
- **레이블 비율**: 데이터의 1%, 10%, 100%만 사용하여 파인튜닝(Fine-tuning)을 진행함으로써 데이터 효율성을 측정하였다.

### 주요 결과
1. **정량적 성능**: 
    - DINO 대비 평균 성능 향상 폭은 레이블 비율에 따라 **1% $\to$ 3.8%**, **10% $\to$ 5.9%**, **100% $\to$ 10.13%**로 나타났다.
    - 특히 매우 적은 데이터(1% 레이블) 상황에서 CASS의 우위가 두드러졌다.
2. **계산 효율성**: 
    - 단일 RTX8000 GPU 기준, 사전 학습 시간을 DINO 대비 평균 **69% 단축**시켰다.
3. **강건성(Robustness)**: 
    - 배치 사이즈(8, 16, 32)와 사전 학습 에포크 수의 변화에 따른 성능 변동성(Variance)을 분석한 결과, CASS가 DINO보다 훨씬 낮은 변동성을 보이며 강건하게 작동함을 확인하였다.
4. **정성적 분석 (Attention Map)**: 
    - CASS로 학습된 Transformer의 어텐션 맵이 단순 지도 학습(Supervised) 모델보다 더 연결되어 있고 지역적 특징을 더 잘 포착함을 확인하였다. 이는 CNN으로부터 지역적 정보가 성공적으로 전이되었음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 해석
CASS는 '아키텍처 간의 상보적 관계'를 SSL의 핵심 동력으로 활용하였다. CNN의 지역적 편향과 Transformer의 전역적 능력을 결합함으로써, 데이터가 극도로 부족한 의료 환경에서도 유의미한 표현을 학습할 수 있었다. 특히, 매우 작은 배치 사이즈에서도 성능 저하가 적다는 점은 컴퓨팅 자원이 제한된 실제 의료 현장에서의 적용 가능성을 크게 높인다.

### 한계 및 비판적 논의
1. **추론 시 모델 선택 문제**: CASS는 CNN과 Transformer 두 개의 모델을 동시에 학습시키지만, 추론 단계에서 어떤 모델(또는 어떤 조합)을 최종적으로 사용할지에 대한 명확한 기준이 제시되지 않았다.
2. **데이터셋의 국한성**: Autoimmune 데이터셋의 경우 특정 기관의 데이터만 사용되었으므로, 일반화 성능에 대한 추가 검증이 필요하다.
3. **메타데이터 부재**: 실제 의료 진단에서는 영상 외에도 환자의 병력, 혈액 검사 결과 등 메타데이터가 매우 중요하지만, 본 연구는 오직 영상 데이터만을 활용하였다.

## 📌 TL;DR

본 논문은 의료 영상 분석의 고질적인 문제인 데이터 부족과 고비용 계산 문제를 해결하기 위해, CNN과 Transformer를 병렬로 배치하여 상호 학습시키는 **CASS(Cross-Architectural Self-Supervision)** 기법을 제안한다.

- **핵심 기여**: 데이터 증강 대신 '서로 다른 아키텍처'를 통해 긍정 쌍을 생성하여, CNN의 지역성과 Transformer의 전역성을 동시에 학습한다.
- **성과**: SOTA 방식인 DINO 대비 학습 시간을 **69% 단축**하면서도, 적은 레이블 데이터 환경에서 더 높은 분류 성능을 보였으며 배치 사이즈 변화에도 매우 강건하다.
- **의의**: 계산 자원이 부족한 환경이나 희귀 질환과 같이 데이터 수집이 어려운 의료 분야에서 딥러닝 모델을 효율적으로 구축할 수 있는 실용적인 방법론을 제시하였다.