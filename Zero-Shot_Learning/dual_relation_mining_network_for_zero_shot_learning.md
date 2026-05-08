# Dual Relation Mining Network for Zero-Shot Learning

Jinwei Han, Yingguo Gao, Zhiwen Lin, Ke Yan, Shouhong Ding, Yuan Gao, Gui-Song Xia (2024)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Learning (ZSL)에서 발생하는 시각적-의미적 관계(visual-semantic relationship)의 모호성과 속성 간의 의미적 관계(semantic relationship) 결여 문제를 해결하고자 한다.

ZSL의 핵심은 학습 단계에서 본 적 없는 클래스(unseen classes)를 인식하기 위해, 본 클래스(seen classes)로부터 공유된 의미적 지식(예: attributes)을 전이하는 것이다. 최근 Spatial Attention 메커니즘을 통해 시각적 특징과 속성을 정렬하는 방법들이 발전하였으나, 다음과 같은 두 가지 한계점이 존재한다.

첫째, 기존 방법들은 시각적-의미적 관계를 오직 공간적 차원에서만 탐색한다. 이로 인해 서로 다른 속성이 유사한 공간적 어텐션 영역을 공유할 경우 분류 모호성(classification ambiguity)이 발생하며, 이는 특히 세밀한 구분(fine-grained)이 필요한 작업에서 성능 저하를 야기한다.

둘째, 속성들 사이의 내재적인 의미적 관계(semantic relationship)에 대한 논의가 부족하다. 속성 간의 관계를 활용하면 데이터가 부족한 상황에서도 더 강건하고 일반화된 표현을 학습할 수 있어 지식 전이 효율을 높일 수 있다.

따라서 본 논문의 목표는 시각적-의미적 관계와 의미적-의미적 관계를 동시에 마이닝하는 Dual Relation Mining Network (DRMN)를 통해 ZSL의 분류 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시각적 특징과 의미적 속성 사이의 관계뿐만 아니라, 속성들 간의 상호 관계를 함께 학습하여 지식 전이의 효율성을 극대화하는 것이다. 이를 위해 다음과 같은 세 가지 핵심 설계 요소를 도입하였다.

1. **Dual Attention Block (DAB):** 다단계 특징 융합(Multi-level Feature Fusion)을 통해 시각적 정보를 풍부하게 하고, 공간적 어텐션(Spatial Attention)과 속성 기반 채널 어텐션(Attribute-guided Channel Attention)을 결합하여 시각적-의미적 관계를 정교하게 추출한다. 특히 채널 어텐션은 공간적으로 겹치는 영역에서도 속성들을 구분해내는 디커플링(decoupling) 역할을 수행한다.
2. **Semantic Interaction Transformer (SIT):** 속성 표현들 사이의 상호작용을 모델링하는 Transformer Encoder를 도입하여, 이미지 간 속성 표현의 일반화 능력을 강화한다.
3. **Global Classification Branch:** 인간이 정의한 속성 외에 모델이 스스로 학습하는 잠재적 특징(latent features)을 보완하기 위해 글로벌 분류 브랜치를 추가하고, 이를 속성 기반 분류 결과와 앙상블하여 최종 예측 성능을 높인다.

## 📎 Related Works

ZSL 연구는 크게 두 가지 방향으로 나뉜다.

1. **Generative-based methods:** GAN이나 VAE를 사용하여 unseen 클래스의 가상 특징을 생성하고, 이를 통해 ZSL 문제를 일반적인 지도 학습 분류 문제로 전환하는 방식이다. 그러나 실제 이미지와 일관되면서도 변별력 있는 다양한 특징을 생성하는 데 어려움이 있다.
2. **Embedding-based methods:** 시각적 특징과 의미적 특징을 임베딩 함수를 통해 연결하는 방식이다. 최근의 Attention 기반 방법들은 이미지의 변별력 있는 영역에 집중하여 시각적-의미적 상호작용을 꾀하지만, 대부분 공간적 어텐션에만 의존하여 속성 간의 중복 영역 문제와 속성 간의 의미적 관계를 간과하는 한계가 있다.

본 연구는 이러한 기존의 임베딩 기반 방법론에서 한 단계 나아가, 채널 차원의 디커플링과 Transformer 기반의 속성 관계 모델링을 통해 시각적-의미적 및 의미적-의미적 관계를 모두 고려한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

DRMN은 동일한 Backbone을 공유하는 두 개의 브랜치로 구성된다. 하나는 속성 기반 분류 브랜치(Attribute-based classification branch)이며, 다른 하나는 글로벌 분류 브랜치(Global classification branch)이다. 테스트 단계에서는 두 브랜치의 결과를 앙상블하여 최종 클래스를 결정한다.

### 2. Dual Attention Block (DAB)

DAB는 시각적-의미적 관계를 마이닝하며, 세 단계의 과정을 거친다.

**가. 다단계 특징 융합 (Multi-level Feature Fusion, MFF)**
Backbone의 마지막 블록에서 나오는 고수준 특징만으로는 저수준의 세부 패턴이 손실될 수 있다. 이를 방지하기 위해 다양한 레벨의 특징을 융합하여 풍부한 시각적 표현 $v_i$를 생성한다.

**나. 공간적 어텐션 (Spatial Attention)**
학습 가능한 속성 프로토타입 $p_a$를 사용하여 각 속성에 해당하는 이미지 영역의 가중치 $\omega$를 계산한다.
$$\omega(p_a, v_{ri}) = \frac{\exp(p_a^\top W_1 v_{ri})}{\sum_{r'=1}^{R} \exp(p_a^\top W_1 v_{r'i})}$$
여기서 $v_{ri}$는 영역 $r$의 시각적 특징이며, $W_1$은 호환성을 측정하는 학습 가능 행렬이다. 이를 통해 추출된 속성 특징 $k_{ai}$는 다음과 같다.
$$k_{ai} = \sum_{r=1}^{R} \omega(p_a, v_{ri}) v_{ri}$$

**다. 속성 기반 채널 어텐션 (Attribute-guided Channel Attention, ACA)**
공간적 어텐션만으로는 서로 다른 속성이 유사한 영역을 가질 때 구분하기 어렵다. 이를 해결하기 위해 채널 디스크립터 $q_{ai}$를 생성하고 MLP와 시그모이드 함수를 거쳐 채널 가중치 $\eta_{ai}$를 얻는다.
$$q_{ai} = \text{norm}(p_a) + \text{norm}\left(\frac{1}{R} \sum_{r=1}^{R} v_{ri}\right)$$
$$\eta_{ai} = \sigma(\text{MLP}(q_{ai}))$$
최종적으로 디커플링된 의미적 특징 $h_{ai}$는 $h_{ai} = k_{ai} \cdot \eta_{ai}$로 계산된다.

### 3. Semantic Interaction Transformer (SIT)

속성 간의 관계를 학습하기 위해, 배치 내의 모든 속성 특징 $H \in \mathbb{R}^{B \times A \times C}$를 시퀀스로 취급하여 Transformer Encoder에 입력한다.
$$H' = \text{LN}(\text{MHSA}(H) + H)$$
$$\hat{H} = \text{LN}(\text{MLP}(H') + H')$$
SIT는 학습 시에만 사용되며, 테스트 시에는 제거 가능한 plug-and-play 모듈로 동작한다.

### 4. Hyperspherical Classifier 및 Global Branch

**Hyperspherical Classifier:** 예측된 속성 점수 $e_i$와 클래스 의미 벡터 $z_c$를 $l_2$-정규화하여 하이퍼스피어 공간으로 투영한 뒤, 스케일링된 코사인 유사도를 통해 클래스 로짓 $o_{ci}$를 계산한다.
$$o_{ci} = \left(\gamma \cdot \frac{z_c}{\|z_c\|}\right)^\top \left(\gamma \cdot \frac{e_i}{\|e_i\|}\right)$$

**Global Classification Branch:** 인간이 정의한 속성 외의 잠재적 특징을 학습하기 위해, 전역 평균 풀링(GAP) 후 선형 레이어를 통해 클래스 로짓 $g_i$를 직접 도출한다.

### 5. 손실 함수 및 학습 절차

전체 손실 함수는 속성 기반 분류 손실($L_{AC}$)과 글로벌 분류 손실($L_{GC}$)의 합으로 정의된다.
$$L_{total} = L_{AC} + \lambda_{GC} L_{GC}$$
$L_{AC}$에는 seen 클래스에 대한 과적합을 방지하기 위한 self-calibration 항이 포함되어 있다.

## 📊 Results

### 실험 설정

- **데이터셋:** CUB (새), SUN (장면), AwA2 (동물)
- **평가 지표:** Conventional ZSL(CZSL)에서는 Top-1 Accuracy를, Generalized ZSL(GZSL)에서는 Seen 클래스 정확도($S$), Unseen 클래스 정확도($U$), 그리고 이들의 조화 평균인 Harmonic Mean($H$)을 측정하였다.
- **구현 세부사항:** ResNet101을 Backbone으로 사용하였으며, Adam 옵티마이저를 통해 학습하였다.

### 주요 결과

Table 1에 따르면, DRMN은 세 가지 벤치마크 모두에서 SOTA 성능을 달성하였다.

- **CZSL:** CUB(82.5%), SUN(66.9%), AwA2(74.6%)로 최고 성능을 기록하였다.
- **GZSL (H-mean):** CUB(76.8%)와 SUN(45.8%)에서 최고치를 기록하였으며, AwA2(71.5%)에서도 매우 경쟁력 있는 성능을 보였다.

### 분석 및 어블레이션 연구

- **모듈별 기여도:** DAB는 CUB 데이터셋의 H-mean을 baseline 대비 약 9.6% 향상시켰으며, SIT와 Global Branch 또한 유의미한 성능 향상을 가져왔다.
- **DAB 내부 분석:** 다단계 특징 융합(MFF)과 채널 어텐션(ACA) 모두 성능 향상에 필수적임을 확인하였다.
- **정성적 결과:** Attention Map 시각화 결과, baseline보다 DRMN이 속성과 관련된 이미지 영역을 더 정확하게 로컬라이즈함을 확인하였다. 이는 MFF가 세밀한 부위(예: 부리 모양) 포착에 도움을 주었기 때문이다.

## 🧠 Insights & Discussion

본 논문은 ZSL에서 단순한 '정렬(alignment)'을 넘어 '관계 마이닝(relation mining)'의 중요성을 입증하였다. 특히 공간적 정보만으로는 해결할 수 없는 속성 간의 중복성 문제를 채널 어텐션을 통한 디커플링으로 해결한 점이 돋보인다.

또한, 인간이 정의한 Semantic Attribute가 완벽하지 않을 수 있다는 점을 인정하고, Global Classification Branch라는 보완책을 통해 latent feature를 함께 학습한 전략은 매우 실용적인 접근이다.

다만, SIT 모듈이 학습 시에만 사용되고 테스트 시에는 제거된다는 점은, 학습 시의 배치 간 상호작용이 테스트 시의 단일 이미지 추론에 어떻게 전이되는지에 대한 더 깊은 이론적 분석이 필요함을 시사한다. 또한 하이퍼파라미터 $\beta$ (앙상블 가중치)와 $\lambda_{GC}$ (손실 가중치)가 데이터셋마다 다르게 설정되어, 이에 대한 자동화된 최적화 방법이 제시되지 않은 점은 한계로 볼 수 있다.

## 📌 TL;DR

본 논문은 ZSL의 고질적인 문제인 속성 간 공간적 모호성과 의미적 관계 결여를 해결하기 위해 **Dual Relation Mining Network (DRMN)**를 제안한다. **Dual Attention Block (DAB)**를 통해 시각적-의미적 관계를 정교하게 추출하고, **Semantic Interaction Transformer (SIT)**로 속성 간 관계를 모델링하며, **Global Branch**로 잠재 특징을 보완한다. 그 결과 CUB, SUN, AwA2 벤치마크에서 SOTA 성능을 달성하였으며, 이는 복잡한 관계 마이닝이 ZSL의 지식 전이 성능을 극대화할 수 있음을 보여준다.
