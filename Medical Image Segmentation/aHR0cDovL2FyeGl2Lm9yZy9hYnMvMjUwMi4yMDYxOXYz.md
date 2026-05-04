# Style Content Decomposition-based Data Augmentation for Domain Generalizable Medical Image Segmentation

Zhiqiang Shen, Peng Cao, Jinzhu Yang, Osmar R. Zaiane, Zhaolin Chen (2025)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 학습된 모델은 서로 다른 의료 영상 모달리티 간의 도메인 시프트(Domain Shift)로 인해 배포 단계에서 성능이 크게 저하되는 문제를 겪는다. 이러한 도메인 시프트는 크게 두 가지 성분으로 분류할 수 있다. 첫째는 조명, 대비, 색상과 같은 이미지의 전역적 특성의 차이인 '스타일(Style)' 시프트이며, 둘째는 해부학적 구조의 국소적 차이인 '콘텐츠(Content)' 시프트이다.

본 논문의 목표는 이러한 스타일과 콘텐츠 시프트를 모두 완화하여, 학습 시 보지 못한 타겟 도메인에서도 강건하게 작동하는 도메인 일반화(Domain Generalization, DG) 능력을 갖춘 의료 영상 분할 모델을 학습시키는 것이다. 특히, 타겟 도메인의 데이터에 접근할 수 없는 단일 소스(Single-source) 환경에서 효과적인 데이터 증강 기법을 제안하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지를 스타일 코드(Style Code)와 콘텐츠 맵(Content Map)으로 명시적으로 분해한 뒤, 이 두 성분을 각각 독립적으로 변형하여 다양한 가상 도메인을 시뮬레이션하는 것이다.

연구진은 특이값 분해(Singular Value Decomposition, SVD)를 통해 이미지의 전역적 특성을 나타내는 특이값(Singular Values)을 스타일 코드로, 랭크-1 행렬(Rank-one Matrices)을 콘텐츠 맵으로 정의하였다. 이를 바탕으로 스타일 코드의 블렌딩과 콘텐츠 맵의 믹싱을 수행하는 **StyCona** 알고리즘을 제안하였다. 이 방법은 별도의 추가 파라미터 학습이나 모델 구조 변경 없이 기존 분할 모델에 즉시 적용 가능한 플러그 앤 플레이(Plug-and-play) 방식의 데이터 증강 모듈이라는 점에 큰 의의가 있다.

## 📎 Related Works

기존의 도메인 일반화(DG) 연구는 크게 두 가지 패러다임으로 나뉜다.

1. **표현 학습(Representation Learning):** 결정론적 또는 통계적 모델링을 통해 도메인 불변 표현(Domain-invariant representation)을 학습하는 방식이다. 그러나 이러한 방법들은 도메인 특정적인 학습에 의존하는 경향이 있어, 새로운 도메인으로의 일반화 능력이 제한적일 수 있다.
2. **데이터 증강(Data Augmentation):** 소스 도메인의 데이터 분포를 확장하여 모델이 암시적으로 불변 특징을 찾도록 유도하는 방식이다. 푸리에 변환(Fourier Transformation), 랜덤 컨볼루션(Random Convolution), 특징 통계 편집(Feature Statistics Editing) 등이 주로 사용되었다.

본 논문은 기존의 증강 기법들이 주로 '스타일' 변형에만 치중되어 있으며, 의료 영상의 핵심인 '콘텐츠(해부학적 구조)'의 변화를 체계적으로 모델링하지 못했다는 한계를 지적하며, 이를 보완하기 위해 스타일-콘텐츠 분해 기반의 접근 방식을 제안한다.

## 🛠️ Methodology

### 전체 파이프라인

StyCona는 크게 세 단계로 구성된다: (1) 스타일-콘텐츠 분해 $\rightarrow$ (2) 스타일 증강 $\rightarrow$ (3) 콘텐츠 증강. 이 과정을 거쳐 생성된 증강 이미지를 사용하여 분할 모델을 학습시킨다.

### 1. 스타일-콘텐츠 분해 (Style Content Decomposition)

이미지 $x$를 SVD를 통해 다음과 같이 분해한다.

$$x = \sum_{r=1}^{R} \sigma_r u_r v_r^T$$

여기서 $R$은 이미지의 랭크(Rank)이며, 각 구성 요소의 역할은 다음과 같다.

- **스타일 코드 ($\sigma_r$):** 각 특이값(Singular Value)은 전역적인 픽셀 강도를 조절하는 스칼라 값으로, 이미지의 전역적 외형(Style)을 결정한다.
- **콘텐츠 맵 ($u_r v_r^T$):** 랭크-1 행렬들의 집합은 이미지의 기저 패턴(Basis pattern) 역할을 하며, 이들의 선형 조합이 이미지의 해부학적 구조(Content)를 정의한다.

### 2. 스타일 증강 (Style Augmentation)

전역적 이미지 특성의 변화를 시뮬레이션하기 위해, 원본 이미지 $x_i$의 스타일 코드와 보조 이미지 $x_j$의 스타일 코드를 원소별로 블렌딩한다.

$$\tilde{\sigma}_r^i = \alpha \times \sigma_r^i + (1 - \alpha) \times \sigma_r^j$$

여기서 $\alpha \sim U(0, 1)$은 믹스 강도를 조절하는 가중치이다.

### 3. 콘텐츠 증강 (Content Augmentation)

국소적 해부학적 구조의 차이를 시뮬레이션하기 위해, 무작위로 선택된 $t$개의 콘텐츠 맵에 대해 좌/우 특이 벡터를 믹싱한다.

$$\bar{u}_r^i = \beta \times u_r^i + (1 - \beta) \times u_r^j \quad \text{or} \quad \bar{v}_r^i = \beta \times v_r^i + (1 - \beta) \times v_r^j$$

여기서 $\beta \sim U(0, 1)$이며, $u, v$는 각각 왼쪽 및 오른쪽 특이 벡터이다.

### 최종 이미지 재구성 및 학습

증강된 스타일 코드와 콘텐츠 맵을 다시 결합하여 최종 증강 이미지 $\bar{\tilde{x}}_i$를 생성한다.

$$\bar{\tilde{x}}_i = \sum_{r=1}^{R} \tilde{\sigma}_r^i \bar{u}_r^i \bar{v}_r^{iT}$$

학습 시에는 이 증강된 이미지를 사용하여 다음과 같은 표준 분할 손실 함수 $L$을 최적화한다.

$$L = \frac{1}{N} \sum_{i=1}^{N} L_{seg}(f(\bar{\tilde{x}}_i, \theta), y_i)$$

## 📊 Results

### 실험 설정

- **데이터셋:**
  - cardiac MRI (MS-CMR): bSSFP $\leftrightarrow$ LGE 시퀀스 간의 교차 도메인 분할.
  - Fundus Image Benchmark: ORIGA를 소스로 하여 REFUGE, Drishti-GS, BinRushed, Magrabia 4개 타겟 도메인에 대한 일반화 성능 평가.
- **기준 모델:** U-Net
- **평가 지표:** DSC (Dice Similarity Coefficient $\uparrow$), ASD (Average Surface Distance $\downarrow$)

### 주요 결과

1. **SOTA 비교:** StyCona는 푸리에 기반(AmpMix 등), 특징 통계 기반(MixStyle 등), 랜덤 컨볼루션 기반(CIDA 등)의 최신 DG 방법론들보다 우수한 성능을 보였다.
    - cardiac MRI에서 bSSFP $\rightarrow$ LGE 방향의 DSC가 73.39%로 가장 높았다.
    - Fundus 이미지 실험에서도 평균 DSC 73.16%를 기록하며 가장 강건한 성능을 보였다.
2. **정성적 분석:** StyCona로 학습된 모델은 타겟 도메인의 이미지에서 객체의 경계를 더욱 정밀하게 묘사하는 것으로 나타났다. 이는 스타일과 콘텐츠 증강이 동시에 이루어져 unseen 도메인의 데이터 분포를 더 넓게 커버했기 때문이다.
3. **Ablation Study:**
    - 스타일 증강만 적용했을 때보다 콘텐츠 증강을 적용했을 때 성능 향상 폭이 더 컸으며(DSC 기준 7% 이상 상승), 두 가지를 모두 적용했을 때 최적의 성능을 얻었다.
    - 변형할 콘텐츠 맵의 개수 $t$에 대해 실험한 결과, $t=16$일 때 가장 좋은 성능을 보였다. $t=32$와 같이 너무 많은 맵을 변형하면 해부학적 구조가 과도하게 왜곡되어 레이블 일관성이 깨지는 문제가 발생했다.

## 🧠 Insights & Discussion

본 연구의 강점은 이미지 분해라는 수학적 근거(SVD)를 통해 도메인 시프트를 '스타일'과 '콘텐츠'라는 두 가지 축으로 명확히 정의하고, 이를 각각 제어 가능한 증강 기법으로 연결했다는 점이다. 특히, 기존의 스타일 중심 증강 기법들이 해결하지 못한 '콘텐츠 시프트'의 중요성을 실험적으로 입증한 점이 돋보인다.

다만, 콘텐츠 증강 시 $t$ 값의 설정에 따라 이미지의 시맨틱 정보가 훼손될 수 있다는 점은 중요한 한계이자 주의점이다. 이는 콘텐츠 증강이 단순한 스타일 변환보다 더 위험하며, 정밀한 하이퍼파라미터 튜닝이 필요함을 시사한다. 또한, SVD 연산 비용이 이미지 크기에 따라 증가할 수 있으나, 학습 단계에서만 수행되는 증강 과정이므로 추론 속도에는 영향을 주지 않는다는 실용적인 이점이 있다.

## 📌 TL;DR

본 논문은 SVD를 이용하여 의료 영상을 스타일 코드와 콘텐츠 맵으로 분해하고, 두 성분을 각각 변형하여 도메인 일반화 성능을 높이는 **StyCona** 증강 알고리즘을 제안한다. 이 방법은 추가 파라미터 없이 기존 모델에 적용 가능하며, 심장 MRI 및 안저 영상 분할 작업에서 기존 SOTA 방법론들을 능가하는 성능을 보였다. 특히 스타일뿐만 아니라 해부학적 구조(콘텐츠)의 변형이 의료 영상의 도메인 일반화에 필수적임을 입증하였다.
