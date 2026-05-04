# Few-Shot Learning for Annotation-Efficient Nucleus Instance Segmentation

Yu Ming, Zihao Wu, Jie Yang, Danyi Li, Yuan Gao, Changxin Gao, Gui-Song Xia, Yuanqing Li, Li Liang and Jin-Gang Yu (2023)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 조직 병리 이미지(histopathology images)에서 핵 인스턴스 분할(nucleus instance segmentation)을 수행할 때 발생하는 **어노테이션의 극심한 노동 집약성과 전문가 의존성** 문제이다. 핵 분할은 종양 미세 환경의 정량적 특성 분석, 면역 조직 화학적 점수 측정, 예후 예측 등 다양한 계산 병리학 작업의 기초가 되는 매우 중요한 단계이다.

하지만 정밀한 인스턴스 라벨을 생성하는 과정은 비용이 매우 높기 때문에, 실제 환경에서는 학습에 사용할 수 있는 라벨링된 데이터가 매우 제한적이다. 따라서 본 논문의 목표는 **매우 적은 양의 어노테이션(few-shot)만으로도 타겟 데이터셋에서 높은 성능의 핵 인스턴스 분할을 달성하는 것**이며, 이를 위해 공개된 외부 데이터셋의 지식을 활용하는 효율적인 학습 패러다임을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 핵 인스턴스 분할 문제를 **Few-Shot Learning (FSL)** 관점에서 재정의하고, 이를 위해 **SGFSIS (Structurally-Guided Generalized Few-Shot Instance Segmentation)** 프레임워크를 설계한 것이다. 주요 기여 사항은 다음과 같다.

1.  **Generalized Few-Shot Instance Segmentation (GFSIS) 도입**: 기존의 FSL은 기본 클래스(base classes)와 새로운 클래스(novel classes)가 완전히 분리되어 있다고 가정한다. 그러나 실제 의료 데이터셋 간에는 클래스가 일부 중복될 가능성이 높으므로, 두 클래스 집합의 부분적 중복을 허용하는 GFSIS 개념을 도입하여 실용성을 높였다.
2.  **Structural Guidance Mechanism 설계**: 핵 분할의 고유한 어려움인 인접한 핵들의 접촉(touching) 문제와 세포의 이질성(heterogeneity)을 해결하기 위해, 서포트 셋(support set)을 활용하여 쿼리 이미지의 구조적 예측(전경 마스크, 경계, 중심점)을 변조하는 구조적 가이드 메커니즘을 제안하였다.
3.  **효율적인 학습 파이프라인**: 외부 데이터셋을 이용한 사전 학습(Pre-training) $\rightarrow$ 에피소드 샘플링 기반의 메타 학습(Meta-training) $\rightarrow$ 타겟 데이터셋에서의 미세 조정(Fine-tuning)으로 이어지는 3단계 학습 전략을 통해 어노테이션 효율성을 극대화하였다.

## 📎 Related Works

### 관련 연구 및 한계
- **완전 지도 학습(Fully-Supervised)**: U-Net 기반의 변형 모델들이나 경계(boundary) 및 거리 맵(distance map)을 예측하여 워터쉐드(watershed) 알고리즘으로 분할하는 방식(예: Hover-Net, StarDist)이 주를 이룬다. 하지만 이는 방대한 양의 정밀 라벨이 필요하다는 한계가 있다.
- **어노테이션 효율적 학습(Annotation-Efficient Learning)**: GAN을 이용한 데이터 증강, 준지도 학습(Semi-supervised learning), 도메인 적응(Domain Adaptation, DA) 등이 연구되었다. 특히 DA는 외부 데이터셋을 활용하지만, 타겟 데이터셋의 클래스가 외부 데이터셋과 완전히 동일해야 한다는 제약이 있어 실제 적용에 한계가 있다.
- **Few-Shot Instance Segmentation (FSIS)**: 컴퓨터 비전 분야에서 소수의 샘플로 새로운 객체를 분할하는 연구가 진행되었으나, 이를 핵 분할과 같은 의료 영상 영역에 적용한 사례는 거의 없었다.

### 기존 방식과의 차별점
본 연구는 단순히 외부 데이터를 가져오는 DA와 달리, **클래스의 중복 가능성을 열어둔 GFSIS**를 적용하였으며, 클래스 정보뿐만 아니라 **구조적 정보(Structural Guidance)**를 함께 전이함으로써 접촉된 핵들을 더 정확하게 분리할 수 있도록 설계되었다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
SGFSIS 프레임워크는 공유 인코더 $\phi$와 4개의 독립적인 디코더 $\theta$로 구성된 브랜치 구조를 가진다.
- **Classification Branch (CB)**: 각 픽셀의 클래스 라벨을 예측한다.
- **Foreground Branch (FB)**: 픽셀이 핵의 전경(foreground)일 확률을 예측한다.
- **Boundary Branch (BB)**: 핵의 경계(boundary)일 확률을 예측한다.
- **Centroid Branch (OB)**: 핵의 중심점(centroid)일 확률을 예측한다.

최종적으로 FB, BB, OB에서 얻은 구조적 정보를 결합하여 **Marker-Guided Watershed** 알고리즘을 적용해 인스턴스 마스크를 생성하고, 이를 CB의 클래스 정보와 결합하여 최종 결과를 도출한다.

### 주요 구성 요소 및 메커니즘

#### 1. Guided Classification Module (GCM)
새로운 클래스의 프로토타입 $p_n$을 서포트 셋의 특징 맵 $F^s_C$와 라벨 $L^{(n)}_C$를 이용해 다음과 같이 계산한다.
$$p_n \leftarrow \text{GAP}(F^s_C \odot L^{(n)}_C)$$
여기서 $\odot$는 마스킹 연산, $\text{GAP}$는 전역 평균 풀링(Global Average Pooling)이다. 외부 데이터셋에서 학습된 기본 클래스 프로토타입 $b_m$이 존재할 경우, 코사인 유사도 $\gamma_n = \cos(p_n, b_m)$를 이용하여 다음과 같이 프로토타입 등록(prototype registration)을 수행한다.
$$\tilde{p}_n = \begin{cases} \gamma_n p_n + (1-\gamma_n)b_m, & \text{if } \exists m, C^{\text{base}}_m = C_n \\ p_n, & \text{otherwise} \end{cases}$$
최종 분류 마스크 $M^{(n)}_C$는 쿼리 특징 맵 $F^q_C$와 보정된 프로토타입 $\tilde{p}_n$의 유사도를 통해 생성된다.

#### 2. Structural Guidance Modules (SGM)
구조적 브랜치(FB, BB, OB)는 클래스에 무관한(class-agnostic) 프로토타입을 사용한다. 경계 브랜치(SGM-B)를 예로 들면, 서포트 셋으로부터 경계 프로토타입 $u_B$를 추출한다.
$$u_B \leftarrow \text{GAP}(\text{conv}_{\omega_B}(F^s_B) \odot L_B)$$
쿼리의 경계 마스크 $M_B$는 다음과 같이 계산된다.
$$M_B \leftarrow \text{conv}_{\omega_B}(F^q_B) \otimes u_B + \text{conv}_{\phi_B}(F^s_B)$$
여기서 $\otimes$는 코사인 유사도 연산을 의미하며, 서포트 셋의 직접적인 특징과 프로토타입 기반의 전이 지식을 모두 활용한다.

#### 3. Marker-Guided Watershed
구조적 예측 결과를 이용해 인스턴스를 분리하는 절차는 다음과 같다.
1. 전경 마스크 $M_F$에서 경계 마스크 $M_B$를 형태학적 침식(erosion) 연산으로 제거하여 $\hat{M}^{(1)}_F$를 얻는다.
2. $\hat{M}^{(1)}_F$와 중심점 마스크 $M_O$에 대해 연결 성분 라벨링(Connected Component Labeling)을 수행한다.
3. $\hat{M}^{(1)}_F$의 성분이 $M_O$의 성분을 여러 개 포함하고 있다면, 이를 분리하여 정밀한 전경 마스크 $\hat{M}^{(2)}_F$를 생성한다.
4. 이 $\hat{M}^{(2)}_F$를 마커로 사용하여 워터쉐드 알고리즘을 수행함으로써 최종 인스턴스 마스크 $M_I$를 획득한다.

### 학습 절차
학습은 총 3단계로 진행된다.
1. **Pre-training**: 외부 데이터셋 $D_{\text{base}}$를 사용하여 인코더와 디코더를 완전 지도 학습 방식으로 사전 학습한다.
2. **Meta-training**: $D_{\text{base}}$에서 서포트 셋과 쿼리를 샘플링하여 에피소드 단위로 학습함으로써, 소수 샘플로부터 빠르게 적응하는 능력을 학습한다.
3. **Fine-tuning**: 타겟 데이터셋의 매우 적은 라벨링 데이터 $S$를 사용하여 프로토타입을 업데이트하고 전체 파라미터를 미세 조정한다.

## 📊 Results

### 실험 설정
- **데이터셋**: ConSep, PanNuke, MoNuSAC, Lizard 등 4개의 공개 데이터셋을 사용하였다.
- **평가 지표**: mPQ (Multi-Class Panoptic Quality)를 주 지표로 사용하며, AJI (Aggregated Jaccard Index), $F_1\text{-novel}$, $F_1\text{-base}$를 함께 측정하였다.
- **비교 대상**: 
    - `FullSup-s`: 서포트 셋만으로 학습한 완전 지도 학습.
    - `TransFT`: 외부 데이터셋 학습 후 서포트 셋으로 미세 조정한 전이 학습.
    - `SemiSup`: MeanTeacher 알고리즘을 적용한 준지도 학습.
    - `FullSup`: 타겟 데이터셋 전체 라벨을 사용한 상한선.

### 정량적 결과
실험 결과, SGFSIS는 모든 데이터셋 설정에서 다른 베이스라인(`FullSup-s`, `TransFT`, `SemiSup`)보다 우수한 성능을 보였다. 특히 샘플 수($K$)가 적을 때 성능 향상 폭이 매우 컸으며, **전체 라벨의 5% 미만($K=50$)만 사용하고도 전체 라벨을 사용한 `FullSup`와 대등한 성능**을 달성하였다.

- **AJI의 큰 향상**: $F_1$ 스코어보다 AJI에서 더 큰 성능 향상이 관찰되었는데, 이는 클래스 구분보다 클래스 무관한 구조적 분할(boundary localization)이 데이터셋 간 전이가 더 쉽기 때문으로 분석된다.
- **GFSL의 효과**: $F_1\text{-novel}$보다 $F_1\text{-base}$에서 더 큰 이득이 있었으며, 이는 기본 클래스의 지식을 새로운 클래스 학습에 성공적으로 활용했음을 시사한다.

### 정성적 결과
시각화 결과, SGFSIS는 특히 **서로 붙어 있는 핵(touching instances)을 분리**하는 성능이 타 방법론에 비해 월등히 뛰어났으며, 새로운 클래스에 대한 분류 정확도 또한 높게 나타났다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 핵 분할이라는 특수 목적에 맞게 FSL을 성공적으로 변형하였다. 특히 단순한 특징 전이를 넘어, **구조적 가이드(Structural Guidance)**를 통해 의료 영상의 고유 문제인 '접촉된 객체 분리' 문제를 해결한 점이 매우 효과적이었다. 또한, 현실적인 클래스 중복 문제를 해결한 GFSIS 설정이 성능 향상에 기여하였다.

### 한계 및 미해결 질문
1. **외부 데이터셋 의존성**: 정의상 완전히 라벨링된 외부 데이터셋이 반드시 필요하다. 이러한 데이터가 없는 새로운 도메인에서는 적용이 어렵다.
2. **저대조도 이미지 문제**: 배경과 대비가 매우 낮은 핵의 경우 인식하지 못하고 누락시키는 경향이 관찰되었다.
3. **미사용 데이터**: 타겟 데이터셋 내의 방대한 **미라벨링 데이터(unlabeled data)**를 전혀 활용하지 않았다. 준지도 학습 기법을 결합한다면 추가적인 성능 향상이 가능할 것으로 보인다.

### 비판적 해석
저자들은 구조적 전이가 클래스 전이보다 쉽다고 주장하며 두 작업을 분리하여 해결할 것을 제안한다. 이는 매우 타당한 통찰이며, 향후 연구에서 클래스 분류 네트워크와 구조 예측 네트워크를 완전히 디커플링하여 각각 최적화하는 방향으로 발전시킬 수 있을 것이다.

## 📌 TL;DR

본 논문은 매우 적은 라벨만으로 핵 인스턴스 분할을 수행하기 위해 **SGFSIS**라는 Few-Shot Learning 프레임워크를 제안한다. 외부 데이터셋의 지식을 전이하는 GFSIS 구조와, 붙어 있는 핵들을 효과적으로 분리하기 위한 구조적 가이드 메커니즘(Structural Guidance)을 도입하였다. 실험 결과, **단 5%의 라벨만으로도 전체 라벨을 사용한 모델에 근접하는 성능**을 보였으며, 특히 인접한 핵의 분리 성능을 획기적으로 개선하였다. 이 연구는 라벨링 비용이 매우 높은 의료 영상 분할 분야에서 실질적인 해결책을 제시했다는 점에서 큰 의미가 있다.