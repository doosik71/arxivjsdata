# Semantic-Promoted Debiasing and Background Disambiguation for Zero-Shot Instance Segmentation

Shuting He, Henghui Ding, Wei Jiang (2023)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Instance Segmentation (ZSIS), 특히 학습 단계에서 보지 못한 카테고리의 객체를 탐지하고 분할해야 하는 Generalized ZSIS (GZSIS) 설정에서 발생하는 두 가지 핵심 문제를 해결하고자 한다.

첫째는 **Seen Categories에 대한 편향(Bias issue)**이다. 모델이 학습 데이터에 존재하는 Seen categories로만 최적화되었기 때문에, 추론 단계에서 Unseen category의 객체를 보았을 때 이를 Seen category 중 하나로 오분류하는 경향이 강하게 나타난다.

둘째는 **배경과의 모호성(Background ambiguation)**이다. 인스턴스 분할 모델의 학습 과정에서 학습 카테고리에 속하지 않는 모든 영역은 배경(Background)으로 처리된다. 이로 인해 모델은 한 번도 본 적 없는 Unseen objects를 단순히 배경의 일부로 인식하여 누락시키는 문제가 발생한다.

결과적으로 본 연구의 목표는 Unseen objects를 배경 및 Seen categories의 지배력으로부터 구출하여, Zero-shot 설정에서도 정밀한 인스턴스 분할 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문은 $D^2Zero$라는 프레임워크를 제안하며, 다음과 같은 중심 아이디어를 통해 앞서 언급한 문제들을 해결한다.

1. **Semantic-Promoted Debiasing**: Seen-Unseen 클래스 간의 시맨틱 관계를 활용하여 Visual feature 학습 단계부터 Unseen category의 정보를 주입하고, Transformer 기반의 **Input-conditional classifier**를 통해 입력 이미지에 따라 동적으로 변화하는 클래스 중심(Class center)을 생성함으로써 편향 문제를 완화한다.
2. **Background Disambiguation**: 고정된 배경 벡터를 사용하는 대신, 모델이 제안한 모든 마스크를 활용해 입력 이미지마다 최적화된 **Image-adaptive background representation**을 생성함으로써 Unseen object와 배경 사이의 모호성을 해소한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 소개하고 한계점을 지적한다.

- **Zero-Shot Learning (ZSL/GZSL)**: 시각적 특징과 시맨틱 임베딩 간의 매핑을 학습하거나 가짜 특징을 생성하는 방식이 사용된다. GZSL에서는 Seen 클래스로의 강한 편향이 주요 문제로 지적된다.
- **Zero-Shot Instance Segmentation (ZSIS)**: 기존의 ZSI [70]는 Seen category로 탐지된 인스턴스를 복제하여 Unseen group 내에서 다시 라벨링하는 '공유(Sharing)' 전략을 사용한다. 그러나 이는 하나의 인스턴스에 두 개의 라벨을 부여하게 되어 많은 False Positive를 생성하는 한계가 있다.
- **Zero-Shot Semantic Segmentation (ZSSS)**: 임베딩 기반 방식과 생성 기반 방식이 존재한다. 생성 기반 방식은 Seen category의 지식을 잊어버리는 문제(Forgetting)가 있으며, 새로운 클래스가 추가될 때마다 분류기를 재구성해야 하므로 실용성이 떨어진다.
- **Language-driven Segmentation**: 텍스트 정보를 활용하지만, 학습 과정에서 Unseen 클래스의 정보가 암시적으로 포함되는 정보 누출(Information leakage) 문제가 있어 엄격한 Zero-shot 설정과는 차이가 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

$D^2Zero$는 ResNet-50을 Backbone으로 사용하며, Mask2Former의 패러다임을 따른다. Mask2Former는 픽셀 분류를 마스크 분류로 변환하여 클래스에 구애받지 않는(Class-agnostic) 마스크 예측 $\{M^n\}$과 제안 임베딩 $\{x^n\}$을 생성한다.

### 2. Semantic-Promoted Visual Feature Debiasing

편향 문제를 해결하기 위해 특징 추출기(Feature Extractor)와 분류기(Classifier) 두 가지 측면에서 접근한다.

#### A. Unseen-Constrained Visual Feature Learning

학습 데이터에 없는 Unseen 클래스의 정보를 학습에 활용하기 위해 시맨틱 유사도 기반의 pseudo label을 도입한다.
먼저, Seen 클래스 $a^s_i$와 Unseen 클래스 $a^u_j$ 사이의 상관 계수 $e_{i,j}$를 다음과 같이 계산한다.
$$e_{i,j} = \frac{\exp(\langle a^s_i, a^u_j \rangle / \tau)}{\sum_{k=1}^{N_u} \exp(\langle a^s_i, a^u_k \rangle / \tau)}$$
여기서 $\langle \cdot, \cdot \rangle$은 코사인 유사도이며, $\tau$는 temperature 파라미터이다. 이후 Gumbel-Softmax trick을 사용하여 이 소프트 확률값을 이산적인 pseudo unseen label $\dot{e}_i \in \{0,1\}^{N_u}$로 변환한다.

분류 손실 함수는 Seen CE loss($L^s$)와 Unseen CE loss($L^u$)의 합으로 구성된다.
$$L = L^s + \lambda L^u$$
이때 $L^u$는 Foreground 객체에 대해 위에서 생성한 pseudo unseen label을 정답으로 하여 학습함으로써, 특징 추출기가 Unseen 클래스를 구분할 수 있는 능력을 갖추도록 강제한다.

#### B. Input-Conditional Classifier

고정된 시맨틱 임베딩을 분류기로 사용하면 특징들이 고정된 중심점으로 군집화되어 편향이 심화된다. 이를 해결하기 위해 Transformer 기반의 동적 분류기를 설계한다.

- **Query ($Q$)**: 시맨틱 임베딩 $\hat{A}$
- **Key ($K$) & Value ($V$)**: 이미지에서 추출된 제안 임베딩 $X = [x_1, x_2, \dots, x_{N_p}]$
- **작동 방식**: Multi-head Attention (MHA)를 통해 이미지 특성에 맞게 투영된 시각적 임베딩 $\ddot{A}$를 생성한다.
$$\ddot{A} = \text{MHA}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
이렇게 생성된 $\ddot{A}$는 입력 이미지에 따라 동적으로 변화하는 클래스 중심으로 작용하여, 시각-시맨틱 도메인 간의 간극을 줄이고 편향 문제를 완화한다.

### 3. Image-Adaptive Background Disambiguation

Unseen object가 배경으로 오인되는 문제를 해결하기 위해 입력 이미지에 적응적인 배경 프로토타입을 생성한다.
먼저, 모델이 예측한 모든 이진 마스크 $\{M^n\}$의 최대값을 취해 전경 영역 $M^f$를 정의하고, 그 반전 영역을 배경 마스크 $M^b$로 정의한다.
$$M^f(x,y) = \max(M^0(x,y), \dots, M^{N_p}(x,y))$$
$$M^b = 1 - M^f$$
이 $M^b$를 사용하여 시각 특징 맵 $F$에 대해 Mask Average Pooling (MAP)을 수행하여 배경 프로토타입 $p^b$를 얻는다.
$$p^b = \frac{\sum_{(x,y)} M^b(x,y) F(x,y)}{\sum_{(x,y)} M^b(x,y)}$$
이 $p^b$를 분류기의 배경 클래스 중심으로 사용하여, 이미지마다 다른 복잡한 배경 특성을 정확하게 반영한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MS-COCO 2014 (80개 클래스). 48/17 split(Seen 48, Unseen 17)과 65/15 split(Seen 65, Unseen 15) 사용.
- **평가 지표**: Recall@100 (IoU 0.4, 0.5, 0.6), mAP (IoU 0.5), 그리고 Seen과 Unseen의 균형을 측정하는 Harmonic Mean (HM) 사용.
- **텍스트 인코더**: CLIP 및 word2vec 임베딩 사용.

### 2. 주요 결과

- **성능 향상**: GZSIS 설정에서 ZSI [70] 대비 비약적인 성능 향상을 보였다. 특히 48/17 split에서 **HM-mAP 기준 16.86%의 향상**을 기록하였다.
- **구성 요소 분석**:
  - Input-conditional classifier 도입 시 Unseen mAP가 크게 상승하며, t-SNE 시각화를 통해 시각-텍스트 임베딩 간의 정렬(Alignment)이 개선됨을 확인하였다.
  - Image-adaptive background 방식이 고정된 배경 벡터나 워드 임베딩 방식보다 HM-mAP 및 HM-Recall 측면에서 우수함을 입증하였다.
  - Unseen CE loss($L^u$) 도입 시 baseline 대비 HM-mAP가 6.05% 상승하여 특징 추출기의 일반화 능력이 개선됨을 확인하였다.
- **일반화 능력**: COCO에서 학습된 모델을 ADE20k 데이터셋으로 전이(Transfer)했을 때도 ZSI보다 훨씬 뛰어난 성능을 보였으며, 이는 분류기 파라미터가 데이터셋 클래스 수에 종속되지 않는 구조 덕분이다.
- **효율성**: $D^2Zero$는 ZSI보다 파라미터 수(45.73M vs 69.6M)와 연산량(227.7G vs 569.3G FLOPs) 면에서 훨씬 효율적이다.

## 🧠 Insights & Discussion

본 논문은 GZSIS의 고질적인 문제인 '편향'과 '배경 모호성'을 구조적, 학습적 관점에서 매우 효과적으로 해결하였다.

특히 인상적인 점은 **분류기를 고정된 벡터의 내적으로 처리하지 않고, Transformer를 통해 입력 이미지의 컨텍스트를 반영한 동적 프로토타입으로 변환**했다는 점이다. 이는 시각적 특징이 몇 개의 고정된 중심점으로 붕괴(Collapse)되는 것을 방지하여 Unseen 클래스에 대한 판별력을 높이는 결정적인 역할을 했다고 판단된다.

또한, 배경을 단순히 학습 가능한 하나의 파라미터로 보는 대신, **추론 시점에 제안된 마스크들의 여집합을 통해 실시간으로 추출**하는 방식은 DETR 계열 모델의 클래스-불가지론적(Class-agnostic) 제안 능력을 극대화하여 활용한 영리한 접근이다.

다만, 본 논문에서 제안한 Pseudo label 생성 방식이 시맨틱 임베딩의 유사도에 전적으로 의존하고 있어, 시맨틱 공간 자체가 불완전하거나 클래스 간 유사도가 낮은 경우 성능 저하가 있을 가능성이 있다.

## 📌 TL;DR

본 연구는 Generalized Zero-Shot Instance Segmentation의 핵심 난제인 **Seen 클래스 편향**과 **배경-Unseen 객체 간 혼동**을 해결하기 위해 $D^2Zero$를 제안한다. 시맨틱 관계를 이용한 **Unseen-constrained 학습**, Transformer 기반의 **입력 조건부 분류기**, 그리고 이미지 기반의 **적응형 배경 표현**을 통해 기존 SOTA(ZSI) 대비 HM-mAP를 최대 16.86% 향상시켰으며, 연산 효율성까지 확보하였다. 이 연구는 향후 Open-vocabulary segmentation 및 Zero-shot 객체 탐지 연구에 중요한 기준점이 될 것으로 보인다.
