# Instance-Specific Feature Propagation for Referring Segmentation

Chang Liu, Xudongong Jiang, Henghui Ding (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 **Referring Image Segmentation (RIS)**이다. RIS는 이미지와 함께 주어진 자연어 표현(natural language expression)을 분석하여, 해당 표현이 지칭하는 특정 대상(target instance)의 세그멘테이션 마스크를 생성하는 작업이다.

이 문제의 핵심적인 난제는 이미지 내의 여러 후보 대상들 사이의 상호작용을 모델링하는 것이다. 예를 들어, "가장 오른쪽에 있는 사람"이라는 쿼리가 주어졌을 때, 모델은 단순히 '사람'이라는 객체를 찾는 것을 넘어, 이미지 내 모든 '사람' 인스턴스들의 상대적 위치를 비교하여 어떤 대상이 가장 오른쪽에 있는지를 판단해야 한다.

기존 방법론은 크게 두 가지로 나뉜다. 첫째, **One-stage 방법**은 시각 정보와 언어 정보를 직접 융합하여 픽셀 단위의 분류를 수행한다. 이는 효율적이지만 인스턴스에 대한 명시적 인식(instance-agnostic)이 부족하여 복잡한 상대적 관계를 파악하는 능력이 떨어진다. 둘째, **Two-stage 방법**은 Mask R-CNN과 같은 모델로 인스턴스 제안(proposal)을 먼저 생성한 뒤 언어 특징과 매칭하여 대상을 선택한다. 하지만 이 방식은 언어 정보가 마스크 생성 단계에 영향을 주지 못하며, 제안된 인스턴스들의 품질에 전체 성능이 종속된다는 한계가 있다.

따라서 본 논문의 목표는 인스턴스 간의 상호작용을 직접 모델링하는 **Instance-aware** 특성과, 대상 식별 및 마스크 생성이 동시에 이루어지는 **Integrated one-stage** 구조를 결합한 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지를 격자(grid) 단위로 나누어 각 격자가 특정 인스턴스를 대표하게 하고, 이들 간의 특징을 전파(propagation)함으로써 대상의 상대적 위치와 관계를 파악하는 것이다.

주요 기여 사항은 다음과 같다.

1. 인스턴스 식별(identification)과 세그멘테이션(segmentation)을 동시에, 그리고 협력적으로 수행하는 새로운 RIS 프레임워크를 제안하였다.
2. 모든 인스턴스 간의 정보 교환을 통해 타겟을 정확히 찾아내기 위한 **Feature Propagation Module (FPM)**을 제안하였다.
3. 초기 단계의 거친(coarse) 마스크를 정교화하여 세부 디테일을 살리는 **Refinement Module**을 도입하였다.
4. RefCOCO 시리즈의 세 가지 데이터셋 모두에서 기존 SOTA(State-of-the-art) 성능을 경신하였다.

## 📎 Related Works

논문에서는 기존 RIS 접근 방식을 다음과 같이 분석한다.

- **One-stage methods**: 시각 및 언어 특징을 융합한 뒤 FCN(Fully Convolutional Network) 등을 통해 마스크를 생성한다. 최근에는 Cross-modal self-attention 등을 사용하여 장거리 의존성을 파악하려는 시도가 있었으나, 여전히 개별 인스턴스 간의 명시적인 비교 및 상호작용 모델링에는 한계가 있다.
- **Two-stage methods**: 인스턴스 분할 모델로 후보군을 먼저 뽑고 매칭하는 방식을 취한다. 이는 인스턴스 간 상호작용을 직접 모델링할 수 있다는 장점이 있지만, 언어 정보가 마스크 생성 과정에 관여하지 못하므로 후보 마스크의 품질이 낮으면 결과적으로 성능이 제한된다.

본 연구는 One-stage의 효율성과 Two-stage의 인스턴스 인식 능력을 동시에 확보하여, 언어 정보가 식별과 세그멘테이션 과정 모두에 유기적으로 작용하도록 설계함으로써 기존 방식들과 차별화를 두었다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 (Overall Pipeline)

제안된 프레임워크는 크게 **Backbone**, **Instance Extraction Module (IEM)**, **Feature Propagation Module (FPM)**, 그리고 **Refinement Module**로 구성된다.

### 2. 상세 구성 요소 및 역할

#### A. Backbone 및 특징 융합

- **Vision Backbone**: FPN(Feature Pyramid Network)을 사용하여 다양한 스케일의 시각 특징 $\text{F}_{vl}, \text{F}_{vm}, \text{F}_{vs}$를 추출한다.
- **Language Backbone**: GloVe 임베딩과 Bi-directional GRU를 통해 언어 특징을 추출하고, Self-attention을 통해 각 단어의 중요도를 가중합한 최종 언어 특징 $\text{F}_t$를 생성한다.
- **Feature Fusion**: 시각 특징과 언어 특징을 원소별 곱셈(element-wise multiplication)으로 융합한다. 픽셀 $(i, j)$에서의 융합 특징 $f_{i,j}$는 다음과 같이 정의된다.
$$f_{i,j} = (f_{i,j}^v W_v) * (F_t W_t)$$
여기서 $W_v, W_t$는 학습 가능한 파라미터이며, $*$는 원소별 곱셈을 의미한다. 융합된 특징은 이후 Identification Branch를 위한 하향 샘플링 경로($\text{F}_{ide}$)와 Segmentation Branch를 위한 상향 샘플링 경로($\text{F}_{seg}$)로 나뉜다.

#### B. Instance Extraction Module (IEM)

IEM은 두 개의 브랜치로 구성된다.

- **Identification Branch**: 이미지를 $S \times S$ 격자로 나누고, 각 격자의 중심에 위치한 인스턴스를 대표하는 **Instance-Specific Feature (ISF)** 맵 $\text{F}_{ins} \in \mathbb{R}^{S \times S \times C}$를 생성한다.
- **Segmentation Branch**: $\text{F}_{seg}$를 입력받아 각 격자에 대응하는 $S^2$개의 바이너리 마스크를 동시에 생성한다. 이때 격자 $(i, j)$에 대응하는 마스크는 채널 $c = i \times S + j$에 저장된다.

#### C. Feature Propagation Module (FPM)

인스턴스 간의 "비교" 관계를 모델링하기 위해 특징 전파 방식을 도입하였다.

- **전파 경로**: DR$\searrow$, DL$\swarrow$, UR$\nearrow$, UL$\nwarrow$ 네 가지 방향의 양방향 경로를 통해 모든 인스턴스가 서로의 정보를 교환하게 한다.
- **업데이트 방정식**: 특정 위치 $(i, j)$의 은닉 상태 $h_{i,j}$는 현재 픽셀의 특징 $v_{i,j}$와 이전 단계의 주변 픽셀(행, 열, 대각선)들의 정보를 합산하여 업데이트된다.
$$h_{i,j} = v_{i,j} + \alpha W_2 \sum_{m,n} h_{m,n}$$
- 여기서 $\alpha$는 망각률을 조절하는 상수이다. 최종적으로 모든 경로의 결과와 입력 ISF를 합산하고 $1 \times 1$ Convolution을 거쳐 $S \times S \times 1$ 크기의 **Identifying Map**을 생성하여 타겟 인스턴스가 위치한 격자의 확률을 계산한다.

#### D. Refinement Module

Segmentation Branch에서 생성된 마스크는 해상도가 낮아 거칠기 때문에, 원본 이미지와 선택된 거친 마스크를 결합하여 세부 디테일을 복원한다. 3개의 $3 \times 3$ Convolution 층과 업샘플링 층을 거쳐 정교해진 최종 마스크를 출력한다.

### 3. 학습 절차 및 손실 함수

전체 손실 함수는 다음과 같이 세 가지 손실의 가중 합으로 정의된다.
$$\mathcal{L} = w_{ide}l_{ide} + w_{seg}l_{seg} + w_{ref}l_{ref}$$

- **Identifying Loss ($l_{ide}$)**: 타겟 인스턴스의 중심이 포함된 격자와 그 주변 8개 격자를 양성(positive)으로 설정하고, Identifying Map에 대해 Binary Cross Entropy (BCE) 손실을 적용한다.
- **Segmentation Loss ($l_{seg}$)**: 식별 브랜치에서 양성으로 판별된 격자에 해당하는 마스크 채널에 대해서만 GT 마스크와 BCE 손실을 계산한다.
- **Refinement Loss ($l_{ref}$)**: 최종 정제된 마스크와 GT 마스크 간의 BCE 손실을 계산한다. 단, 식별 단계에서 선택된 마스크의 $\text{IoU}$가 임계값 $\theta$보다 높을 때만 학습에 반영하는 적응형 전략을 사용한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: RefCOCO, RefCOCO+, RefCOCOg 세 가지 벤치마크를 사용하였다.
- **백본**: Darknet53 및 GloVe 임베딩을 사용하였다.
- **평가 지표**: $\text{IoU}$ 및 $\text{Precision@X}$ (예측 $\text{IoU}$가 $X$ 이상인 이미지의 비율)를 사용하였다.

### 2. 정량적 결과

- **SOTA 달성**: 제안 방법은 세 데이터셋 모두에서 기존 모델들(MCN, CGAN 등)보다 $\text{IoU}$ 성능이 약 $1\%$ 이상 향상되었으며, 특히 RefCOCO+ testB 세트에서는 이전 SOTA인 CGAN보다 $2\%$ 이상 높은 성능을 보였다.
- **정밀도 분석**: $\text{Precision@0.6, 0.7, 0.8}$ 지표에서 가장 높은 점수를 기록하여, 타겟 식별 능력뿐만 아니라 생성된 마스크의 품질 또한 우수함을 입증하였다.

### 3. 절제 연구 (Ablation Study)

- **IEM의 효과**: Baseline(One-stage) 대비 $\text{IoU}$가 크게 상승하였으며, 특히 낮은 임계값의 Precision($\text{Pr@0.5}$)이 증가하여 인스턴스 인식 능력이 향상되었음을 확인하였다.
- **FPM의 효과**: FPM 추가 시 $\text{IoU}$가 추가로 상승하여, 인스턴스 간의 관계 추론이 타겟 식별에 기여함을 보였다.
- **RM의 효과**: 고임계값 Precision($\text{Pr@0.8, 0.9}$)에서 성능 향상이 뚜렷하게 나타나, 마스크의 세부 디테일 개선 효과를 증명하였다.

## 🧠 Insights & Discussion

### 1. 강점

본 모델은 특히 "두 번째로 오른쪽에 있는 사람"이나 "사과 위에 있는 레몬"과 같이 **상대적 위치와 관계**를 설명하는 쿼리에 매우 강한 면모를 보인다. 이는 FPM이 전역적으로 인스턴스 특징을 전파함으로써 인스턴스 간의 비교 분석을 가능하게 했기 때문이다. 또한, 마스크 생성 단계에 언어 정보를 주입함으로써 타겟 대상의 특징을 더 잘 부각시킬 수 있었다.

### 2. 한계 및 비판적 해석

- **모호한 표현에 대한 취약성**: 텍스트가 지나치게 단순하거나(예: 단순히 숫자 '8'만 제공), OCR 능력이 필요한 경우 식별에 실패하는 경향이 있다.
- **유사 외형 객체의 중첩**: 색상과 모양이 매우 유사한 객체들이 서로 겹쳐 있을 때, 이를 개별 인스턴스로 정확히 분리하여 식별하는 데 어려움을 겪는다.
- **분석**: 성능 갭 분석 결과, 식별 브랜치보다 세그멘테이션 브랜치를 GT로 대체했을 때 성능 상승 폭이 더 컸다. 이는 현재의 세그멘테이션 브랜치 구조가 매우 단순하여, 향후 이 부분의 아키텍처를 고도화한다면 더 큰 성능 향상을 기대할 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 RIS 작업에서 One-stage의 효율성과 Two-stage의 인스턴스 인식 능력을 결합한 새로운 프레임워크를 제안한다. 이미지를 격자로 나누어 인스턴스별 특징(ISF)을 추출하고, 이를 **Feature Propagation Module (FPM)**을 통해 전역적으로 교환함으로써 복잡한 상대적 관계를 파악하여 타겟을 식별한다. 또한 **Refinement Module**을 통해 마스크의 정밀도를 높였다. 결과적으로 RefCOCO 시리즈 데이터셋에서 SOTA 성능을 달성하였으며, 특히 관계 기반의 타겟 식별 능력에서 탁월한 성능을 보였다. 이 연구는 향후 복잡한 장면 이해가 필요한 시각-언어 통합 모델 연구에 중요한 기초가 될 것으로 보인다.
