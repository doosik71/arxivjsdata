# 5th Place Solution for YouTube-VOS Challenge 2022: Video Object Segmentation

Wangwang Yang, Jinming Su, Yiting Duan, Tingyi Guo and Junfeng Luo (2022)

## 🧩 Problem to Solve

본 논문은 Semi-supervised Video Object Segmentation (VOS) 과제를 해결하기 위한 방법론을 제시한다. Semi-supervised VOS는 비디오의 첫 번째 프레임에서 주어진 객체 마스크(object mask)를 바탕으로 전체 비디오 시퀀스에서 해당 객체 인스턴스를 분할하는 것을 목표로 한다. 

연구진은 기존 VOS 모델들이 직면한 다음과 같은 세 가지 핵심 문제를 해결하고자 하였다:
1. **유사 객체 간의 혼동 (Similar objects confusion):** 외형이 비슷한 서로 다른 객체들이 존재할 때, 프레임 간 추적 과정에서 이를 동일한 객체로 잘못 매칭하는 문제가 발생한다.
2. **소형 객체 탐지 어려움 (Tiny objects detection):** 객체의 크기가 매우 작거나 프레임에 따라 크기 변화가 심할 경우, 정확한 탐지와 추적이 어렵다.
3. **시나리오 및 시맨틱의 다양성 (Diversity of semantics and scenes):** 학습 데이터셋에 포함되지 않은 새로운 장면이나 객체가 등장할 때 모델의 일반화 성능이 저하되는 문제가 있다.

결과적으로 본 논문의 목표는 데이터 보강, 모델 구조 개선, 그리고 정교한 후처리를 통해 YouTube-VOS 챌린지에서 최상위 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 단일 모델의 개선에 그치지 않고, **데이터-모델-앙상블-후처리**로 이어지는 전체 파이프라인을 최적화하는 통합 솔루션을 구축하는 것이다.

- **데이터 측면:** YouTube-VOS 데이터셋의 한계를 극복하기 위해 다양한 정적 이미지 데이터셋과 최신 비디오 분할 데이터셋을 추가하여 모델의 시맨틱 이해도와 강건성을 높였다.
- **모델 측면:** 서로 다른 특성을 가진 세 가지 기본 네트워크(AOT, STCN, FAMNet)를 채택하고, 특히 AOT의 핵심 모듈인 LSTT(Long Short-Term Transformer) 블록을 개선하여 객체 매칭 능력을 향상시켰다.
- **전략적 최적화:** 학습 후반부에 강한 데이터 증강(strong augmentation)을 제거하여 테스트 데이터 분포와의 간극을 줄였으며, 모델 앙상블과 정교한 후처리(BPR, Crop-then-zoom)를 통해 경계선 정밀도와 소형 객체 탐지력을 높였다.

## 📎 Related Works

논문에서는 VOS 분야의 주요 기존 연구들로 STM, STCN, AOT를 언급한다.

- **STM (Space-Time Memory network):** 과거 프레임의 마스크 정보를 외부 메모리로 저장하고, 현재 프레임(query)과 픽셀 수준에서 밀집 매칭(dense matching)을 수행하여 외형 변화나 폐쇄(occlusion) 문제를 처리한다.
- **STCN (Space-Time Correspondence Network):** 직접적인 이미지-이미지 대응 관계를 활용하여 STM보다 효율적이고 강건한 유사도 측정 방식을 제안하였다.
- **AOT (Associating Objects with Transformers):** 식별 메커니즘(identification mechanism)을 통해 여러 타겟을 고차원 임베딩 공간에 매핑함으로써, 다중 객체 시나리오에서도 효율적인 매칭과 분할 디코딩을 수행한다.

연구진은 이러한 기존 방법들이 전반적인 성능을 높였음에도 불구하고, 여전히 **유사 객체, 소형 객체, 복잡한 배경**에서의 성능 저하라는 한계가 있음을 지적하며 본인들의 솔루션을 제안한다.

## 🛠️ Methodology

### 1. 데이터 보강 전략 (Data Matters)
학습 과정은 2단계 전략을 따른다.
- **1단계 (Pre-training):** COCO, ECSSD, MSRA10K, PASCAL-S, PASCAL-VOC 등 대규모 정적 이미지 데이터셋을 사용하여 기초적인 시맨틱 학습을 수행한다. 이를 통해 모델이 픽셀 수준의 시공간 특징 매칭을 위한 강건한 특징 임베딩을 추출할 수 있게 한다.
- **2단계 (Main Training):** YouTube-VOS 외에도 YouTubeVIS, OVIS, VSPW와 같은 최신 비디오 데이터셋을 추가하여 학습시킨다. 특히 OVIS는 폐쇄 시나리오가 많고, VSPW는 고해상도 밀집 주석을 제공하므로 모델의 일반화 성능과 폐쇄 대응 능력을 크게 향상시킨다.

### 2. 기본 모델 및 LSTT Block V2 개선
연구진은 AOT, FAMNet, STCN 세 가지 아키텍처를 베이스라인으로 사용하였다. 특히 AOT의 핵심인 LSTT 블록을 다음과 같이 개선하여 **LSTT Block V2**를 제안하였다.

기존의 일반적인 Attention 기반 매칭 수식은 다음과 같다:
$$\text{Att}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{C}}\right)V$$

AOT에서는 타겟 식별 임베딩 $E$를 Value 부분에 결합하여 사용하였다:
$$\text{Att}(Q,K,V+E)$$

개선된 **LSTT Block V2**는 다음과 같은 수식을 사용한다:
$$\text{Att}(Q, K \cdot \sigma(W^G_l E), V + W^{ID}_l E)$$

이 개선 사항의 핵심은 두 가지이다:
1. **Value 부분:** 임베딩 $E$를 각 LSTT 레이어마다 서로 다른 가중치를 가진 선형 층 $W^{ID}_l$에 통과시켜 모델의 자유도를 높였다.
2. **Key 부분:** $E$를 이용하여 단일 채널 맵을 생성하고 이를 Key 임베딩 $K$에 곱해줌으로써, 매칭 과정 자체에 타겟 정보가 직접적으로 반영되도록 하였다.

### 3. 학습 및 추론 트릭
- **강한 증강 제거 (Turning off strong augmentation):** 학습 마지막 몇 에포크 동안 random cropping을 제외한 나머지 데이터 증강을 꺼서 테스트 데이터셋의 분포와 일치시켰다.
- **모델 앙상블 (Model Ensemble):** 서로 다른 백본(Swin, EfficientNet, ResNext)과 아키텍처를 가진 모델들의 소프트 예측 점수(soft prediction scores)를 단순히 평균 내는 오프라인 앙상블 방식을 사용하였다.
- **후처리 전략:**
    - **BPR (Boundary Patch Refinement):** 객체의 경계 영역을 정밀하게 다듬어 경계선 퀄리티를 높였다.
    - **Crop-then-zoom:** Tracker가 제공하는 박스 정보와 VOS 모델의 1차 예측 결과를 결합하여 소형 객체의 대략적인 위치를 파악한 뒤, 해당 영역을 crop 하여 고해상도로 다시 분할(secondary segmentation)함으로써 소형 객체 탐지력을 높였다.

## 📊 Results

### 실험 설정
- **데이터셋:** YouTube-VOS 2022 테스트 셋.
- **손실 함수:** Bootstrapped cross-entropy loss와 soft Jaccard loss의 합을 최소화한다.
- **최적화:** AdamW 옵티마이저와 EMA (Exponential Moving Average)를 사용하였다.
- **추론 기법:** 온라인 플립(flip) 및 멀티스케일 테스트(1.2$\times$, 1.3$\times$, 1.4$\times$ 480p 해상도)를 적용하고 앙상블하였다.

### 정량적 결과
본 솔루션은 YouTube-VOS 2022 테스트 셋에서 **종합 점수 86.1%**를 기록하며 **최종 5위**를 차지하였다. 특히, 분석 결과 'seen' 카테고리($J_{seen}, F_{seen}$)에서 매우 높은 성능을 보였다.

### 절제 실험 (Ablation Study)
YouTube-VOS 2019 검증 셋을 기준으로 AOT-R50 베이스라인에서 성능 향상 요인을 분석한 결과는 다음과 같다:
- **Baseline (AOT-L R50):** 85.3%
- **+ SwinB Backbone:** 85.5%
- **+ LSTT Block V2:** 85.7%
- **+ More real video data:** 86.2%
- **+ Turn off strong augmentation:** 86.6%

## 🧠 Insights & Discussion

본 논문은 특정 아키텍처의 혁신보다는 **현실적인 엔지니어링 최적화**를 통해 성능을 극대화한 사례이다. 

**강점:**
- 다양한 외부 비디오 데이터셋을 활용하여 VOS 모델의 고질적인 문제인 일반화 성능 부족을 효과적으로 해결하였다.
- 단순한 모델 앙상블뿐만 아니라, Tracker를 결합한 'Crop-then-zoom' 전략을 통해 VOS 모델이 구조적으로 해결하기 어려운 소형 객체 문제를 보완하였다.
- LSTT V2의 수정은 Attention 메커니즘에 타겟 식별 정보를 더 직접적으로 주입함으로써 매칭 정확도를 높이는 효과적인 접근이었다.

**한계 및 논의:**
- Top-k filtering을 통해 메모리의 노이즈를 제거하려는 시도가 있었으나, 모든 모델에서 항상 작동하지는 않았다고 명시되어 있다. 이는 메모리 뱅크의 정보 선택 전략이 데이터셋이나 모델 구조에 따라 매우 민감함을 시사한다.
- 제안된 솔루션의 상당 부분이 후처리 및 앙상블에 의존하고 있어, 실시간 추론(real-time inference) 환경에서는 계산 복잡도와 지연 시간(latency) 문제가 발생할 가능성이 크다.

## 📌 TL;DR

본 연구는 YouTube-VOS 2022 챌린지에서 5위를 기록한 솔루션으로, **대규모 비디오 데이터셋 보강 $\rightarrow$ LSTT V2 기반의 AOT 구조 개선 $\rightarrow$ 모델 앙상블 $\rightarrow$ 경계선 및 소형 객체 특화 후처리**로 이어지는 파이프라인을 제안한다. 특히 소형 객체를 위해 Tracker와 연동한 Crop-then-zoom 전략과 데이터 분포 일치를 위한 학습 트릭이 주요 기여 사항이며, 이는 향후 고정밀 VOS 시스템 구축 시 실질적인 가이드라인이 될 수 있다.