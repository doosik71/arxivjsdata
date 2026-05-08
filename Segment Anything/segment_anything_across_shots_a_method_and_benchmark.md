# Segment Anything Across Shots: A Method and Benchmark

Hengrui Hu, Kaining Ying, Henghui Ding (2026)

## 🧩 Problem to Solve

본 논문은 **Multi-shot Semi-supervised Video Object Segmentation (MVOS)** 문제를 해결하고자 한다. 일반적인 Video Object Segmentation (VOS)는 단일 샷(single-shot) 비디오를 가정하여 첫 프레임의 마스크를 기반으로 객체를 추적하지만, 실제 인터넷 콘텐츠나 편집된 영상은 여러 개의 샷으로 구성된 Multi-shot 비디오가 주를 이룬다.

Multi-shot 비디오에서는 샷 전환(shot transition)이 발생할 때 객체의 외관(appearance), 공간적 위치(spatial location), 그리고 배경(background)이 급격하게 변하는 불연속성이 나타난다. 기존의 SOTA VOS 모델들(XMem, DEVA, Cutie, SAM2 등)은 이러한 샷 전환 상황에서 성능이 급격히 저하되는 한계를 보인다. 또한, Multi-shot 비디오에 대한 고품질의 세그멘테이션 어노테이션 데이터가 매우 부족하다는 데이터 희소성(data sparsity) 문제가 존재한다. 따라서 본 논문의 목표는 샷 전환에 강건한 세그멘테이션 모델을 구축하고, 이를 평가하기 위한 새로운 벤치마크를 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 크게 세 가지로 요약된다.

1. **Transition Mimicking Data Augmentation (TMA) 전략**: 단일 샷(single-shot) 데이터셋을 활용하여 다양한 샷 전환 시나리오를 모사(mimicking)함으로써, Multi-shot 전용 어노테이션 데이터 없이도 모델의 교차 샷(cross-shot) 일반화 능력을 향상시키는 데이터 증강 전략을 제안한다.
2. **SAAS (Segment Anything Across Shots) 모델**: SAM2를 기반으로 하며, 샷 전환을 실시간으로 감지하는 **Transition Detection Module (TDM)**, 전환 상태를 이해하고 메모리를 정제하는 **Transition Comprehension Module (TCH)**, 그리고 객체의 세부 특징을 저장하는 **Local Memory Bank ($B_{local}$)**를 도입하여 Multi-shot 환경에서의 강건성을 확보하였다.
3. **Cut-VOS 벤치마크**: 기존의 YouMVOS보다 더 높은 샷 전환 빈도와 더 다양한 객체 카테고리, 그리고 정밀한 마스크 어노테이션을 포함하는 새로운 MVOS 벤치마크를 구축하여 공개함으로써 향후 연구의 기반을 마련하였다.

## 📎 Related Works

### 기존 VOS 연구 및 한계

VOS 연구는 크게 Fine-tuning 기반, Matching 기반, Propagation 기반 방법론으로 발전해 왔으며, 최근에는 XMem이나 Cutie와 같이 메모리 뱅크(Memory Bank)를 통해 과거 프레임의 정보를 효율적으로 저장하고 활용하는 방식이 주류를 이루고 있다. 특히 SAM2는 대규모 데이터 학습과 강력한 메모리 아키텍처를 통해 비약적인 성능 향상을 이루었다. 그러나 이러한 방법론들은 대부분 단일 샷 비디오에 최적화되어 있어, 샷 전환으로 인한 급격한 시각적 변화가 발생하는 상황에서는 추적 성능이 현저히 떨어진다.

### Multi-shot 비디오 이해 연구

기존의 Multi-shot 관련 연구들은 주로 샷 경계 검출(Shot Boundary Detection)이나 비디오 캡셔닝(Video Captioning), 이벤트 로컬라이제이션(Event Localization) 등에 집중되어 있었다. 하지만 픽셀 수준의 정밀한 인스턴스 세그멘테이션(Pixel-level instance segmentation)을 Multi-shot 환경에서 수행하려는 시도는 부족했다. 본 논문은 이러한 공백을 메우기 위해 픽셀 수준의 MVOS 작업을 정의하고 해결책을 제시한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구조

SAAS는 SAM2의 이미지 인코더를 사용하여 다단계 시각적 특징 $\{F_{li}^t\}$를 추출하며, 매 타임스텝 $t$마다 다음과 같은 흐름으로 작동한다.

1. **TDM**이 현재 프레임에서 샷 전환 발생 여부를 감지한다.
2. 전환이 감지되지 않은 경우, 표준 SAM2 파이프라인을 따라 세그멘테이션을 수행한다.
3. 전환이 감지된 경우, **TCH**가 인접 프레임과 배경 컨텍스트를 통해 전환 상태를 이해하고 메모리 토큰을 정제한다.
4. **$B_{local}$**에서 저장된 세부 특징과 정제된 메모리를 결합하여 최종 마스크 $\hat{M}^t$를 예측한다.

### 2. Transition Mimicking Augmentation (TMA)

데이터 희소성 문제를 해결하기 위해, 단일 샷 데이터에서 다음과 같은 패턴을 시뮬레이션하여 Multi-shot 샘플을 합성한다.

- **강한 변환(Strong Transforms)**: 수평 뒤집기, 랜덤 스케일링, 아핀 변환 등을 통해 클로즈업(Close-up)이나 원거리 뷰(Distant view) 전환을 모사한다.
- **단일/다중 전환(Single/Multiple Transitions)**: 동일 비디오 내의 서로 다른 세그먼트로 점프하거나, 완전히 다른 비디오로 전환했다가 다시 돌아오는(Cut away $\rightarrow$ Cut in) 상황을 생성한다.
- **점진적 이동(Gradual Translations)**: 객체를 복제하고 점진적으로 이동시켜 장면 변경(Scene change) 및 지연된 컷인(Delayed cut in)을 구현한다.

### 3. Transition Detection & Comprehension

#### Transition Detection Module (TDM)

Dilated Convolution Pyramid를 사용하여 현재 프레임 $I^t$와 이전 $N$개 프레임 간의 관계를 분석하고, 전환 확률 $\hat{p}_{i,tr}$을 예측한다.
$$\hat{p}_{i,tr} = \text{Sigmoid}(F^{TDM}(F^t, F^{t-i}_{i=1,2,...,N}))$$
$\hat{p}_{i,tr}$이 임계값 $\tau_{tr}$보다 크면 전환 전략을 채택한다.

#### Transition Comprehension Module (TCH)

TCH는 $\text{B}_{scene}$에서 배경 정보를 읽어와 현재 특징 $F_{l3}^t$에 통합하여 $F_{l3}^{\prime t}$를 생성한다. 이후 학습 가능한 벡터 $Q_{init}$이 Attention 레이어를 통해 이전/현재 프레임과 상호작용하며 전환 상태 표현 $Q_i^n$을 생성한다.
$$Q_i^n = \text{Attn}(\text{Attn}(Q_i^{n-1}, F_{l3}^{\prime t}), F_{l3}^{t-1})$$
모델의 학습을 돕기 위해 두 가지 보조 목표를 도입한다.

- **Presence Prediction**: 객체가 다음 프레임에 존재하는지 예측 ($\mathcal{L}_{exis}$, BCE Loss 사용)
- **Bounding Box Regression**: 이전 bbox와 $Q_i$를 이용하여 전환 후의 bbox를 예측 ($\mathcal{L}_{box}$, MCE Loss 사용)

### 4. Local Memory Bank ($B_{local}$)

객체의 세부 특징(옷의 무늬, 차량의 표식 등)을 캡처하기 위해 도입되었다.

- 조건 프레임의 특징 맵에 **Minimum Spanning Tree (MST)**를 구축하여 시맨틱 클러스터링과 공간 구조를 보존하며 객체를 여러 하위 영역으로 분할한다.
- 각 분할 영역의 중심점을 긍정(positive) 포인트 프롬프트로 사용하여 고해상도 세부 특징을 추출하고 이를 $B_{local}$에 저장한다. 전환 발생 시 이 정보들이 세그멘테이션 가이드로 사용된다.

## 📊 Results

### 실험 설정

- **데이터셋**: YouMVOS, Cut-VOS (본 논문 제안).
- **지표**: $\text{J\&F}$ (영역 유사도 및 윤곽선 정확도)와 교차 샷 추적 능력을 측정하는 $\text{J}_t$를 사용한다. $\text{J}_t$는 샷 전환 직후 프레임과 객체가 다시 나타나는 프레임의 IoU 평균으로 계산된다.
- **비교 대상**: XMem, DEVA, Cutie, SAM2-B+, SAM2-L 등.

### 정량적 결과 (Table 2 참조)

- **YouMVOS**: SAAS-L은 $\text{J\&F}$ 74.4%, $\text{J}_t$ 74.2%를 기록하며 기존 모델들을 압도한다.
- **Cut-VOS**: 훨씬 어려운 벤치마크임에도 불구하고, SAAS-L은 $\text{J\&F}$ 62.0%, $\text{J}_t$ 54.0%를 달성하였다. 이는 baseline인 SAM2-L($\text{J\&F}$ 59.4%, $\text{J}_t$ 50.7%) 대비 뚜렷한 향상이다.
- **TMA의 효과**: Cutie 모델에 TMA 전략만 적용했을 때 ($\text{Cutie+TMA}$) 성능이 향상됨을 확인하여, 제안한 데이터 증강 전략의 일반적인 유효성을 입증하였다.

### 정성적 결과 및 분석

- **지연된 컷인(Delayed cut in)** 및 **급격한 위치 이동** 상황에서 SAM2는 타겟을 놓치거나 유사한 외관의 다른 객체로 오인하는 반면, SAAS는 성공적으로 추적하였다.
- **복잡한 관계의 군집 장면**에서도 $B_{local}$과 씬 이해 능력을 통해 타겟 객체를 일관되게 분리해 내는 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 연구는 Multi-shot VOS라는 실무적으로 매우 중요하지만 데이터 부족으로 간과되었던 문제를 정의하고, 이를 해결하기 위한 데이터 증강 전략(TMA)과 모델 구조(SAAS)를 통합적으로 제시하였다. 특히, 단순히 메모리를 늘리는 것이 아니라 샷 전환이라는 특정 이벤트에 맞춘 '감지 $\rightarrow$ 이해 $\rightarrow$ 정제' 파이프라인을 구축한 점이 주효했다.

### 한계 및 미해결 과제

- **극단적인 외관 변화**: 옷을 갈아입거나 헤어스타일이 완전히 바뀌는 등 시각적 특징이 완전히 변하는 경우, 본 모델 역시 실패하는 경향이 있다.
- **추론 능력의 부재**: 현재의 SAAS는 여전히 시각적 특징 매칭(visual feature matching)에 의존하고 있다. 극단적인 뷰 전환(예: 갑작스러운 대시보드 확대) 상황에서는 인간과 같은 상식적 추론(commonsense reasoning)이나 촬영 의도에 대한 이해가 필요하며, 이는 향후 연구 과제로 남는다.

## 📌 TL;DR

본 논문은 샷 전환이 빈번한 Multi-shot 비디오에서 객체 세그멘테이션 성능을 높이기 위해, **단일 샷 데이터를 이용해 전환 상황을 모사하는 TMA 증강 전략**과 **전환 감지 및 이해 모듈을 갖춘 SAAS 모델**을 제안하였다. 또한, 현실적인 난이도를 반영한 **Cut-VOS 벤치마크**를 구축하여 공개하였다. 이 연구는 기존 VOS 모델들이 해결하지 못한 샷 불연속성 문제를 효과적으로 완화하였으며, 향후 실제 영상 편집 및 자율 주행 등 복잡한 영상 분석 시스템의 성능 향상에 기여할 가능성이 높다.
