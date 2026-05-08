# ASI-Seg: Audio-Driven Surgical Instrument Segmentation with Surgeon Intention Understanding

Zhen Chen, Zongming Zhang, Wenwu Guo, Xingjian Luo, Long Bai, Jinlin Wu, Hongliang Ren, Hongbin Liu (2024)

## 🧩 Problem to Solve

기존의 수술 도구 분할(Surgical Instrument Segmentation) 알고리즘들은 입력 이미지 내에서 미리 정의된 모든 카테고리의 도구를 직접적으로 탐지하고 분할하는 방식을 취한다. 하지만 실제 임상 환경에서 외과의는 수술 단계에 따라 서로 다른 도구에 집중하며, 특정 시점에는 특정 도구만을 필요로 하는 '외과의의 의도(Surgeon's Intention)'가 존재한다. 모든 도구를 한꺼번에 분할하는 기존 방식은 불필요한 도구들로 인한 시각적 방해를 초래할 수 있다.

최근 등장한 Segment Anything Model (SAM)은 프롬프트(Prompt)에 따라 객체를 분할하는 강력한 능력을 보여주었으나, 수술실 내에서 포인트나 바운딩 박스와 같은 수동 프롬프트를 입력하는 것은 수술 흐름을 방해하므로 비현실적이다. 따라서 본 논문은 수동 입력 없이 외과의의 오디오 명령을 통해 의도를 파악하고, 필요한 도구만을 정밀하게 분할할 수 있는 시스템을 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 외과의의 음성 명령을 기반으로 분할 의도를 해석하고, 이를 SAM의 프롬프트로 변환하여 원하는 도구만을 선택적으로 분할하는 **ASI-Seg** 프레임워크를 제안하는 것이다. 주요 기여 사항은 다음과 같다.

1. **의도 지향적 다중모달 융합(Intention-Oriented Multimodal Fusion):** 오디오 명령에서 분할 의도를 해석하고, 텍스트 설명과 시각적 정보를 결합하여 타겟 도구의 상세 특징을 추출한다.
2. **대조 학습 프롬프트 인코더(Contrastive Learning Prompt Encoder):** 필요한 도구의 특징과 불필요한 도구의 특징을 효과적으로 구분하기 위해 대조 학습을 도입하여 SAM의 마스크 디코더에 최적화된 프롬프트를 제공한다.
3. **오디오 기반의 실시간 제어 가능성:** 수동 어노테이션 없이 음성만으로 특정 도구를 분할함으로써 외과의의 인지 부하를 줄이고 수술 워크플로우를 개선한다.

## 📎 Related Works

### 기존 수술 도구 분할 연구

TernausNet, ISINet, Dual-MF, S3Net 등의 연구들은 네트워크 구조 개선, 모션 플로우 활용, 카테고리 식별 능력 향상 등을 통해 분할 정확도를 높여왔다. 그러나 이들은 공통적으로 이미지 내의 모든 사전 정의된 도구를 분할하며, 사용자의 특정 의도를 반영하는 기능이 결여되어 있다.

### 의료 영상 분야의 SAM 활용

SAM을 의료 영상에 적용하려는 시도(MedSAM, SurgicalSAM 등)가 이어지고 있다. 특히 SurgicalSAM은 클래스 프로토타입을 통해 카테고리 정보를 가이드로 사용한다. 하지만 대부분의 의료용 SAM은 여전히 추론 단계에서 포인트나 박스 같은 수동 프롬프트에 의존하거나, 막대한 계산 자원을 필요로 하는 파인튜닝 과정이 필수적이라는 한계가 있다. ASI-Seg는 이를 오디오 프롬프트로 대체하여 실용성을 높였다.

## 🛠️ Methodology

### 전체 시스템 구조

ASI-Seg는 크게 **Intention-Oriented Multimodal Fusion** 모듈과 **Contrastive Learning Prompt Encoder** 모듈로 구성된다. 오디오 명령이 입력되면 의도를 파악하고, 이미지 및 텍스트 뱅크의 정보와 융합하여 타겟 도구의 특징을 생성한 뒤, 이를 SAM의 마스크 디코더에 입력하여 최종 마스크를 생성한다.

### 상세 구성 요소

#### 1. 오디오 의도 인식 (Audio Intention Recognition)

raw 오디오 신호 $a$를 16K Hz로 샘플링하고 Mel-spectrogram $\text{A}^{\text{mel}}$로 변환한다.
$$\text{A}^{\text{mel}} = \pi(a, a', C_s, W_s, s)$$
이후 훈련 데이터의 평균 $\mu$를 이용하여 $[-1, 1]$ 범위로 정규화한 $\text{A}^{\text{norm}}$을 오디오 인코더 $E_A$와 분류기 $\phi$에 통과시켜 외과의의 의도(타겟 도구 클래스) $C$를 예측한다.
$$C = \phi(E_A(\text{A}^{\text{norm}}))$$

#### 2. 텍스트 융합 (Text Fusion)

단순한 도구 이름만으로는 시각적 특징 추출이 어려우므로, 각 도구의 상세 설명이 저장된 **Instrument Description Bank** $\{B_k\}_{k=1}^K$를 사용한다. 텍스트 인코더 $E_T$를 통해 추출된 텍스트 특징 $f_t$와 학습 가능한 쿼리 $f_c$를 상호 교차 주의 집중(Mutual Cross-Attention) 메커니즘으로 융합하여 도구 쿼리 $q$를 생성한다.
$$q = \text{MLP}(\text{concat}(q_t, q_c))$$

#### 3. 시각적 융합 (Visual Fusion)

이미지 인코더 $E_I$를 통해 이미지 특징 $f_i$를 추출하고, 앞서 생성한 도구 쿼리 $q$와의 유사도 행렬 $S_k$를 계산한다. 이를 통해 이미지와 텍스트 정보가 모두 포함된 다중모달 특징 $\text{F} = \{f_{i-t}^k\}_{k=1}^K$를 생성한다.
$$\{f_{i-t}^n\}_{k=1}^K = \{f_i \cdot S_k + f_i\}_{k=1}^K$$

#### 4. 의도 기반 특징 할당 (Feature Assignment)

오디오 인식 결과 $C$를 바탕으로 전체 특징 $\text{F}$를 필요한 특징 $F^+$와 불필요한 특징 $F^-$로 분리한다.
$$F^+ = \{f_{i-t}^C\}, \quad F^- = \{f_{i-t}^{k, k \neq C}\}_{k=1}^K$$

#### 5. 대조 학습 프롬프트 인코더 (Contrastive Learning Prompt Encoder)

- **구분 교차 주의 집중 (Distinguishing Cross-Attention):** $F^+$와 $F^-$ 사이의 유사한 영역을 찾아내고, 역 잔차 메커니즘(Inverse Residual Mechanism)을 통해 불필요한 특징과 겹치는 정보를 제거하여 타겟 도구만의 고유한 속성을 유지한 특징 $P^*$를 생성한다.
$$P^* = P - \text{Attention}(F^+, F^-)$$
- **대조 학습 (Contrastive Learning):** 필요한 도구 특징 $P$가 불필요한 특징 $N$과는 멀어지고, 정답 마스크(GT)로 필터링된 실제 이미지 특징 $v$와는 가까워지도록 대조 손실 $\mathcal{L}_{CL}$을 정의한다.
$$\mathcal{L}_{CL} = -\frac{1}{K} \sum_{n=1}^K \log \frac{\exp(P^{(C)} \cdot v^{(C)} / \tau)}{\sum_{n=1}^K \exp(P^{(C)} \cdot v^{(n)} / \tau)}$$

#### 6. 마스크 디코더 및 최적화

최종적으로 $P^*$를 foreground 프롬프트로, $F^-$를 background 프롬프트로 사용하여 SAM의 마스크 디코더가 정확한 마스크를 생성하게 한다. 학습 시 이미지/오디오/텍스트 인코더는 동결(freeze)하며, 분류기, 마스크 디코더, 융합 모듈 및 프롬프트 인코더만 최적화한다. 전체 손실 함수는 다음과 같다.
$$\mathcal{L} = \mathcal{L}_{\text{DICE}} + \mathcal{L}_{CL}$$

## 📊 Results

### 실험 설정

- **데이터셋:** EndoVis2017 및 EndoVis2018 데이터셋 사용.
- **비교 대상:** TernausNet, ISINet, S3Net 등 기존 모델 및 Mask2Former+SAM, SurgicalSAM 등 최신 SAM 기반 모델.
- **지표:** Challenge IoU, IoU, mean class IoU (mc IoU).

### 정량적 결과

- **Semantic Segmentation (모든 도구 분할):** EndoVis2018에서 IoU 82.37%, EndoVis2017에서 IoU 71.64%를 기록하며 SOTA 성능을 달성했다. 특히 SurgicalSAM 대비 각각 2.04%, 1.70% 향상된 성능을 보였다.
- **Intention-oriented Segmentation (의도 기반 분할):** 타겟 도구만을 분할하는 실험에서 mc IoU 기준 EndoVis2018 64.18%, EndoVis2017 68.37%를 기록했다. 이는 SurgicalSAM 대비 EndoVis2018에서 5.31%의 큰 성능 향상을 보인 수치이다.

### 정성적 분석 및 강건성

- **시각적 결과:** 정성적 비교 결과, ASI-Seg는 수동 프롬프트 없이도 오디오 명령을 정확히 이해하여 타겟 도구에 최적화된 마스크를 생성했다.
- **강건성 테스트:** 외과의가 도구 이름을 잘못 발음(예: Bipolar Forceps $\rightarrow$ Bipolyr Frocips)하더라도, 모델이 의도를 정확히 파악하여 올바른 도구를 분할하는 강건함을 보였다.

### 절제 연구 (Ablation Study)

- **Instrument Description Bank:** 적용 시 mc IoU가 51.00% $\rightarrow$ 59.42%로 8.42% 상승하여, 텍스트 지식이 도구 간 변별력을 높이는 데 핵심적임을 입증했다.
- **Contrastive Learning:** 적용 시 mc IoU가 55.98% $\rightarrow$ 64.18%로 4.98% 상승하여, 불필요한 특징을 억제하는 효과를 확인했다.

## 🧠 Insights & Discussion

### 강점

본 연구는 수술실이라는 특수한 환경에서 '손을 쓸 수 없는' 외과의의 제약 사항을 오디오 인터페이스로 해결했다는 점에서 실용적 가치가 매우 높다. 특히 단순한 분류를 넘어 SAM의 강력한 분할 능력을 오디오-텍스트-이미지 다중모달 융합을 통해 제어했다는 점이 기술적인 성과이다. 또한, 대조 학습을 통해 타겟 도구와 유사한 다른 도구를 구분해내는 능력을 키운 점이 성능 향상의 주요 원인으로 분석된다.

### 한계 및 논의

논문에서는 인코더들을 동결하여 효율적인 학습을 진행했으나, 이는 사전 학습된 모델의 성능에 의존한다는 것을 의미한다. 수술 도구의 종류가 매우 다양해지거나 환경이 극단적으로 변할 경우, 도구 설명 뱅크(Description Bank)의 내용만으로 충분할지에 대한 의문이 남는다. 또한, 오디오 입력의 실시간 처리 지연 시간(Latency)에 대한 구체적인 분석이 명시되지 않아 실제 수술실에서의 즉각적인 반응성을 확인하기 어렵다.

## 📌 TL;DR

ASI-Seg는 외과의의 음성 명령을 통해 분할 대상 도구를 결정하고, 다중모달 특징 융합과 대조 학습 기반의 프롬프트 생성기를 통해 SAM(Segment Anything Model)을 제어하는 오디오 기반 수술 도구 분할 프레임워크이다. 이 연구는 수동 프롬프트 입력 없이도 높은 정확도의 의도 기반 분할을 가능하게 하여, 실제 수술 환경에서 외과의의 인지 부하를 줄이고 워크플로우를 효율화하는 데 기여할 가능성이 매우 크다.
