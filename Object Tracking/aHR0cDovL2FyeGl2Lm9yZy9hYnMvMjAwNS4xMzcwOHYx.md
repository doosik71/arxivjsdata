# AFAT: Adaptive Failure-Aware Tracker for Robust Visual Object Tracking

Tianyang Xu, Zhen-Hua Feng, Xiao-Jun Wu, Josef Kittler (2020)

## 🧩 Problem to Solve

최근 Visual Object Tracking 분야에서 Siamese 네트워크 기반의 접근 방식은 대규모 데이터셋을 이용한 오프라인 학습을 통해 우수한 성능을 보여왔다. 그러나 이러한 Siamese 패러다임은 기본적으로 **One-shot learning** 방식을 채택하고 있어, 추적 과정 중에 발생하는 온라인 적응(Online adaptation)이 어렵다는 치명적인 한계가 있다.

구체적으로, Siamese 트래커들은 전체 비디오 시퀀스 동안 초기 템플릿 모델을 고정하여 사용하므로, 타겟의 외형이 급격하게 변하거나 복잡한 환경에 노출될 때 추적 위치가 어긋나거나 완전히 실패하는 **Tracking shift** 및 **Failure** 문제가 발생한다. 또한, 현재의 추적 결과(Response)가 얼마나 불확실한지를 측정하는 메커니즘이 부족하여, 잠재적인 추적 실패를 인지하지 못한 채 잘못된 결과를 계속 출력하는 문제가 존재한다. 따라서 본 논문의 목표는 추적 단계에서 온라인으로 추적 품질을 예측하고, 실패를 감지하여 이를 교정할 수 있는 **Failure-aware system**을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 추적 결정 단계(Decision level)에서 **시공간적 응답 맵(Spatio-temporal response maps)의 패턴을 분석하여 추적 실패를 실시간으로 예측**하는 것이다.

1.  **Quality Prediction Network (QPN) 제안**: 단순한 임계값(Threshold) 기반의 품질 측정을 넘어, CNN과 LSTM을 결합하여 여러 프레임의 Response map에서 시공간적 특징을 추출하고, 현재 추적 상태가 성공(Success)인지 실패(Lost)인지를 분류하는 네트워크를 설계하였다.
2.  **Adaptive Failure-Aware Tracker (AFAT) 프레임워크**: 경량화된 모델(Base tracker)과 고성능 모델(Correction tracker)을 유동적으로 선택하는 구조를 제안하였다. QPN이 실패 신호를 보내면, 더 정교한 모델을 통해 추적 결과를 교정함으로써 정확도와 속도의 균형을 맞추었다.
3.  **효율적인 온라인 적응 메커니즘**: QPN의 추론 시간이 매우 짧아(1.2~1.5ms), 기존의 어떤 Siamese 트래커와도 쉽게 결합하여 성능을 향상시킬 수 있는 범용적인 구조를 제시하였다.

## 📎 Related Works

기존의 추적 품질 측정 방식은 주로 다음과 같은 한계가 있었다.

*   **Hand-crafted functions 및 Threshold 기반 방식**: 많은 트래커들이 Response map의 최대값이나 단순한 통계적 수치를 이용하여 임계값과 비교함으로써 실패 여부를 판단하였다. 그러나 이러한 방식은 복잡한 환경에서 발생하는 Response map의 다양한 변동성을 충분히 반영하지 못한다.
*   **Siamese Network의 정적 특성**: SiamRPN++와 같은 최신 모델들은 매우 강력한 특성 추출 능력을 갖추고 있으나, 온라인 업데이트가 없기 때문에 한번 실패하면 회복하기 어렵다.
*   **Online Discriminative Learning**: DCF(Discriminative Correlation Filter) 기반 방식들은 온라인 업데이트를 수행하지만, 잘못된 샘플이 학습에 포함될 경우 모델이 오염되는(Contamination) 문제가 발생한다.

본 논문은 이러한 한계를 극복하기 위해, Response map 자체를 이미지처럼 취급하여 딥러닝 모델(QPN)이 직접 품질을 판단하게 함으로써 더 정교한 실패 감지를 가능하게 하였다.

## 🛠️ Methodology

### 1. Quality Prediction Network (QPN) 구조
QPN은 현재 프레임과 이전 $K$개 프레임의 Response map 시퀀스를 입력으로 받아 현재의 추적 품질 $q_t$를 예측한다.

$$q_t = \text{QPN}(f_{t-K+1}, f_{t-K+2}, \dots, f_t)$$

여기서 $f_i \in \mathbb{R}^{C \times N \times N}$는 $i$번째 프레임의 Response map이며, 본 실험에서는 $K=20$으로 설정하였다.

#### (1) Spatial Feature Extraction (공간 특징 추출)
Response map에서 타겟의 중심 확률 분포를 분석하기 위해 CNN을 사용한다.
*   **전처리**: Response map을 원형 이동(Circular shift)시켜 최대 피크 지점을 네 모서리에 분산 배치함으로써 문맥 정보를 강화한다.
*   **구조**: 3개의 Convolutional layer와 1개의 Fully Connected(FC) layer를 통해 각 프레임의 공간적 표현 $\phi(f_i)$를 추출한다.

#### (2) Temporal Feature Fusion (시간 특징 융합)
타겟의 위치와 외형은 인접 프레임 간에 부드럽게 변한다는 가정하에, 시간적 순서 정보를 활용한다.
*   **구조**: 추출된 공간 특징 시퀀스 $\Phi = \{\phi(f_{t-K+1}), \dots, \phi(f_t)\}$를 두 개의 **LSTM(Long Short-Term Memory)** 모듈에 입력하여 최종 품질 $q_t$를 예측하는 함수 $g(\Phi)$를 수행한다.

### 2. 학습 절차 및 손실 함수
*   **데이터 생성**: TrackingNet 데이터셋에서 기본 트래커(SiameseRPN++ mobilev2)를 실행하여 Response map을 수집한다.
*   **라벨링**: 예측된 Bounding box와 Ground Truth 사이의 IOU를 기준으로 라벨을 부여한다.
    $$y_i = \begin{cases} \text{success}, & \text{if } \text{IOU} > 0.5 \\ \text{lost}, & \text{if } \text{IOU} < 0.1 \\ \text{unassigned}, & \text{others} \end{cases}$$
*   **손실 함수**: Cross-entropy loss를 사용하며, 데이터 불균형(Success 샘플이 훨씬 많음)을 해결하기 위해 가중치를 부여한다. (Success: 0.002, Lost: 1)

### 3. AFAT 추적 알고리즘
AFAT는 **Base tracker**($\text{SiamRPN++}_{\text{mobilev2}}$)와 **Correction tracker**($\text{SiamRPN++}_{\text{resnet50}}$)를 동시에 운용한다.

1.  $\text{SiamRPN++}_{\text{mobilev2}}$를 사용하여 Response map $f$를 계산하고 추적 결과를 기록한다.
2.  QPN에 최근 20개 프레임의 $f$ 시퀀스를 입력하여 품질 $q_t$를 예측한다.
3.  만약 $q_t = \text{lost}$라면, 현재 프레임의 추적 결과를 $\text{SiamRPN++}_{\text{resnet50}}$를 이용해 다시 계산하여 교정하고, Response list를 초기화한다.

## 📊 Results

### 1. 실험 설정
*   **데이터셋**: OTB2015, UAV123, LaSOT, VOT2016, VOT2018, VOT2019.
*   **지표**: Precision, Success rate, AUC, DP, EAO, Accuracy(A), Robustness(R).
*   **비교 대상**: ATOM, Meta-Tracker, SiamRPN++, DaSiam 등 최신 트래커들.

### 2. 주요 결과 분석
*   **Ablation Study**: AFAT는 기본 모델인 $\text{SiamRPN++}_{\text{mobilev2}}$보다 월등한 성능을 보였으며, 심지어 고정적으로 무거운 모델을 쓴 $\text{SiamRPN++}_{\text{resnet50}}$보다도 VOT 데이터셋의 EAO와 Accuracy 면에서 더 높은 성능을 기록하였다. 이는 QPN이 적절한 시점에 교정 모델을 호출함으로써 효율적인 추적이 가능했음을 의미한다.
*   **VOT2018/2019**: VOT2018에서 EAO 0.419, Accuracy 0.605로 SOTA 성능을 달성하였다. VOT2019에서도 EAO 0.295로 최상위권 성능을 보였다.
*   **UAV123 & LaSOT**: UAV123에서는 AUC 0.612로 기존 Siamese 트래커들을 능가하였다. LaSOT에서는 NP(Normalised Precision) 지표에서 향상을 보였으나, AUC 면에서는 $\text{SiamRPN++}_{\text{resnet50}}$보다 약간 낮은 수치를 보였다. 이는 LaSOT의 시퀀스 길이가 매우 길어 장기적 강건성(Long-term robustness)에 대한 요구치가 더 높기 때문으로 분석된다.
*   **속도**: QPN의 오버헤드가 매우 적어, 실패 감지 횟수에 따라 70 FPS에서 87 FPS 사이의 실시간 속도를 유지하였다.

## 🧠 Insights & Discussion

본 연구는 단순한 모델의 고도화가 아니라, **"추적 결과의 신뢰도를 스스로 판단할 수 있는가"**라는 메타 인지적 관점을 추적 시스템에 도입하였다는 점에서 큰 의의가 있다.

**강점:**
*   **효율적인 모델 스위칭**: 항상 무거운 모델을 사용하는 대신, 필요할 때만 고성능 모델을 사용하는 전략을 통해 정확도와 속도라는 트레이드-오프(Trade-off) 문제를 효과적으로 해결하였다.
*   **시공간적 정보 활용**: 단일 프레임의 응답 값만 보는 것이 아니라, LSTM을 통해 시간적 흐름을 분석함으로써 일시적인 노이즈와 실제 실패를 더 잘 구분할 수 있게 되었다.

**한계 및 논의사항:**
*   **데이터 의존성**: QPN이 특정 베이스 트래커($\text{SiamRPN++}$)의 Response map 패턴을 학습했기 때문에, 다른 구조의 트래커에 적용하려면 QPN을 다시 학습시켜야 할 가능성이 크다.
*   **장기 추적의 한계**: LaSOT 결과에서 나타났듯이, 매우 긴 시퀀스에서의 완전한 강건성을 확보하기 위해서는 단순히 모델을 교체하는 것 이상의 온라인 템플릿 업데이트 전략이 병행되어야 할 것으로 보인다.
*   **오경보(False Alarm)**: 논문 내 Fig 4에서 보듯 일부 오경보가 발생하지만, 저자들은 보수적인 학습 전략을 통해 이를 완화하려 하였다.

## 📌 TL;DR

본 논문은 Siamese 트래커의 고질적인 문제인 '온라인 적응 부재'와 '실패 인지 불가' 문제를 해결하기 위해, **Response map의 시공간적 패턴을 학습하여 추적 실패를 예측하는 QPN(Quality Prediction Network)**을 제안한다. 이를 통해 경량 모델로 추적하다가 실패가 감지되면 고성능 모델로 교정하는 **AFAT** 프레임워크를 구축하였으며, 이를 통해 실시간성(70+ FPS)을 유지하면서도 VOT2018, 2019 등 주요 벤치마크에서 SOTA 수준의 성능을 달성하였다. 이 연구는 향후 자가 진단 능력을 갖춘 강건한 비주얼 트래커 설계에 중요한 방향성을 제시한다.