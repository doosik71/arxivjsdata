# SuPRA: Surgical Phase Recognition and Anticipation for Intra-Operative Planning

Maxence Boels, Yang Liu, Prokar Dasgupta, Alejandro Granados, Sebastien Ourselin (2024)

## 🧩 Problem to Solve

본 논문은 수술 중 실시간으로 현재의 수술 단계(Surgical Phase)를 인식하는 것을 넘어, 향후 진행될 단계를 예측(Anticipation)함으로써 수술실 내의 상황 인지 능력을 높이고 수술자의 의사결정 및 계획 수립을 돕는 것을 목표로 한다.

기존의 온라인 수술 단계 인식 시스템은 주로 현재 어떤 단계인지를 맞추는 데 집중되어 있으며, 이는 수술 후 비디오 분석에는 유용할 수 있으나 실제 수술이 진행 중인 상황에서 수술자에게 직접적인 도움을 주기에는 한계가 있다. 반면, 미래의 단계를 예측하는 기능은 수술자가 즉각적이고 장기적인 계획을 세우는 데 실질적인 가이드를 제공할 수 있다. 따라서 본 연구는 현재 단계의 인식(Recognition)과 미래 단계의 예측(Anticipation)을 동시에 수행하는 통합 프레임워크를 구축하여 수술 중 실시간 지원 시스템의 효용성을 극대화하고자 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 수술 단계의 인식과 예측을 단일 네트워크에서 동시에 수행하는 다중 작업(Multi-task) 아키텍처인 **SuPRA (Surgical Phase Recognition and Anticipation)**를 제안한 것이다.

중심적인 설계 아이디어는 과거와 현재의 정보를 활용해 현재 단계를 인식함과 동시에, 미래 세그먼트를 생성하여 다음에 올 단계와 그 지속 시간(Duration)을 예측하는 것이다. 이는 인식과 예측을 별개의 작업으로 처리하던 기존 방식에서 벗어나, 두 작업을 통합함으로써 서로가 서로의 학습을 돕도록 설계되었다. 또한, 단순한 프레임 단위 정확도뿐만 아니라 시간적 연속성을 평가할 수 있는 **Edit Score**와 **F1 Overlap**이라는 세그먼트 수준의 평가 지표를 도입하여 분석의 정밀도를 높였다.

## 📎 Related Works

### 기존 연구 및 한계

1. **Surgical Phase Recognition**: 초기에는 HMM이나 통계적 방법이 사용되었으며, 이후 CNN과 RNN(LSTM)으로 발전하였다. 최근에는 Temporal ConvNets(TCN)를 활용한 TeCNO나 Trans-SVNet(TSVN) 등이 제안되었으나, TCN은 메모리 요구량이 많고 확장된 수용 영역(Receptive field)으로 인해 세부 정보가 손실될 위험이 있다. Transformers는 긴 시간적 의존성을 학습하는 데 유리하지만, 계산 복잡도로 인해 비디오 인식 적용에 제약이 있었다.
2. **Surgical Workflow Prediction**: 미래의 동작이나 단계를 예측하려는 시도가 있었으며, 주로 회귀(Regression) 방식을 통해 다음 단계가 나타날 때까지 남은 시간을 예측하는 방식이 주를 이뤘다. 하지만 이러한 접근법은 장기적인 계획 수립에 필요한 긴 시퀀스의 세그먼트와 지속 시간을 직접적으로 예측하지 못한다는 한계가 있다.

### 차별점

SuPRA는 단순히 남은 시간을 예측하는 회귀 작업이 아니라, 미래의 단계 시퀀스와 그 지속 시간을 분류(Classification) 및 예측하는 통합 Transformer 구조를 채택함으로써 더 포괄적인 수술 워크플로우 가이드를 제공한다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

SuPRA는 크게 **Spatial Feature Extractor**, **Long-Term Compression**, **Future Generation**, **Frame Recognition**, **Segment Prediction**의 다섯 가지 모듈로 구성된다.

1. **Spatial Feature Extractor**: ViT-B/16 기반의 백본을 사용하여 각 프레임 $x_t$를 압축된 표현 $f_t$로 인코딩한다.
2. **Long-Term Compression (Past-Present Encoder)**:
   - 슬라이딩 윈도우 셀프 어텐션(Sliding window self-attention)을 통해 단기적인 시간적 패턴을 캡처한다.
   - 이후 **Key-pooling**이라 불리는 max-pooling 연산을 적용하여, 현재까지 관찰된 가장 중요한 특징들을 추출한 저차원 벡터 $K_t$를 생성한다.
3. **Future Generation (Future Decoder)**:
   - Future Decoder $\text{Dec}_F$는 압축된 특징 $K_t$와 입력 쿼리 $Q=\{q_1, \dots, q_n\}$를 받아 미래의 디코딩된 세그먼트 $S=\{s_1, \dots, s_n\}$를 생성한다.
4. **Frame Recognition**:
   - $K_t$와 최근 $w$개 프레임의 인코딩 결과 $E_t$를 결합(Fusion)하여 현재 프레임의 클래스 확률을 예측한다. 이때 시간적 일관성을 높이기 위해 Consistency Cross-Entropy 손실 함수를 사용한다.
5. **Segment Prediction**:
   - 두 개의 분류 헤드를 통해 다음에 올 단계 $\hat{y}$와 그 지속 시간 $\hat{d}$를 동시에 예측한다.

### 학습 목표 및 손실 함수

전체 손실 함수 $L_{\text{total}}$은 다음과 같이 정의된다.

$$L_{\text{total}} = L_{\text{current-phase}} + L_{\text{next-phase}} + L_{\text{next-duration}} (+ L_{\text{next-k-features}})$$

- **$L_{\text{current-phase}}, L_{\text{next-phase}}$**: 현재 및 다음 단계 예측의 정확도를 높이기 위해 Cross-Entropy(CE) 손실을 사용한다.
- **$L_{\text{next-duration}}$**: 다음 단계의 지속 시간을 정밀하게 추정하기 위해 Mean Squared Error(MSE) 손실을 사용한다.
- **$L_{\text{next-k-features}}$**: (선택 사항) 미래의 핵심 특징(Key features)을 정확하게 재구성하기 위해 MSE 손실을 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80(C80, 7개 클래스) 및 AutoLaparo21(AL21)을 사용하였다.
- **비교 대상**: SKiT 및 Trans-SVNet(TSVN)과 성능을 비교하였다.
- **지표**: Frame-level Accuracy, Edit Score, F1 Overlap을 측정하였다.

### 주요 결과

1. **Phase Recognition (인식 성능)**:
   - Cholec80에서 SuPRA는 $91.8\%$의 정확도를 기록하며 SKiT($92.4\%$)와 대등한 수준을 보였으며, Edit Score와 F1 Score에서도 유사한 성능을 나타냈다.
   - AutoLaparo21에서는 SuPRA가 $79.3\%$의 정확도를 달성하며 SKiT($77.3\%$)보다 우수한 성능을 보였으며, 모든 지표에서 SOTA(State-of-the-art)를 달성하였다.
2. **Next Phase Prediction (예측 성능)**:
   - Cholec80에서 SuPRA의 정확도는 $83.3\%$로 SKiT($83.7\%$)와 비슷했으나, Edit Score($16.4\%$)와 F1 Score에서는 다른 모델들을 압도하였다.
   - AutoLaparo21에서는 정확도 $66.1\%$를 기록하며 경쟁 모델들보다 월등히 높은 성능을 보였다.
3. **Ablation Study**:
   - 미래 단계를 예측하는 작업(Anticipation)을 추가했을 때, AutoLaparo21 데이터셋에서는 인식(Recognition) 성능이 향상되는 경향을 보였으나, Cholec80에서는 뚜렷한 향상이 없었다. 이는 데이터셋의 특성에 따라 미래 정보가 현재 인식에 주는 영향이 다름을 시사한다.
   - 예측 대상 세그먼트의 수를 늘릴수록(1개 $\rightarrow$ 4개) 예측 정확도는 점진적으로 감소하는 경향을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석

SuPRA는 인식과 예측을 통합함으로써 단일 작업만 수행하는 모델보다 더 강건한 표현력을 학습할 수 있음을 보여주었다. 특히 AutoLaparo21과 같이 복잡한 시나리오에서 예측 작업이 인식 작업의 정규화(Regularization) 역할을 하여 성능을 끌어올린 점이 주목할 만하다. 또한, 기존의 프레임 단위 정확도 지표가 과분할(Over-segmentation) 문제를 제대로 포착하지 못한다는 점을 지적하고, Edit Score 등을 통해 시간적 흐름의 정확성을 평가한 점이 학술적으로 가치가 있다.

### 한계 및 미해결 과제

실험 결과에서 Edit Score와 F1 Score가 전반적으로 낮게 나타났는데, 이는 모델이 예측 과정에서 빈번하게 클래스를 전환하는 '진동(Oscillation)' 현상이 발생함을 의미한다. 이는 수술 과정의 내재적인 가변성이나 레이블링의 거친 특성, 혹은 아키텍처의 한계 때문일 수 있다.

### 향후 연구 방향

- 예측된 단계의 지속 시간(Duration)을 실제 길이와 더 정밀하게 일치시키는 재구성 연구가 필요하다.
- 순서가 덜 정형화된 수술 단계(Surgical steps)나 개별 동작(Actions) 예측과 같은 더 어려운 작업으로 확장할 필요가 있다.
- 텍스트 생성 모델에서 사용되는 자기회귀적(Autoregressive) 디코딩 방식을 도입하여 수술 계획 생성의 효율성을 시험해 볼 수 있다.

## 📌 TL;DR

본 논문은 수술 중 현재 단계를 인식함과 동시에 미래 단계를 예측하는 통합 Transformer 모델인 **SuPRA**를 제안하였다. 이 모델은 Cholec80 및 AutoLaparo21 데이터셋에서 SOTA 성능을 기록했으며, 특히 세그먼트 수준의 평가 지표를 통해 시간적 연속성 측면에서의 우수성을 입증하였다. 본 연구는 단순한 사후 분석을 넘어, 수술자에게 실시간 가이드를 제공할 수 있는 수술 워크플로우 생성 시스템의 기반을 마련했다는 점에서 중요한 의미를 갖는다.
