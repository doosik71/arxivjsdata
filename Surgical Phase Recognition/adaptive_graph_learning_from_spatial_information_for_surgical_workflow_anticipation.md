# Adaptive Graph Learning from Spatial Information for Surgical Workflow Anticipation

Francis Xiatian Zhang, Jingjing Deng, Robert Lieck, Hubert P. H. Shum (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 로봇 보조 수술(Robotic-Assisted Surgery, RAS) 환경에서 실시간 비디오 데이터를 통해 향후 발생할 수술 이벤트의 타이밍을 예측하는 **Surgical Workflow Anticipation**이다. 수술 워크플로우 예측은 수술 도구의 준비 시간을 단축하고, 지능형 로봇 보조 시스템을 설계하며, 수술 팀 간의 의사소통을 원활하게 하여 환자의 안전과 수술실 운영 효율성을 높이는 데 매우 중요하다.

기존의 예측 방법론들은 다음과 같은 세 가지 주요 한계를 가지고 있다. 첫째, 수술 도구에만 집중하고 수술 대상(Surgical Target)을 무시하며, 공간 정보 추출 과정의 불확실성(Uncertainty)을 고려하지 않아 불완전한 장면 표현을 제공한다. 둘째, 도구와 대상 간의 상호작용을 정적인 그래프(Static Graph)로 모델링하여, 수술 단계에 따라 역동적으로 변하는 상호작용의 특성을 반영하지 못한다. 셋째, 고정된 시간 지평(Fixed Time Horizon, $h$)을 사용하여 학습하고 평가함으로써, 환자의 상태에 따라 다양하게 변하는 수술 지속 시간에 유연하게 대응하지 못하고 일반화 성능이 떨어진다.

따라서 본 논문의 목표는 수술 대상과 도구를 모두 포함하는 새로운 공간 표현법을 도입하고, 동적인 상호작용을 캡처하는 적응형 그래프 학습(Adaptive Graph Learning) 및 다양한 시간 지평을 동시에 최적화하는 다중 시간 지평 목적 함수(Multi-horizon Objective)를 통해 예측 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수술 장면의 역동성과 불확실성을 반영할 수 있는 **적응형 그래프 학습 프레임워크**를 구축하는 것이다. 주요 기여 사항은 다음과 같다.

1. **포괄적인 공간 정보 표현**: 수술 도구뿐만 아니라 수술 대상(Target)의 Bounding Box와 검출 신뢰도(Confidence level)를 함께 사용하는 새로운 표현법을 제안한다. 이를 위해 Cholec80 및 Cataract101 데이터셋에 대한 추가적인 타겟 어노테이션을 제공하여 모델의 견고함을 높였다.
2. **Adaptive Graph Learning**: 고정된 그래프 대신, 각 프레임의 상황에 적합한 후보 그래프를 동적으로 선택하는 메커니즘을 도입하여 수술 중 발생하는 가변적인 상호작용을 효과적으로 모델링한다.
3. **Multi-horizon Training Strategy**: 단일한 고정 시간 지평이 아니라, 여러 시간 지평에 대한 손실 함수를 학습 가능한 가중치로 결합한 다중 시간 지평 목적 함수를 설계하여, 제약 없는(Unconstrained) 예측이 가능하도록 하였다.

## 📎 Related Works

수술 시나리오에서 공간 정보의 활용에 관한 기존 연구들은 주로 3D 메쉬 모델이나 단순한 시각적 특징 추출에 의존하였다. 일부 연구는 도구의 Bounding Box나 키포인트를 사용하였으나, 수술 대상(Target)의 움직임을 무시하여 상호작용 표현이 불완전했다.

그래프 학습(Graph Learning) 분야에서는 수술 장면 이해나 워크플로우 분석을 위해 그래프 구조를 도입한 사례가 많으나, 대부분의 경우 모든 프레임에 동일한 정적 그래프를 적용하였다. 이는 수술 과정에서 도구 간의 역할과 관계가 계속해서 변하는 역동적 특성을 반영하지 못한다는 한계가 있다.

Surgical Workflow Anticipation 분야의 기존 SOTA 모델들(예: IIA-Net)은 시각적 특징과 비시각적 특징을 융합하여 사용하지만, 여전히 고정된 시간 지평($h$) 하에서 학습되는 경향이 있어 다양한 수술 케이스에 대한 일반화 능력이 부족하다.

## 🛠️ Methodology

본 논문이 제안하는 전체 파이프라인은 **공간 정보 추출 $\rightarrow$ 적응형 그래프 학습 $\rightarrow$ 다중 시간 지평 예측**의 세 단계로 구성된다.

### 1. 공간 정보 표현 (Spatial Information Representation)

원시 비디오 프레임에서 YOLOv5를 사용하여 수술 도구와 대상의 Bounding Box를 추출한다. Bounding Box는 픽셀 단위의 세그멘테이션보다 비정형적인 수술 환경에서 더 안정적인 표현을 제공한다.

추출된 특징 벡터 $b_t$는 다음과 같이 정의된다:
$$b_t = \{x_t, y_t, w_t, h_t, c_t\}$$
여기서 $(x_t, y_t)$는 중심 좌표, $(w_t, h_t)$는 너비와 높이, $c_t$는 검출 신뢰도이다. 또한, 검출 결과의 노이즈를 줄이기 위해 Temporal Attention 메커니즘을 적용하여 신뢰할 수 있는 프레임에 더 높은 가중치를 부여한다:
$$\text{Attn} = \sigma(\text{Conv1D}(\text{MaxPool}(b_T)) + \text{Conv1D}(\text{AvgPool}(b_T)))$$
$$\hat{b}_T = b_T \cdot \text{Attn}_{\text{Temporal}}$$

### 2. 적응형 그래프 학습 (Adaptive Graph Learning)

이 단계는 후보 그래프 선택(Candidate Graph Selection)과 그래프 기반 특징 학습(Graph-based Feature Learning)으로 나뉜다.

**A. 후보 그래프 선택**
학습 데이터에서 가장 빈번하게 발생하는 도구-대상 조합을 바탕으로 초기 후보 그래프 집합 $G$를 구성한다. 이후 Policy Network(TCN 기반)와 **Gumbel-Sinkhorn** 연산을 통해 각 프레임에 가장 적합한 상위 $k$개의 그래프를 미분 가능한 방식으로 선택한다. Gumbel-Sinkhorn은 이산적인 선택 과정을 연속적인 최적화 문제로 변환하여 엔드투엔드 학습을 가능하게 한다.

**B. 그래프 기반 특징 학습**

1. **Spatial Graph Convolution**: 선택된 각 그래프에 대해 독립적인 그래프 컨볼루션을 적용하여 노드 특징을 업데이트한다.
    $$\tilde{H}_g^{(s)l} = \Lambda_k^{-1/2} (A_k + I) \Lambda_k^{-1/2} \tilde{H}_g^{(k)l-1} W_g^{(k)l}$$
2. **Node Fusion**: Squeeze-and-Excitation(SE) 채널 어텐션을 사용하여 서로 다른 후보 그래프에서 추출된 특징들을 적응적으로 융합한다.
3. **Temporal Convolution**: Dilated Causal 2D Convolution을 통해 시간축으로 특징을 집계하여 장기적인 시간적 상관관계를 학습한다.

### 3. 다중 시간 지평 목적 함수 (Multi-horizon Objective)

특정 시간 지평 $h$에 종속되지 않는 일반화된 예측을 위해, 여러 $h$에 대한 손실 함수를 결합한다. 학습 가능한 분산 표현 $\lambda$를 도입하여 각 지평의 손실 가중치를 자동으로 조절한다.

전체 손실 함수 $L$은 다음과 같다:
$$L = \text{oMAE}_H + \sum_{h \in \mathcal{H}} \left( \frac{\text{inMAE}_h}{2\hat{\lambda}_h} + \log(\hat{\lambda}_h) \right)$$
여기서 $\hat{\lambda}_h = \text{SoftPlus}(\lambda_h)$이며, $\text{inMAE}_h$는 지평 내부의 예측 오차, $\text{oMAE}_H$는 최대 지평 외부의 예측 오차를 의미한다. 이 구조는 모델이 단기 예측에 더 높은 가중치를 두면서도 장기적인 예측 능력을 유지하게 한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80(담낭 절제술), Cataract101(백내장 수술)
- **작업**: 수술 도구 및 단계 예측(Instrument/Phase Anticipation), 남은 수술 시간 예측(Remaining Surgical Duration, RSD)
- **지표**: wMAE(inMAE와 oMAE의 평균), eMAE(매우 단기 예측 정확도), MAE
- **비교 대상**: TimeLSTM, RSDNet, TempAgg, B-CNN-LSTM, IIA-Net, GCN-MSTCN, CataNet 등

### 주요 결과

1. **수술 도구 및 단계 예측**: Cholec80 데이터셋에서 SOTA 모델인 IIA-Net 대비 단계 예측에서 약 3%의 MAE 감소를 보였다. 특히 매우 단기 예측 성능을 나타내는 eMAE에서 우수한 성능을 기록하여 RAS의 실시간 대응 가능성을 입증하였다.
2. **남은 수술 시간(RSD) 예측**: Cataract101 데이터셋에서 CataNet 대비 약 9%, GCN-MSTCN 대비 최대 68%의 성능 향상을 달성하였다. 이는 정적 그래프 기반 방식보다 적응형 그래프 방식이 수술의 전체 흐름을 파악하는 데 훨씬 효과적임을 보여준다.
3. **강건성 분석**: 가우시안 블러(Gaussian Blur)나 조명 변화와 같은 시각적 아티팩트가 존재하는 상황에서도 Bounding Box 기반 표현이 픽셀 기반 시각 모델보다 훨씬 안정적인 예측 추세를 유지함을 확인하였다.

## 🧠 Insights & Discussion

본 논문의 강점은 수술 장면의 **불확실성과 역동성**을 수학적, 구조적으로 잘 풀어냈다는 점이다. 특히 세그멘테이션 마스크는 수술 도구의 상호작용 시 형태가 급격히 변해 노이즈가 발생하기 쉬우나, Bounding Box는 위치와 크기라는 핵심 정보에 집중함으로써 훨씬 안정적인 입력값을 제공한다. 또한, 다중 시간 지평 학습을 통해 단일 모델로 다양한 시간 범위의 예측을 수행할 수 있게 하여 실용성을 높였다.

다만, 한계점으로는 Bounding Box 기반 표현이 텍스처(Texture) 정보와 같은 저수준의 세부 디테일을 손실한다는 점이 있다. 실제로 Clipper 사용 단계와 같은 특정 세부 단계의 예측에서 오차가 발생하는 경향이 있는데, 이는 단순한 박스 정보만으로는 복잡한 해부학적 구조를 완전히 캡처하기 어렵기 때문으로 분석된다.

비판적으로 해석하자면, 제안된 모델이 단기 및 중기 예측에서는 SOTA를 상회하지만, 매우 장기적인 예측에서는 고정 지평 모델만큼의 압도적인 차이를 보이지 않는 경우가 있다. 이는 모든 지평을 동시에 최적화하는 과정에서 발생하는 트레이드오프(Trade-off)일 가능성이 크며, 향후 확산 모델(Diffusion-based models) 등을 통한 장기 의존성 모델링이 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 RAS의 워크플로우 예측을 위해 **수술 대상-도구의 Bounding Box 기반 공간 표현**, **동적 상호작용을 캡처하는 적응형 그래프 학습**, 그리고 **학습 가능한 가중치를 가진 다중 시간 지평 손실 함수**를 제안하였다. 실험 결과, 기존 SOTA 모델 대비 수술 단계 예측 오차를 3%, 남은 수술 시간 예측 오차를 9% 감소시켰으며, 특히 시각적 노이즈에 강건한 성능을 보였다. 이 연구는 실제 수술실에서 의료진의 도구 준비 및 협업 효율성을 높여 수술 안전성을 개선하는 데 기여할 가능성이 매우 높다.
