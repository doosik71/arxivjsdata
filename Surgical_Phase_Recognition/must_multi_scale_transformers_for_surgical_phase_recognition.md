# MuST: Multi-Scale Transformers for Surgical Phase Recognition

Alejandra Pérez, Santiago Rodríguez, Nicolás Ayobi, Nicolás Aparicio, Eugénie Dessevres, and Pablo Arbeláez (2024)

## 🧩 Problem to Solve

본 논문은 수술 비디오에서의 **Surgical Phase Recognition(수술 단계 인식)** 문제를 해결하고자 한다. 수술 단계 인식은 수술 절차의 순차적인 단계들을 자동으로 이해함으로써 컴퓨터 보조 수술 시스템을 강화하고, 사후 분석 및 의료진 교육에 기여하는 핵심적인 작업이다.

이 문제의 주요 어려움은 수술 데이터 내에서 각 단계의 **지속 시간(duration)이 매우 가변적**이라는 점과, 서로 다른 단계 간의 **시맨틱 유사성(semantic similarity)**이 높다는 점이다. 기존의 방법론들은 주로 고정된 시간 윈도우(fixed temporal windows)를 사용하여 비디오를 분석하는데, 이는 복잡한 수술 절차를 완전히 이해하는 데 필요한 단기(short-term), 중기(mid-term), 장기(long-term) 정보를 동시에 캡처하는 데 한계가 있다. 따라서 본 논문의 목표는 다양한 시간 척도(temporal scales)의 정보를 통합적으로 처리하여, 수술 단계의 가변성에 유연하게 대응하고 시간적 일관성(temporal consistency)이 높은 예측을 수행하는 모델을 개발하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **Multi-Scale Transformer** 구조를 통해 수술 비디오의 다중 시간 척도 정보를 계층적으로 추출하고 이를 통합하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Multi-Term Frame Encoder (MTFE)**: 다양한 샘플링 비율로 구성된 **Temporal Multi-Scale Pyramid**와 **Multi-Temporal Attention Module**을 도입하여, 단기 및 중기 시간 범위의 의존성을 동시에 캡처하는 풍부한 프레임 임베딩을 생성한다.
2. **Temporal Consistency Module (TCM)**: MTFE에서 생성된 임베딩 시퀀스를 가벼운 Long-term Transformer Encoder로 처리함으로써, 수술 절차 전체의 장기적 의존성을 모델링하고 예측의 시간적 일관성을 향상시킨다.

## 📎 Related Works

기존의 수술 단계 인식 연구는 다음과 같은 흐름으로 발전해 왔다.

* **전통적 머신러닝 및 CNN**: 초기에는 수동 설계된 특징(manually designed features)을 사용하였으며, 이후 EndoNet과 같은 CNN 기반 모델이 등장하여 공간적 특징을 추출하였다. 하지만 이들은 시간적 맥락(temporal context)을 반영하지 못했다.
* **RNN 및 LSTM 기반 모델**: PhaseNet, EndoLSTM 등은 LSTM을 사용하여 시간적 의존성을 모델링하려 했으나, 순차적 처리 특성상 Gradient Vanishing 문제로 인해 장기 의존성 학습에 한계가 있었다.
* **TCN 기반 모델**: TeCNO 등은 Dilated Convolution을 사용하는 Temporal Convolutional Networks(TCN)를 통해 수용 영역을 넓혔으나, 이 방식은 세밀한 단기 정보(fine-grained information)를 놓치는 경향이 있다.
* **Transformer 기반 모델**: OperA, Trans-SVNet 등은 Transformer를 도입하여 장기 의존성을 해결하려 했으나, 여전히 TCN에 의존하거나 고정된 크기의 윈도우를 사용하여 시간적 일관성이 부족하거나 세밀한 정보 손실이 발생하는 문제가 있었다. TAPIS와 같은 최신 모델은 MViT 백본을 사용하지만, 여전히 고정 윈도우 방식과 프레임 단위 예측으로 인해 일관성 확보에 어려움이 있다.

MuST는 이러한 한계를 극복하기 위해 고정 윈도우 대신 **다중 척도의 피라미드 샘플링**과 **단계적 Transformer 구조(MTFE $\rightarrow$ TCM)**를 제안하여 차별성을 가진다.

## 🛠️ Methodology

MuST는 크게 두 단계의 파이프라인으로 구성된다.

### 1. Multi-Term Frame Encoder (MTFE)

MTFE는 특정 키프레임(keyframe)을 중심으로 단기 및 중기 문맥을 캡처하여 풍부한 임베딩을 생성하는 역할을 한다.

* **Temporal Multi-Scale Pyramid**: 키프레임을 중심으로 샘플링 간격(stride)이 점진적으로 증가하는 $N$개의 서브 시퀀스 집합 $X=\{x_i\}_{i=1}^N$를 구성한다. 낮은 레벨은 조밀한 샘플링을 통해 세밀한 정보를, 높은 레벨은 성긴 샘플링을 통해 넓은 문맥 정보를 제공한다.
* **Video Backbone**: MViT(Multiscale Vision Transformer)를 사용하여 각 시퀀스 $x_i$로부터 시공간 임베딩 $l_i$와 클래스 토큰 $cls_i$를 추출한다.
* **Multi-Temporal Attention Module**: 서로 다른 척도의 임베딩 간 상호작용을 계산한다.
  * **Multi-Temporal Cross-Attention (MTCA)**: 각 척도의 임베딩 $l_i$를 쿼리(Query)로, 모든 척도의 통합 임베딩 $l' = \text{concat}(L)$을 키(Key)와 값(Value)으로 사용하여 교차 주의 집중을 수행한다.
        $$\text{MTCA}(l_i, l'_i) = \text{softmax}\left(\frac{Q_i \cdot K_i^T}{\sqrt{d_{k_i}}}\right)V_i$$
  * **Multi-Temporal Self-Attention (SA)**: MTCA의 결과물 $C=\{c_i\}_{i=1}^N$와 원래의 $l_i$를 결합하여 다시 한 번 셀프 어텐션을 수행함으로써 정보를 정제한다.
* **Final Embedding**: 각 척도에서 추출된 클래스 토큰들을 연결(concatenate)한 후, MLP를 통과시켜 최종적인 **Multi-term frame embedding** $p$를 생성한다.

### 2. Temporal Consistency Module (TCM)

MTFE가 생성한 프레임 임베딩들은 개별적으로 생성되므로 시간적 흐름이 끊길 수 있다. TCM은 이를 보완하여 장기적인 일관성을 부여한다.

* **구조**: 생성된 임베딩 $p$들을 겹치는 윈도우(overlapping windows) 형태로 구성하여 Transformer Encoder에 입력한다.
* **위치 임베딩**: 시간적 순서 정보를 제공하기 위해 Cosine Positional Embedding ($PE$)을 추가한다.
* **연산**: 표준 Self-attention 메커니즘을 통해 윈도우 내 프레임들 간의 관계를 학습한다.
    $$\text{TCM}(v) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)V$$
* **추론**: 오프라인 추론 시, 각 프레임에 대해 여러 윈도우에서 계산된 확률 분포의 평균을 내어 최종 클래스를 결정함으로써 예측의 부드러움(smoothing)과 일관성을 확보한다.

## 📊 Results

### 실험 설정

* **데이터셋**: HeiChole(온라인 설정, F1-score), GraSP 및 MISAW(오프라인 설정, mAP)의 세 가지 벤치마크에서 평가하였다. 추가적으로 Cholec80 데이터셋에서도 검증하였다.
* **구현 세부사항**: MViT-B를 백본으로 사용하였으며, MTFE에서는 4개의 샘플링 척도(1, 4, 8, 12초 간격)를 사용하였다. TCM의 윈도우 크기는 비디오 평균 길이의 5~10%로 설정하였다.

### 주요 결과

MuST는 모든 벤치마크에서 기존 SOTA(State-of-the-art) 모델들을 능가하는 성능을 보였다.

| Dataset | TAPIS | TeCNO | Trans-SVNet | **MuST (Ours)** |
| :--- | :---: | :---: | :---: | :---: |
| **GraSP (mAP)** | 76.07 | 77.10 | 76.54 | **79.14** |
| **MISAW (mAP)** | 97.14 | 95.58 | 90.38 | **98.08** |
| **HeiChole (F1)** | 73.41 | 69.35 | 71.85 | **77.25** |

* **정성적 분석**: 특히 GraSP 데이터셋과 같이 단계별 지속 시간이 매우 가변적인 경우, MuST의 다중 척도 추론 능력이 급격한 단계 변화를 더 정확하게 포착함을 확인하였다.
* **Ablation Study**: 실험 결과, $\text{Multi-Sequence} \rightarrow \text{Multi-Scale Pyramid} \rightarrow \text{Attention Module} \rightarrow \text{TCM}$ 순으로 구성 요소를 추가할 때마다 성능이 단계적으로 향상됨을 입증하였다. 특히 TCM의 유무에 따라 예측의 일관성이 크게 달라짐이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 수술 비디오 분석에서 **'시간적 해상도의 가변성'**을 어떻게 처리할 것인가에 대한 효과적인 해답을 제시하였다.

**강점**:
단순히 윈도우 크기를 키우는 것이 아니라, 피라미드 구조의 샘플링을 통해 계산 효율성을 유지하면서도 단기/중기 문맥을 동시에 잡았다는 점이 뛰어나다. 또한, MTFE에서 생성된 풍부한 특징을 TCM이라는 별도의 장기 모델링 모듈로 정제함으로써, 딥러닝 모델이 흔히 겪는 '예측 결과의 진동(jittering)' 문제를 효과적으로 해결하였다.

**한계 및 논의**:
논문에서는 오프라인 추론 시 윈도우 평균화(averaging)를 통해 성능을 높였는데, 이는 실시간 시스템(online setup)에서는 완전한 적용이 어려울 수 있다. 또한, 사용된 MViT 백본의 사전 학습(pretrained) 상태에 따라 성능 영향이 클 수 있다는 점이 잠재적인 변수로 작용한다. 하지만 전반적으로 수술 워크플로우 분석에서 Transformer의 활용 방식을 다중 척도로 확장했다는 점에서 학술적 가치가 높다.

## 📌 TL;DR

MuST는 수술 단계 인식의 고질적 문제인 **단계별 지속 시간의 가변성**을 해결하기 위해, 다중 척도 샘플링 피라미드(MTFE)와 장기 일관성 모듈(TCM)을 결합한 Transformer 기반 모델이다. 이 구조를 통해 단기-중기-장기 정보를 계층적으로 통합함으로써 GraSP, MISAW, HeiChole 등 주요 벤치마크에서 SOTA 성능을 달성하였으며, 특히 복잡하고 가변적인 수술 절차의 시간적 흐름을 매우 정밀하게 복원할 수 있음을 보여주었다. 이는 향후 실시간 수술 보조 시스템의 정확도를 높이는 데 중요한 기초 연구가 될 것으로 기대된다.
