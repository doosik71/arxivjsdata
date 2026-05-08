# ST-USleepNet: A Spatial-Temporal Coupling Prominence Network for Multi-Channel Sleep Staging

Jingying Ma, Qika Lin, Ziyu Jia, and Mengling Feng (2024)

## 🧩 Problem to Solve

수면 단계 분석(Sleep staging)은 수면의 질을 평가하고 수면 장애를 진단하는 데 필수적이다. 전통적으로는 전문가가 다채널 생체 신호(EEG, EOG, EMG 등)를 분석하여 수면 단계를 분류하지만, 이는 매우 노동 집약적이며 대규모 데이터셋에 적용하기 어렵다는 한계가 있다. 최근 인공지능을 이용한 자동 수면 단계 분류 연구가 진행되고 있으나, 다음과 같은 두 가지 핵심적인 문제가 남아 있다.

첫째, 다채널 raw 신호로부터 특성 수면 파형(characteristic sleep waveforms)과 같은 **Temporal feature**와 주요 뇌 네트워크(salient spatial brain networks)와 같은 **Spatial feature**를 동시에 효과적으로 추출하는 것이 어렵다. 기존 연구들은 주로 단일 채널 신호에 집중하거나, 다채널 신호의 복잡한 공간적 상호작용을 충분히 반영하지 못했다.

둘째, 뇌 활동을 정확히 이해하기 위해 필수적인 **Spatial-Temporal Coupling patterns**(시공간 결합 패턴)을 캡처하는 메커니즘이 부족하다. 대부분의 기존 모델은 시간적 상관관계나 공간적 의존성 중 하나에만 치중하여, 시간이 흐름에 따라 변화하는 뇌 영역 간의 복잡한 상호작용을 포착하지 못하고 있다.

따라서 본 논문의 목표는 다채널 raw 신호에서 시공간 결합 패턴을 모델링하고, 유의미한 시공간 특징을 동시에 추출하여 수면 단계 분류의 정확도를 높이는 ST-USleepNet 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수면 신호를 시공간 그래프로 변환하여 결합 패턴을 학습하고, 이미지 세그멘테이션에서 사용되는 U-shaped 구조를 시공간 스트림에 적용하여 "가장 두드러진(prominent)" 특징만을 분리해내는 것이다.

주요 기여 사항은 다음과 같다:

1. **USleepNet 모듈 개발**: Temporal prominence network와 Spatial prominence network로 구성된 U-shaped 구조를 통해, 다채널 특성 수면 파형과 핵심 공간 뇌 네트워크를 동시에 추출한다.
2. **Spatial-Temporal Graph Construction 모듈 설계**: raw 신호를 시공간 그래프로 변환함으로써 뇌의 시공간 결합 패턴을 모델링한다.
3. **SOTA 성능 달성**: 세 가지 공개 수면 데이터셋에서 기존 베이스라인 모델들을 능가하는 최고 수준의 성능을 입증하였다.
4. **해석 가능성 제공**: 모델 시각화를 통해 각 수면 단계별로 추출된 특징(파형, 뇌 네트워크, 결합 패턴)이 실제 생리학적 특성과 일치함을 보여주어 모델의 판단 근거를 제시하였다.

## 📎 Related Works

기존의 수면 단계 분류는 Random Forest나 SVM과 같은 전통적인 머신러닝 방법론에 의존하였으나, 이는 전문가의 수작업 특징 추출(manual feature crafting)이 필요하다는 단점이 있었다. 이를 해결하기 위해 CNN, RNN, Transformer 기반의 딥러닝 모델들이 제안되었다.

1. **Temporal Modeling**: CNN 기반의 U-time이나 SalientSleepNet 등이 제안되었으나, 이들은 주로 단일 채널 신호나 추상적인 시간적 특징에 집중하여 공간적 의존성을 간과하였다.
2. **Spatial Modeling**: GraphSleepNet과 같이 Graph Convolutional Network(GCN)를 적용한 연구들이 등장하였으나, 이들은 주로 전역적인 공간 특징 캡처에 치중하였으며, 정밀한 '핵심 공간 네트워크(salient spatial brain networks)' 추출이나 시공간 결합 패턴의 모델링에는 한계가 있었다.

ST-USleepNet은 이러한 한계를 극복하기 위해 raw 신호를 기반으로 시공간 결합 패턴을 명시적으로 모델링하고, U-Net 구조를 통해 핵심 특징만을 정교하게 추출하는 방식을 취함으로써 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. Spatial-Temporal Graph Construction

이 모듈은 다채널 raw 신호를 시공간 그래프 $G_s = (V, A, X)$로 변환한다. 여기서 $V$는 노드 집합, $A$는 인접 행렬, $X$는 특징 행렬이다. 신호를 $n$개의 패치(patch)로 나누어 각 패치가 그래프의 노드가 된다.

인접 행렬 $A$는 단순한 유사도가 아니라, 다음 세 가지 요소의 곱으로 정의되어 시공간적 제약 조건을 반영한다:
$$A = S \cdot M_t \cdot M_p$$

- **Similarity Matrix ($S$):** 두 노드 간의 코사인 유사도를 계산하며, 임계값 $\delta$를 넘는 경우에만 엣지를 생성한다.
$$s_{ij} = \begin{cases} \text{sim}_{\cos}(X_i, X_j) & \text{if } \text{sim}_{\cos}(X_i, X_j) > \delta \\ 0 & \text{otherwise} \end{cases}$$
- **Temporal Weighted Mask ($M_t$):** 시간적으로 가까운 패치일수록 높은 가중치를 부여한다.
$$mt_{ij} = \delta_t^{|t_i - t_j|}$$
- **Positional Weighted Mask ($M_p$):** 동일 채널 내의 노드보다 서로 다른 채널 간의 노드에 페널티($\delta_p$)를 부여하여 채널 내 응집성을 높인다.
$$mp_{ij} = \begin{cases} \delta_p & \text{if } c_i \neq c_j \\ 1 & \text{otherwise} \end{cases}$$

### 2. U-shaped Sleep Network (USleepNet)

USleepNet은 Temporal stream과 Spatial stream이라는 두 개의 상호 연결된 네트워크로 구성된다.

#### (1) Temporal Prominence Network

- **Temporal Encoder ($T_{en}$):** 1D Convolution 레이어와 Max Pooling을 사용하여 다중 스케일의 시간적 특징을 추출하고 차원을 축소한다.
- **Temporal Decoder ($T_{de}$):** Transposed 1D Convolution과 Skip Connection을 사용하여 특징을 복원하며, 이를 통해 핵심 수면 파형을 세그멘테이션 하듯 분리한다.
- **Segment Classifier ($T_c$):** Softmax와 Mean Pooling, Fully Connected layer를 통해 최종 수면 단계를 분류한다.

#### (2) Spatial Prominence Network

- **Spatial Encoder ($S_{en}$):** GCN 레이어와 Graph Pooling layer($gPool$)를 사용한다. $gPool$은 학습 가능한 투영 벡터 $P$를 이용해 노드의 중요도 점수 $i_c$를 계산하고, 상위 $k$개의 노드만을 선택하여 서브그래프를 생성한다.
- **Spatial Decoder ($S_{de}$):** Graph Unpooling($gUnpool$)과 GCN 레이어를 통해 압축된 서브그래프를 다시 원래 크기의 슈퍼그래프로 복원하며, 핵심 공간 뇌 네트워크를 식별한다.

#### (3) Spatial-Temporal Block Fusion

전체 구조는 $l$개의 ST-Encoder($T_{en} \rightarrow S_{en}$), Bottom Block($ST_b$), $l$개의 ST-Decoder($S_{de} \rightarrow T_{de}$) 순으로 구성된다. 인코더 단계에서 $T_{en}$을 먼저 배치하는 이유는 high-resolution 신호의 노이즈를 먼저 제거하고 차원을 축소하여 이후의 복잡한 그래프 연산 비용을 줄이기 위함이다.

## 📊 Results

### 실험 설정

- **데이터셋**: ISRUC-S1 (100명), ISRUC-S3 (10명), MASS-SS3 (62명)의 3가지 공개 데이터셋 사용.
- **분류 대상**: Wake, N1, N2, N3, REM의 5개 단계.
- **평가 지표**: Accuracy (Acc), F1-score.
- **검증 방법**: 10-fold cross-validation.

### 정량적 결과

ST-USleepNet은 모든 데이터셋에서 기존 SOTA 모델들을 능가하였다. 특히 ISRUC-S3 데이터셋에서 가장 큰 성능 향상을 보였으며, 가장 강력한 베이스라인인 GraphSleepNet 대비 Accuracy가 1.46% 상승하였다.

| Model | ISRUC-S1 (Acc/F1) | ISRUC-S3 (Acc/F1) | MASS-SS3 (Acc/F1) |
| :--- | :---: | :---: | :---: |
| GraphSleepNet | 76.35 / 76.67 | 77.97 / 77.73 | 84.86 / 84.27 |
| **ST-USleepNet** | **78.28 / 77.95** | **79.11 / 78.90** | **85.36 / 84.88** |

### 분석 및 검증

- **Ablation Study**: Temporal 네트워크(-T), Spatial 네트워크(-S), 혹은 시공간 그래프 구축 모듈(-ST) 중 어느 하나라도 제거했을 때 성능이 유의미하게 하락하였다. 이는 세 구성 요소가 상호 보완적으로 작동함을 의미한다.
- **Hyperparameter 분석**: 인코더에서는 커널 크기와 차원을 줄이고, 디코더에서는 늘리는 전략이 최적이었으며, 네트워크 깊이는 4층일 때 가장 높은 성능을 보였다.
- **시각화**:
  - **Temporal**: Wake의 $\alpha$파, N2의 Spindle, N3의 $\delta$파 등을 정확히 포착함을 확인하였다.
  - **Spatial**: 수면 단계별로 활성화되는 뇌 영역의 변화(예: N3에서 뇌 활동의 급격한 감소)를 효과적으로 추출하였다.
  - **Coupling**: 시공간 그래프를 통해 시간 흐름에 따른 뇌 영역 간 상호작용 패턴을 시각화하였다.

## 🧠 Insights & Discussion

본 논문은 단순한 분류 성능 향상을 넘어, 수면 단계 분류에 있어 **'무엇이 중요한 특징인가'**에 대한 구조적인 접근을 시도하였다.

**강점:**

- **구조적 정당성**: U-Net의 세그멘테이션 개념을 수면 신호에 도입하여, 불필요한 배경 정보(노이즈)를 제거하고 핵심 파형과 핵심 네트워크만을 추출하려 한 시도가 매우 효과적이었다.
- **시공간 통합**: 단순한 concatenation이 아니라, 인접 행렬 설계 단계부터 시간적/공간적 제약을 부여한 그래프를 구축함으로써 뇌의 생리학적 특성을 모델에 내재화하였다.
- **해석 가능성**: 딥러닝의 블랙박스 문제를 해결하기 위해, 시공간 그래프의 시각화를 통해 임상적으로 유의미한 수면 특징들을 추출하고 있음을 증명하였다.

**한계 및 논의:**

- **계산 복잡도**: 그래프 생성 및 GCN 연산은 일반적인 CNN보다 연산 비용이 높다. 비록 $T_{en}$을 통해 차원을 축소하였으나, 실시간 진단 시스템에 적용하기 위해서는 추론 속도에 대한 분석이 추가적으로 필요할 것이다.
- **데이터 의존성**: ISRUC-S3와 같이 소규모 데이터셋에서도 높은 성능 향상을 보였으나, 더 다양한 인구통계학적 특성을 가진 대규모 데이터셋에서의 일반화 성능 검증이 필요하다.

## 📌 TL;DR

ST-USleepNet은 다채널 수면 신호를 **시공간 그래프(Spatial-Temporal Graph)**로 변환하고, **U-shaped 구조의 시공간 스트림**을 통해 핵심 수면 파형과 뇌 네트워크를 동시에 추출하는 프레임워크이다. 이 모델은 기존의 시간/공간 단일 접근법을 넘어 **시공간 결합 패턴**을 학습함으로써 SOTA 성능을 달성하였으며, 시각화를 통해 높은 해석 가능성을 입증하였다. 향후 다채널 생체 신호를 이용한 정밀 의료 진단 및 뇌 활동 분석 연구에 중요한 기반이 될 것으로 기대된다.
