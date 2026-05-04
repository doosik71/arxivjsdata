# Vertical Semi-Federated Learning for Efficient Online Advertising

Wenjie Li, Shu-Tao Xia, Jiangke Fan, Teng Zhang, Xingxing Wang (2025)

## 🧩 Problem to Solve

본 논문은 전통적인 수직 연합 학습(Vertical Federated Learning, VFL)이 실제 산업 현장, 특히 온라인 광고 시스템에 적용될 때 발생하는 두 가지 핵심적인 한계를 해결하고자 한다.

첫째는 **중첩 샘플(overlapped samples)에 국한된 적용 범위**이다. 기존 VFL은 두 파티(Active party A와 Passive party B) 모두가 공통으로 보유한 유저 데이터에 대해서만 학습 및 추론이 가능하다. 그러나 실제 환경에서 중첩 유저의 비중은 전체의 일부에 불과하며, 이는 학습 데이터의 풍부함을 제한하고 비중첩 유저(non-overlapped users)가 연합 학습의 이점을 전혀 누리지 못하게 만든다.

둘째는 **실시간 연합 추론(federated serving)의 높은 시스템 비용**이다. VFL의 분산 구조는 추론 시 파티 간 데이터 전송으로 인한 추가 지연 시간(latency)을 발생시킨다. 수백만 QPS(Queries Per Second)와 10~100ms 수준의 엄격한 실시간 응답 속도를 요구하는 광고 시스템에서 이러한 오버헤드는 실무적으로 수용 불가능한 수준이다.

따라서 본 논문의 목표는 모든 라벨링된 데이터를 활용하여 학습하면서도, 추론 시에는 외부 통신 없이 독립적으로 수행 가능한 **Semi-VFL(Vertical Semi-Federated Learning)** 설정과 이를 구현하기 위한 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 연합 학습의 이점을 유지하면서 로컬 추론을 가능하게 하는 **Joint Privileged Learning (JPL)** 프레임워크를 제안한 것이다. JPL의 중심 아이디어는 다음과 같다.

1. **두 가지 경로(Two-branch) 아키텍처**: 로컬 필드(Active party A)의 귀납적 편향(inductive bias)을 학습하는 'Local branch'와 연합 모델의 지식을 증류(distill)하는 'Federated branch'를 동시에 운영하여 상호 보완적인 학습을 수행한다.
2. **Federated Equivalence Imitation**: 수동 파티(Passive party B)의 특성(feature)이 부재한 상황을 해결하기 위해, 로컬 특성을 B-side 특성으로 변환하는 매핑을 학습시키고, 이를 로짓(logit) 수준과 특성(feature) 수준에서 모방하게 하여 B-side의 지식을 로컬 모델로 전이한다.
3. **Cross-Head Rank Alignment**: 서로 다른 두 브랜치의 예측 헤드(head) 간에 랭킹 일관성을 강제함으로써, 중첩 및 비중첩 샘플 모두에서 일관된 성능 향상을 달성한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들의 한계를 지적하며 차별점을 제시한다.

- **기존 VFL 및 증류 기반 접근 방식(FPD 등)**: 일부 연구들이 특권 증류(privileged distillation)를 통해 로컬 추론을 시도했으나, 이는 주로 직접적인 로짓 증류(logit distillation)에 의존하며 파티 간의 포괄적인 지식 전이가 부족하다는 한계가 있다.
- **비정렬 데이터 활용 연구(FedUD, FedCVT)**: 비중첩 샘플을 통합하거나 잠재 공간에서 B-side 특성을 보간(imputation)하여 연합 학습 성능을 높이려는 시도가 있었으나, 이들은 주로 연합 추론 성능 향상에 집중되어 있으며 로컬 추론을 지원하도록 설계되지 않았다.

JPL은 이러한 한계를 넘어, 로컬 추론이라는 제약 조건 하에서 연합 지식을 효과적으로 전이하고 전체 샘플 공간(중첩+비중첩)에 적응하는 것을 목표로 한다.

## 🛠️ Methodology

JPL 프레임워크는 크게 2단계 파이프라인으로 구성된다. 먼저 중첩 데이터셋으로 연합 교사 모델(Federated teacher model)을 사전 학습시키고, 이후 전체 샘플 공간에서 학생 모델(Student model)을 학습시키는 방식이다.

### 1. 전체 아키텍처

학생 모델은 두 개의 브랜치로 구성된다.

- **Local Branch**: Active party A의 필드만을 사용하여 로컬의 귀납적 편향을 학습한다.
- **Federated Branch**: 교사 모델로부터 지식을 증류하며, $f_{A \to B}(\cdot)$라는 변환 인코더를 통해 A-side 특성을 B-side 특성으로 모방한다.

### 2. Federated Equivalence Imitation

B-side 특성이 없는 상황을 극복하기 위해 두 가지 수준의 모방을 수행한다.

**가. Logit Imitation (로짓 모방)**
비중첩 샘플 $x_{unA}$에 대해 모방된 특성을 $\tilde{u}_B = f_{A \to B}(x_{unA})$라 할 때, 다음과 같이 이진 교차 엔트로피 손실($CE$)을 통해 판별 능력을 강화한다.
$$L_{ce\_dmi}(x_{unA}) = CE(y, g_{Fed}(f_{T\_A}(x_{unA}), \tilde{u}_B)) + CE(y, g_B(\tilde{u}_B))$$
여기서 $g_{Fed}$는 교사의 동결된 분류기, $g_B$는 학생의 보조 분류기이다. 중첩 샘플의 경우, 교사의 예측값 $\hat{y}_T$와 학생의 예측값 간의 KL-divergence($KL$)를 최소화하여 일관성을 맞춘다.

**나. Feature Imitation (특성 모방)**
특성을 직접 복구하는 대신, 샘플 간의 상대적 유사성을 유지하는 전략을 취한다. 중첩 샘플 배치 $\mathbf{X}_A$에 대해 유사도 행렬 $C$를 정의한다.
$$C = \tilde{H}_B \odot (H_{T\_B})^\top - H_{T\_B} \odot (H_{T\_B})^\top$$
여기서 $\odot$는 행 단위 $L_2$ 정규화 후의 행렬 곱을 의미하며, 손실 함수 $L_{B\_dmi}$는 이 행렬 $C$의 대각 성분과 비대각 성분의 오차를 최소화하는 방향으로 정의된다. 비중첩 샘플에 대해서는 교사 모델의 특성을 앵커(anchor)로 사용하여 $A$ 공간과 $B$ 공간의 유사도 격차를 줄이는 $L_{AB\_dmi}$를 적용한다.

### 3. Cross-Head Rank Alignment (PRC)

두 브랜치의 예측 결과를 융합하기 위해 **Privileged Ranking Consistency (PRC)** 손실을 도입한다. 예측값 $\hat{y}$의 랭킹을 부분 순서 행렬(Partial Order Matrix, POM) $R$로 표현하며, $R_{ij} = \sigma(\hat{y}_i - \hat{y}_j)$로 정의한다.

중첩 샘플에 대해 로컬 헤드($R_A$)와 연합 헤드($R_{Fed}$)의 랭킹 일관성을 맞추는 손실 함수는 다음과 같다.
$$L_{rank A \leftarrow Fed} = \frac{\|R_{++}^A - \text{sg}(R_{++}^{Fed})\|_F}{\|\text{sg}(R_{++}^{Fed})\|_F} + \frac{\|R_{--}^A - \text{sg}(R_{--}^{Fed})\|_F}{\|\text{sg}(R_{--}^{Fed})\|_F} - \|R_{+-}^A\|_F$$
여기서 $\text{sg}(\cdot)$는 stop-gradient 연산이다. 최종 예측값 $\hat{y}$는 두 헤드의 로짓 평균을 통해 산출된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Criteo 및 Avazu CTR 예측 데이터셋을 사용하여 시뮬레이션 환경 구축.
- **비교 대상 (Baselines)**: Local(A-side 전용), FPD(특권 증류), FedUD(비정렬 데이터 활용), FedCVT(특성 보간).
- **평가 지표**: AUC $\uparrow$, Logloss $\downarrow$. 모든 모델은 **로컬 추론 모드**에서 평가되었다.

### 2. 주요 결과

- **전체 성능**: JPL은 Avazu와 Criteo 모든 데이터셋에서 Local 및 다른 베이스라인 모델들을 압도하는 성능을 보였다. 특히 Avazu의 중첩 데이터셋에서는 연합 교사 모델(Fed)보다 더 높은 성능을 기록하여, 샘플 공간과 필드 공간의 이점을 모두 통합했음을 입증했다.
- **소거 연구 (Ablation Study)**: 로짓 모방(Logit Imitation)의 제거가 가장 큰 성능 하락을 야기했으며, 그 뒤를 랭킹 정렬(Rank Alignment)과 특성 모방(Feature Imitation)이 이었다. 이는 로짓 모방이 가장 직접적인 감독 신호를 제공하며, 랭킹 정렬이 두 브랜치의 보완적 정보를 효과적으로 융합함을 시사한다.
- **강건성 확인**: 비중첩 데이터의 양을 변화시키거나 Passive party의 필드 수를 조절해도 JPL의 상대적 우위는 일정하게 유지되었다. 또한 DCN, DeepFM 등 다양한 백본 아키텍처에서도 일관된 성능 향상을 보였다.

## 🧠 Insights & Discussion

본 논문은 실무적인 제약 사항인 '실시간성'과 '데이터 부족'을 해결하기 위해 **Semi-VFL**이라는 새로운 설정을 정의하고 이를 성공적으로 구현했다.

**강점**:

- 기존 VFL의 치명적 약점인 추론 지연 시간을 완전히 제거하면서도, 연합 학습의 지식을 로컬 모델에 성공적으로 이식했다.
- 단순한 로짓 증류를 넘어 특성 수준의 유사도 모방과 랭킹 일관성 정렬이라는 정교한 메커니즘을 도입하여 지식 전이의 효율을 높였다.

**한계 및 논의사항**:

- 본 방법론은 여전히 중첩 데이터로 사전 학습된 '교사 모델'의 존재를 전제로 한다. 만약 중첩 데이터가 극단적으로 적은 경우, 교사 모델의 품질 저하가 학생 모델의 성능 하락으로 이어질 가능성이 있다.
- 특성 모방 시 직접적인 복구가 아닌 유사도 기반의 상대적 재구성을 택했는데, 이는 B-side 특성의 차원이나 분포가 A-side와 극명하게 다를 때 어느 정도의 정보 손실이 발생하는지에 대한 추가 분석이 필요해 보인다.

## 📌 TL;DR

이 논문은 광고 시스템의 실시간 추론 제약을 해결하기 위해, 연합 학습의 지식을 로컬 모델로 전이하여 독립 추론을 가능하게 하는 **Semi-VFL** 설정과 **JPL(Joint Privileged Learning)** 프레임워크를 제안한다. JPL은 로짓/특성 수준의 모방 학습과 브랜치 간 랭킹 정렬을 통해 비중첩 유저에게도 연합 학습의 이점을 제공하며, 실제 CTR 데이터셋에서 기존 방식 대비 우수한 성능을 입증하였다. 이는 실시간 응답이 필수적인 산업 현장에서 VFL을 실용적으로 배포할 수 있는 중요한 방법론이 될 가능성이 높다.
