# Multi-Sample Dynamic Time Warping for Few-Shot Keyword Spotting

Kevin Wilkinghoff, Alessia Cornaggia-Urrigshardt (2024)

## 🧩 Problem to Solve

본 논문은 소수의 학습 샘플만으로 특정 키워드를 탐지하는 **Few-Shot Keyword Spotting (KWS)** 상황에서 발생하는 연산 효율성과 탐지 성능 사이의 트레이드-오프 문제를 해결하고자 한다.

일반적으로 시계열 데이터의 유사도를 측정하기 위해 **Dynamic Time Warping (DTW)** 또는 **Sub-sequence DTW**가 사용된다. 하지만 각 클래스(키워드)당 $K$개의 샘플이 있을 때, 모든 샘플을 개별적으로 쿼리하는 방식은 연산 복잡도가 $O(N \cdot M \cdot C \cdot K)$로 증가하여(여기서 $N, M$은 시퀀스 길이, $C$는 클래스 수), 샘플 수가 많아질수록 실시간 처리가 불가능해지는 문제가 있다.

이를 해결하기 위해 여러 샘플의 평균치인 **Fréchet mean**을 단일 템플릿으로 사용하는 방법이 제안되었으나, 이는 연산 복잡도를 $O(N \cdot M \cdot C)$로 낮추는 대신 각 샘플이 가진 다양성(Variability)을 충분히 반영하지 못해 탐지 성능이 크게 저하되는 한계가 있다. 따라서 본 논문의 목표는 **개별 샘플을 모두 사용하는 것과 유사한 성능을 유지하면서도, 연산 시간을 획기적으로 줄일 수 있는 새로운 DTW 프레임워크를 구축하는 것**이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Multi-sample DTW**라는 새로운 접근 방식을 제안한 것이다. 이 방법의 중심 직관은 모든 개별 샘플에 대해 DTW 경로를 각각 찾는 대신, **클래스별로 통합된 Cost Tensor를 생성하고 이를 단일 Cost Matrix로 변환하여 단 한 번의 DTW 경로 탐색만으로 다중 샘플의 특성을 모두 고려**하는 것이다.

또한, 기존의 표준 Fréchet mean보다 더 많은 변동성을 캡처할 수 있는 **Altered Fréchet Mean** 생성 절차를 제안하여 성능을 추가적으로 향상시켰다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구와 그 한계를 언급한다.

1. **DTW 및 Sub-sequence DTW**: 시계열 데이터의 시간적 편차를 허용하며 유사도를 측정하는 표준적인 방법이다. 하지만 샘플 수가 증가함에 따라 연산량이 선형적으로 증가한다.
2. **Fréchet Mean 및 DBA (DTW Barycenter Averaging)**: 여러 시퀀스의 평균을 구해 대표 템플릿을 만드는 방법이다. 연산 효율은 높지만, 개별 샘플들의 다양성이 소실되어 성능이 떨어진다.
3. **Cluster-based Means**: 샘플들을 클러스터링하여 각 클러스터의 평균을 사용하는 방법이 제안되었으나, 여전히 여러 개의 템플릿이 필요하며 다양성을 완전히 보존하지 못한다.
4. **Discriminative Embeddings (TACos)**: 최근에는 신경망을 통해 DTW에 적합한 임베딩을 학습하여 성능을 높이는 추세이며, 본 논문에서도 이를 피처로 사용한다.

## 🛠️ Methodology

Multi-sample DTW의 전체 파이프라인은 다음과 같은 4단계로 구성된다.

### 1. Altered Fréchet Mean 계산 (Step 1)

먼저 각 클래스에 대해 기준이 되는 **Reference Template**을 생성한다.

- 표준 DBA(DTW Barycenter Averaging) 알고리즘을 통해 기본 Fréchet mean을 구한다.
- 이후, 비표준 단계인 $(1,1), (1,2), (2,1)$ 스텝을 적용하여 DBA를 한 번 더 수행한다.
- 이 과정은 템플릿 내에 서로 다른 두 가지 대안적 패턴을 교차로 배치하게 하여, 단일 템플릿임에도 더 많은 변동성을 담을 수 있게 한다.

### 2. 샘플의 시간 차원 변환 (Step 2)

모든 쿼리 샘플들을 앞서 구한 Reference Template과 동일한 시간 차원(길이)을 갖도록 변환한다.

- 각 쿼리 샘플을 Reference Template과 매칭시키고, 매칭된 지점들의 산술 평균으로 템플릿의 각 엔트리를 대체하는 DBA의 단일 반복 과정을 수행한다.

### 3. Cost Tensor 생성 및 Matrix 변환 (Step 3)

이 단계가 본 논문의 핵심인 연산 최적화 과정이다.

- **Cost Tensor 생성**: 테스트 샘플(Target sequence)과 변환된 모든 쿼리 샘플들 간의 로컬 거리(Cosine distance)를 계산하여 3차원 텐서를 생성한다.
- **Cost Matrix 변환**: 생성된 텐서의 샘플 차원에 대해 **원소별 최소값(Element-wise minimum)**을 취하여 2차원 Cost Matrix로 압축한다.
- 수식적으로 표현하면, 테스트 샘플 $T$와 $K$개의 쿼리 샘플 $S_k$ 사이의 비용 행렬 $C_k$들에 대해 다음과 같이 통합 행렬 $C_{final}$을 구한다.
  $$C_{final}(i, j) = \min_{k \in \{1, \dots, K\}} C_k(i, j)$$
- 이 과정을 통해 DTW 경로가 서로 다른 쿼리 샘플들의 비용 행렬 사이를 유연하게 오갈 수 있게 된다.

### 4. DTW 적용 (Step 4)

최종적으로 변환된 단일 Cost Matrix에 대해 표준 Sub-sequence DTW를 적용하여 클래스별 유사도 점수를 산출한다.

### 연산 복잡도 분석 (Runtime Analysis)

- **개별 샘플 사용 시**: $O(N \cdot M \cdot C \cdot K)$의 복잡도를 가지며, $K$번의 DTW 경로 탐색(가장 비용이 큰 작업)이 필요하다.
- **Multi-sample DTW**: 전체적인 복잡도는 여전히 $O(N \cdot M \cdot C \cdot K)$이지만, **DTW 경로 탐색은 클래스당 단 한 번($O(N \cdot M \cdot C)$)**만 수행한다.
- Cost Tensor를 Matrix로 변환하는 과정은 병렬 처리가 가능하므로, 실제 실행 시간은 Fréchet mean을 사용하는 것과 유사하게 매우 빠르다.

## 📊 Results

### 실험 설정

- **데이터셋**: Few-shot open-set KWS 데이터셋인 **KWS-DailyTalk**를 사용하였다.
- **피처**:
  - **HFCCs**: 학습이 필요 없는 수작업 피처.
  - **Discriminative Embeddings**: TACos 손실 함수를 통해 학습된 신경망 기반 임베딩.
- **지표**: Micro-averaged event-based F-Score, Precision, Recall 및 실행 시간(Runtime).
- **환경**: Intel i7-8700 CPU @3.20 GHz.

### 주요 결과

- **성능 비교**:
  - **Embeddings $\gg$ HFCCs**: 모든 설정에서 신경망 기반 임베딩이 수작업 피처보다 월등한 성능을 보였다.
  - **Multi-sample DTW $\approx$ Individual Samples**: Multi-sample DTW는 모든 개별 샘플을 사용했을 때와 거의 동일하거나 때로는 더 높은 F-Score를 기록하였다.
  - **Multi-sample DTW $\gg$ Fréchet Means**: 단일 평균 템플릿을 사용하는 방식보다 훨씬 높은 성능을 보였으며, 특히 HFCCs에서 그 차이가 두드러졌다.
- **실행 시간**:
  - 실행 시간은 `Fréchet means < Multi-sample DTW (Parallelized) < Multi-sample DTW < Individual samples` 순으로 나타났다.
  - 특히 샘플 수($K$)가 증가함에 따라 개별 샘플 방식은 시간이 선형적으로 급증하지만, Multi-sample DTW는 매우 완만하게 증가하여 실용성을 입증하였다.

## 🧠 Insights & Discussion

본 연구는 DTW의 고질적인 문제인 연산량과 일반화 성능 사이의 충돌을 **"Cost 영역에서의 통합"**이라는 아이디어로 해결하였다.

**강점 및 분석**:

- **다양성 보존**: 단순히 평균을 내는 것이 아니라, 비용 행렬에서 최소값을 선택함으로써 여러 샘플 중 가장 적합한 패턴을 동적으로 선택하는 효과를 거두었다. 이는 인위적인 가상 샘플(Artificial samples)을 생성하는 것과 같은 효과를 주어 성능을 높인 것으로 분석된다.
- **실용적 효율성**: 가장 연산 비용이 높은 DTW 경로 최적화 단계를 클래스당 1회로 제한함으로써, 개별 샘플 방식의 정확도와 Fréchet mean 방식의 속도를 동시에 잡았다.

**한계 및 논의**:

- **메모리 사용량**: Cost Tensor를 생성해야 하므로, 샘플 수 $K$가 극단적으로 많아질 경우 메모리 부하가 발생할 가능성이 있다.
- **가정**: 본 논문은 쿼리 샘플들이 동일한 클래스 내에서 어느 정도의 일관성을 가진다는 전제하에 $\min$ 연산을 수행한다. 만약 클래스 내 변동성이 너무 크다면 $\min$ 연산이 노이즈를 선택할 위험이 있다.

## 📌 TL;DR

본 논문은 Few-shot Keyword Spotting에서 다수의 쿼리 샘플을 효율적으로 처리하기 위한 **Multi-sample DTW**를 제안한다. 이 방법은 각 샘플의 비용 행렬을 3차원 텐서로 쌓은 뒤 원소별 최소값을 취해 단일 행렬로 압축함으로써, **연산 속도는 단일 템플릿 방식(Fréchet mean)에 근접시키면서 탐지 성능은 모든 샘플을 개별적으로 사용하는 수준으로 유지**한다. 결과적으로 저자원 환경에서의 실시간 키워드 탐지 시스템 구축에 중요한 기여를 할 수 있는 연구이다.
