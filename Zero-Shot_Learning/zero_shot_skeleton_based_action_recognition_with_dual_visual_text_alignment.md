# Zero-Shot Skeleton-based Action Recognition with Dual Visual-Text Alignment

Jidong Kuang, Hongsong Wang, Chaolei Han, Yang Zhang, Jie Gui (2025)

## 🧩 Problem to Solve

본 논문은 스켈레톤 기반의 Zero-Shot Action Recognition (ZSAR)에서 발생하는 시각적 특징(Visual features)과 텍스트 시맨틱 벡터(Semantic vectors) 사이의 정렬 문제를 해결하고자 한다. Zero-Shot 학습의 핵심은 학습 과정에서 보지 못한(unseen) 클래스의 동작을 인식하기 위해 시각적 공간과 텍스트 공간을 효과적으로 연결하는 것이다.

기존 방법론들은 시각적 특징을 텍스트 공간으로 직접 투영(Direct projection)하거나 두 모달리티 간의 공유 임베딩 공간을 학습하는 방식을 사용하였다. 그러나 단순 투영은 두 모달리티 사이의 거대한 간극(Semantic gap)을 메우기에 부족하며, 강건하고 판별력 있는 공유 임베딩 공간을 설계하는 것은 매우 어려운 과제이다. 특히 RGB 비디오 데이터와 달리 스켈레톤 데이터는 구조적 특성이 강해 텍스트와의 정렬이 더욱 까다롭다. 따라서 본 연구의 목표는 시각적-텍스트 정렬을 강화하는 이중 구조의 네트워크를 통해 unseen 클래스에 대한 일반화 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Dual Visual-Text Alignment (DVTA)** 프레임워크를 통해 시각적 특징과 텍스트 특징을 다각도로 정렬하는 것이다.

1. **이중 정렬 전략 (Dual Alignment Strategy):** 단순한 단일 정렬이 아니라, 초기 정렬을 담당하는 Direct Alignment (DA)와 분포 정렬을 강화하는 Augmented Alignment (AA)를 결합하여 모달리티 간의 연결성을 강화한다.
2. **시맨틱 설명 강화 (Semantic Description Enhancement, SDE):** 단순한 클래스 라벨 대신, GPT-3를 이용해 생성한 동작의 맥락적 설명(Contextual descriptions)을 추가하고 Cross-attention 메커니즘을 통해 시각적 정보가 통합된 강화된 텍스트 특징을 생성한다.
3. **심층 시각 투영기 (Deep Visual Projector):** 단순 선형 층이 아닌 심층 신경망 구조의 투영기를 설계하여 스켈레톤 특징을 시맨틱 공간으로 더 효과적으로 매핑한다.
4. **LeakySigmoid 활성화 함수:** AA 모듈 내에서 유사도 점수를 정규화하기 위해, vanishing gradient 문제를 완화하고 학습 속도를 높이는 LeakySigmoid 함수를 도입하였다.

## 📎 Related Works

### 관련 연구 및 한계

- **Zero-Shot Action Recognition:** 주로 시각적 공간과 시맨틱 공간을 정렬하는 연구가 진행되어 왔으며, 최근에는 GAN을 이용해 unseen 클래스의 특징을 생성하는 방식 등이 제안되었다. 하지만 대부분 RGB 데이터를 기반으로 하여 배경 노이즈에 취약하다는 한계가 있다.
- **Skeleton-Based Action Recognition:** ST-GCN, Shift-GCN, CTR-GCN 등 Graph Convolutional Network (GCN) 기반의 연구들이 스켈레톤의 시공간적 특징 추출에서 뛰어난 성능을 보였다.
- **Zero-Shot Skeleton Action Recognition:** DeViSE나 SMIE와 같은 방법들이 제안되었으나, SMIE와 같은 최근 연구들은 원본 추출 특징에 직접 연결 모델을 적용하여 두 공간의 근본적인 차이를 간과하는 경향이 있다. 또한 일부 연구(PURLS, STAR)는 방대한 양의 수동 텍스트 정보를 필요로 하여 실제 적용에 어려움이 있다.

### 기존 방식과의 차별점

DVTA는 방대한 수동 텍스트 준비 대신 최소한의 라벨 설명과 GPT-3 기반의 자동 생성을 사용하며, 구조적으로는 심층 투영기와 이중 정렬 네트워크를 통해 모달리티 간의 간극을 더 정교하게 좁힌다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인

DVTA는 크게 **Visual-Text Embedding**, **Direct Alignment (DA)**, **Augmented Alignment (AA)** 세 단계로 구성된다. 스켈레톤 특징과 텍스트 특징은 각각의 투영기를 통해 공통 임베딩 공간으로 매핑되며, SDE를 통해 강화된 텍스트 특징이 생성된다. 이후 DA와 AA가 동시에 작동하여 두 모달리티를 정렬하며, 최종적으로 KL divergence 손실 함수를 통해 전체 네트워크를 최적화한다.

### 주요 구성 요소 및 상세 설명

#### 1. Visual-Text Embedding 및 Deep Feature Projection

스켈레톤 특징 $v$와 텍스트 특징 $t$는 각각 $L_2$ 정규화를 거친 후 투영 네트워크에 입력된다.

- **Skeleton Projector ($F_s$):** 스켈레톤 특징을 시맨틱 공간의 차원 $h$로 매핑하기 위해 학습 가능한 심층 비선형 매핑 투영기를 사용한다.
- **Text Projector ($F_t$):** 텍스트 특징의 풍부함을 보존하기 위해 단순 선형 층을 사용한다.
- 결과적으로 임베딩된 특징은 다음과 같다:
$$v_e = F_s(v), \quad t_e = F_t(t)$$

#### 2. Semantic Description Enhancement (SDE)

동작 라벨에 대한 맥락적 설명을 GPT-3로 생성하여 $t_{cont}$를 얻는다. 이후 Cross-attention 메커니즘을 통해 시각적 특징 $v_e$를 쿼리($Q$)로, 라벨 특징 $t_e$와 맥락 특징 $t_{cont}$의 결합을 키($K$)와 값($V$)으로 사용하여 강화된 텍스트 특징 $t_{aug}$를 생성한다.
$$t_{aug} = \text{softmax}\left(\frac{QK^T}{\sqrt{h}}\right)V$$
여기서 $K, V = \text{stack}(t_e, t_{cont}) \in \mathbb{R}^{2 \times h}$이다.

#### 3. Direct Alignment (DA)

DA는 스켈레톤 특징과 텍스트 특징 간의 코사인 유사도를 계산하여 직접적으로 정렬한다. 하나의 라벨에 여러 스켈레톤 샘플이 존재하는 특성을 고려하여, Cross-Entropy 대신 KL divergence를 사용하여 다수의 긍정 샘플(positive examples)을 학습한다.
$$p_{v2t}^1(v_i) = \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{B} \exp(\text{sim}(v_j, t_j)/\tau)}$$
여기서 $\tau$는 학습 가능한 온도 파라미터이며, 손실 함수 $L_1$은 텍스트-to-스켈레톤 및 스켈레톤-to-텍스트 방향의 KL divergence 합으로 정의된다.

#### 4. Augmented Alignment (AA)

AA는 Deep Metric Network (DMN)를 사용하여 두 특징의 연결(concatenation)을 입력받아 유사도 점수 $G(v, t)$를 예측한다. 이때 **LeakySigmoid** 함수를 사용하여 출력값을 $[0, 1]$ 범위로 정규화한다.
$$\text{LeakySigmoid}(x) = \begin{cases} \frac{1}{1+\exp(-x)}, & \text{if } x > 0 \\ \gamma \cdot \exp(\gamma \cdot x), & \text{if } x \leq 0 \end{cases}$$
$\gamma$는 음수 입력 시의 기울기를 조절하여 vanishing gradient 문제를 방지한다.

#### 5. 최종 학습 목표 및 추론

최종 유사도 점수는 DA의 점수 $p^1$과 AA의 점수 $p^2$의 평균으로 계산된다:
$$p_{v2t} = \frac{p_{v2t}^1 + p_{v2t}^2}{2}$$
최종 손실 함수 $L$은 이 통합된 점수와 실제 정답(Ground Truth) 사이의 KL divergence를 최소화하는 방향으로 학습된다. 테스트 시에는 입력 스켈레톤과 각 unseen 클래스의 텍스트 특징 간의 유사도를 측정하여 가장 높은 클래스를 선택한다.

## 📊 Results

### 실험 설정

- **데이터셋:** NTU RGB+D 60, NTU RGB+D 120, PKU-MMD.
- **백본:** 시각적 특징 추출기(Shift-GCN, ST-GCN, CTR-GCN), 텍스트 특징 추출기(Sentence-BERT, CLIP).
- **평가 지표:** Accuracy (%).
- **비교 대상:** DeViSE, RelationNet, ReViSE, JPoSE, CADA-VAE, SynSE, SMIE, PURLS.

### 주요 결과

- **SOTA 달성:** DVTA는 모든 벤치마크 데이터셋에서 기존 방법론보다 우수한 성능을 보였다. 특히 NTU-120의 96/24 split에서 SMIE 대비 6.51%p 상승한 51.81%의 정확도를 기록하였다.
- **SDE의 효과:** SDE를 제거한 모델(DVTA w/o SDE)보다 성능이 높게 나타났으며, 특히 "waving", "punching"과 같은 손 관련 동작의 인식률이 크게 향상되었다.
- **투영기의 중요성:** PCA 시각화 결과, Deep Visual Projector를 사용했을 때 unseen 클래스 간의 특징 분포가 더 명확하게 구분됨이 확인되었다.
- **LeakySigmoid의 효율성:** 일반 Sigmoid나 활성화 함수가 없는 경우보다 수렴 속도가 빠르고 최종 정확도가 높음을 확인하였다.
- **계산 효율성:** PURLS보다 훨씬 적은 파라미터와 FLOPs를 가지면서도 더 높은 FPS를 유지하여 실시간 적용 가능성을 보여주었다.

## 🧠 Insights & Discussion

### 강점 및 해석

- **보완적 정렬:** DA가 거시적인 특징 정렬을 수행한다면, AA는 Deep Metric Learning을 통해 미세한 분포 차이를 조정함으로써 서로 보완적인 역할을 수행한다.
- **컨텍스트의 힘:** 단순 라벨 임베딩은 동작의 세부 특성을 담기 어려우나, GPT-3를 통한 맥락 설명과 Cross-attention을 결합함으로써 시각적 특징과 더 밀접하게 연결된 시맨틱 표현을 생성할 수 있었다.
- **수학적 정교함:** LeakySigmoid의 도입은 단순히 성능 향상을 넘어, 딥러닝 학습의 고질적인 문제인 기울기 소실을 해결하여 학습 안정성을 높였다는 점에서 의미가 있다.

### 한계 및 비판적 해석

- **특징 추출 의존성:** 본 논문은 pre-extracted features를 사용하므로, 특징 추출 단계에서 발생하는 정보 손실을 복구할 수 없다. 저자 역시 이를 한계로 언급하며 향후 end-to-end 학습의 필요성을 제시하였다.
- **GPT-3 의존성:** SDE를 위해 외부 LLM을 사용하는데, 이는 추론 시에는 문제가 없으나 학습 데이터 준비 단계에서 외부 API나 모델에 의존해야 한다는 제약이 있다.

## 📌 TL;DR

본 논문은 스켈레톤 기반 Zero-Shot 동작 인식의 핵심 난제인 시각-텍스트 간의 semantic gap을 해결하기 위해 **Dual Visual-Text Alignment (DVTA)** 프레임워크를 제안한다. 이 모델은 **심층 투영기**, **LLM 기반 맥락 강화(SDE)**, 그리고 **이중 정렬 구조(DA+AA)**를 통해 unseen 클래스에 대한 일반화 성능을 획기적으로 높였다. 특히 NTU-60/120 및 PKU-MMD 데이터셋에서 SOTA 성능을 달성하였으며, 이는 향후 오픈셋(open-set) 동작 인식이나 실시간 휴먼-컴퓨터 인터랙션 시스템의 기반 기술로 활용될 가능성이 높다.
