# Prototype Completion for Few-Shot Learning

Baoquan Zhang, Xutao Li, Yunming Ye, and Shanshan Feng (2021)

## 🧩 Problem to Solve

본 논문은 소수의 샘플만으로 새로운 클래스를 인식해야 하는 Few-Shot Learning (FSL)의 한계를 해결하고자 한다. 기존의 많은 FSL 방법론, 특히 Pre-training 기반 방법론들은 특징 추출기(Feature Extractor)를 먼저 학습시킨 후 Meta-learning을 통해 이를 Fine-tuning 하는 방식을 채택한다. 그러나 저자들은 이러한 Fine-tuning 단계가 성능 향상에 기여하는 바가 매우 미미하다는 점에 주목하였다.

분석 결과, Pre-trained 특징 공간에서 Base 클래스들은 매우 조밀한 클러스터를 형성하는 반면, Novel 클래스들은 분산(Variance)이 매우 큰 그룹으로 퍼져 있다는 것이 확인되었다. 이는 특징 추출기를 Fine-tuning 하여 클러스터를 조밀하게 만드는 것이 Base task에 과적합(Overfitting)될 위험이 크고 실효성이 낮음을 의미한다. 따라서 본 연구의 목표는 특징 추출기의 수정보다는, 소수의 샘플이 클래스 중심에서 멀리 떨어져 있더라도 이를 보완하여 **대표성 있는 프로토타입(Representative Prototype)을 추정**하는 것에 집중하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Prototype Completion**이다. 저자들은 클래스 중심에서 벗어난 샘플들이 대개 일부 대표적인 속성(Attribute) 특징이 결여된 '불완전한(Incomplete)' 상태라는 직관에서 출발한다. 이를 해결하기 위해 외부의 기본 지식(Primitive Knowledge, 예: 클래스별 부분/속성 주석)을 도입하여, 부족한 시각적 특징을 채워 넣음으로써 보다 정확한 클래스 프로토타입을 생성하는 프레임워크를 제안한다.

## 📎 Related Works

기존 FSL 연구는 크게 Metric-based, Optimization-based, Graph-based, Semantics-based 방법론으로 나뉜다. 
- **Metric-based 및 Pre-training 기반 방법론:** 최근 연구들은 Pre-training이 성능을 크게 높인다는 것을 발견했으나, 정작 Meta-learning 단계의 Fine-tuning이 왜 효과가 적은지에 대한 분석은 부족했다.
- **Zero-Shot Learning (ZSL):** 시맨틱 공간과 시각적 공간 사이의 매핑 함수를 학습하여 샘플이 전혀 없는 상태에서 인식하는 방식이다. 본 논문은 ZSL의 속성(Attribute) 활용 아이디어를 가져오되, 소수의 샘플이 존재하는 FSL 설정에 맞게 프로토타입을 보완하는 방식으로 차별화하였다.
- **Visual Attributes:** 객체의 구성 요소 특징을 활용하는 연구들이 존재하지만, 본 논문은 이를 단순히 특징 표현을 좋게 만드는 것이 아니라 프로토타입을 '완성'시키는 전략으로 사용한다.

## 🛠️ Methodology

### 1. 전체 파이프라인
제안된 프레임워크는 총 4단계로 구성된다: **Pre-Training $\rightarrow$ Learning to Complete Prototypes $\rightarrow$ Meta-Training $\rightarrow$ Meta-Test**.

### 2. 주요 구성 요소 및 상세 설명

#### (1) Part/Attribute Transfer Network (PATNet)
Base 클래스에 존재하는 속성($A_{seen}$)은 학습 가능하지만, Novel 클래스에만 존재하는 속성($A_{unseen}$)은 시각적 특징을 직접 추출할 수 없다. PATNet은 속성의 시맨틱 임베딩(Glove 사용)과 시각적 특징 분포 사이의 관계를 학습하여, 본 적 없는 속성의 특징 분포를 추론한다.
- **입력:** 속성의 시맨틱 임베딩 $h_{a_i}$
- **출력:** 해당 속성의 특징 분포 $\mathcal{N}(\hat{\mu}_{a_i}, \text{diag}(\hat{\sigma}^2_{a_i}))$
- **학습 목표:** Seen attributes에 대해 실제 분포와 추론된 분포 사이의 KL Divergence를 최소화한다.
$$\min_{\theta_p} \mathbb{E}_{a_i \in A_{seen}} \text{KL}(\mathcal{N}(\hat{\mu}_{a_i}, \text{diag}(\hat{\sigma}^2_{a_i})), \mathcal{N}(\mu_{a_i}, \text{diag}(\sigma^2_{a_i})))$$

#### (2) Prototype Completion Network (ProtoComNet)
불완전한 프로토타입 $p_k$를 입력받아 대표성 있는 $\hat{p}_k$로 완성하는 Encoder-Aggregator-Decoder 구조의 네트워크이다.
- **Encoder:** 불완전한 프로토타입 $p_k$와 각 속성 특징 $z_{a_i}$를 저차원 잠재 코드 $z'_k, z'_{a_i}$로 인코딩한다.
- **Aggregator:** Attention 메커니즘을 통해 클래스 $k$와 속성 $a_i$ 사이의 중요도를 계산하고, 가중 합을 통해 통합된 표현 $g_k$를 생성한다.
$$\alpha^{ka_i} = R^{ka_i} g_{\theta_{ca}}(p_k || h_k || h_{a_i}), \quad g_k = \sum_{a_i} \alpha^{ka_i} z'_{a_i} + z'_k$$
- **Decoder:** $g_k$를 이용하여 최종 완성된 프로토타입 $\hat{p}_k$를 예측하며, Base 클래스의 실제 프로토타입 $p^{real}_k$와의 MSE Loss를 통해 학습한다.

#### (3) Gaussian-based Prototype Fusion Strategy (GaussFusion)
단순 평균 기반 프로토타입 $p_k$는 샘플 부족으로 편향될 수 있고, 완성된 프로토타입 $\hat{p}_k$는 외부 지식의 노이즈로 인해 오류가 있을 수 있다. 이를 보완하기 위해 베이지안 추정(Bayesian Estimation)을 통해 두 프로토타입을 융합한다.
- **수학적 원리:** $p_k$와 $\hat{p}_k$를 각각 다변량 가우시안 분포에서 샘플링된 것으로 간주하고, 두 분포의 곱(Product)을 통해 사후 분포(Posterior)를 구한다.
- **융합된 프로토타입 $\hat{\mu}'_k$:**
$$\hat{\mu}'_k = \frac{\sigma^2_k \odot \hat{\mu}_k + \hat{\sigma}^2_k \odot \mu_k}{\hat{\sigma}^2_k + \sigma^2_k}$$
여기서 $\odot$은 원소별 곱(element-wise product)을 의미한다.

#### (4) 파라미터 추정 방법 (Improved EM-based)
$\mu_k, \sigma_k$ 등의 변수를 추정하기 위해 EM 알고리즘을 확장하여 사용한다. 특히 E-step에서 가우시안 밀도 함수 대신 FSL에서 더 효과적이라고 알려진 **Cosine-based Classifier**를 사용하여 사후 확률을 계산함으로써 추정 정확도를 높였다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** miniImageNet, tieredImageNet, CUB-200-2011 (세밀한 분류 데이터셋).
- **평가 설정:** Inductive FSL (테스트 샘플 정보 사용 불가) 및 Transductive FSL (테스트 샘플의 분포 활용 가능) 모두 수행.
- **비교 대상:** Metric-based, Optimization-based, Semantics-based 및 최신 Pre-training 기반 방법론들.

### 2. 주요 결과
- **정량적 성능:** 모든 데이터셋에서 기존 SOTA 모델들보다 우수한 성능을 보였다. 특히 1-shot 태스크에서 성능 향상이 매우 두드러지는데, 이는 1-shot일 때 프로토타입 추정 오류가 가장 심각하기 때문이다.
- **프로토타입 정확도:** 실제 클래스 중심($p^{real}_k$)과의 코사인 유사도를 측정한 결과, 본 방법론의 융합된 프로토타입 $\hat{p}'_k$가 가장 높은 유사도를 기록하여 가장 대표성 있는 프로토타입을 생성함을 입증하였다.
- **강건성:** 기본 지식(Primitive Knowledge)에 인위적인 노이즈를 추가했을 때, GaussFusion을 적용한 경우 성능 저하가 매우 적어 외부 지식의 불완전함에 대해 강건함을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
- **직관의 검증:** 시각화 실험을 통해 클래스 중심에서 멀리 떨어진 샘플들이 실제로 일부 신체 부위나 속성이 가려지거나 누락된 '불완전한' 이미지임을 확인하였다. 이는 Prototype Completion이라는 접근 방식의 타당성을 뒷받침한다.
- **상호 보완적 융합:** 1-shot/2-shot에서는 완성된 프로토타입 $\hat{p}_k$가 더 정확하고, 3-shot 이상에서는 평균 기반 $p_k$가 더 정확해지는 경향이 있다. GaussFusion은 이 두 가지의 장점을 베이지안 관점에서 수학적으로 결합하여 모든 Shot 설정에서 안정적인 성능을 낸다.

### 2. 한계 및 논의사항
- **외부 지식 의존성:** WordNet과 같은 외부 지식이 필수적이며, 이러한 지식이 전혀 없는 도메인에서는 적용이 어렵다.
- **가정의 단순함:** 프로토타입 분포를 다변량 가우시안 분포로 가정하였으나, 실제 특징 공간의 분포가 더 복잡할 경우 성능 한계가 있을 수 있다.

## 📌 TL;DR

본 논문은 FSL에서 Novel 클래스의 샘플들이 특징 공간에서 큰 분산을 가지며 '불완전'하다는 점을 발견하고, 이를 해결하기 위해 **외부 속성 지식을 활용해 프로토타입을 보완하는 Prototype Completion 프레임워크**를 제안한다. PATNet으로 속성 특징을 추론하고, ProtoComNet으로 프로토타입을 완성하며, GaussFusion으로 평균 기반 프로토타입과 융합함으로써, 특히 데이터가 극도로 부족한 1-shot 설정에서 매우 강력한 성능을 달성하였다. 이 연구는 향후 외부 지식을 시각적 보완에 활용하는 FSL 연구에 중요한 이정표가 될 것으로 보인다.