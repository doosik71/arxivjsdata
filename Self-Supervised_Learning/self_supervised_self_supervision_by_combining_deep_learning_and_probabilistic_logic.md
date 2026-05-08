# Self-supervised self-supervision by combining deep learning and probabilistic logic

Hunter Lang, Hoifung Poon (2020)

## 🧩 Problem to Solve

머신러닝, 특히 딥러닝 모델의 성능을 높이기 위해서는 대규모의 레이블링된 데이터가 필수적이다. 그러나 생의학(biomedicine)과 같이 전문 지식이 필요한 도메인에서는 데이터 레이블링 비용이 매우 높고 시간 소모적이며, 크라우드소싱을 적용하기 어렵다는 문제가 있다.

이를 해결하기 위해 기존에는 Prior Knowledge를 활용하여 노이즈가 포함된 레이블을 자동으로 생성하는 Self-supervision 방법론들이 제안되었다. 하지만 이러한 방법론들, 특히 Deep Probabilistic Logic (DPL)이나 Data Programming과 같은 프레임워크는 여전히 전문가가 직접 Self-supervision 템플릿(규칙)을 설계해야 한다는 한계가 있다. 높은 정확도를 얻기 위해 수많은 가상 증거(Virtual Evidence)를 수동으로 찾아내고 정의하는 과정은 매우 지루하고 도전적인 작업이다.

본 논문의 목표는 전문가의 수동 설계 부담을 최소화하기 위해, 모델이 스스로 새로운 Self-supervision 규칙을 제안하고 학습할 수 있는 **Self-Supervised Self-Supervision (S4)** 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존의 DPL 프레임워크에 **구조 학습(Structure Learning)**과 **능동 학습(Active Learning)** 능력을 결합하여 Self-supervision의 자동화를 달성하는 것이다.

- **Structured Self-Training (SST):** 학습된 딥 뉴럴 네트워크의 어텐션(Attention)이나 엔트로피(Entropy)를 분석하여, 모델이 스스로 유용한 새로운 Self-supervision 규칙을 제안하고 이를 시스템에 직접 추가한다.
- **Feature-based Active Learning (FAL):** 모델이 불확실성이 높다고 판단하는 규칙을 선정하여 인간 전문가에게 검증을 요청함으로써, 적은 비용으로 고품질의 Self-supervision을 확보한다.
- **통합 프레임워크:** 초기 시드(Seed) 규칙에서 시작하여 SST와 FAL을 반복적으로 수행함으로써, 사람이 직접 개입하는 노력은 최소화하면서도 지도 학습(Supervised Learning)에 근접하는 성능을 달성한다.

## 📎 Related Works

본 연구는 다음과 같은 기존 접근 방식들을 계승하고 차별화한다.

- **Deep Probabilistic Logic (DPL):** 딥러닝의 표현력과 확률적 로직(Probabilistic Logic)의 추론 능력을 결합한 프레임워크이다. 레이블을 잠재 변수로 처리하고 가상 증거를 통해 Prior Belief를 통합하지만, 규칙을 수동으로 정의해야 한다는 점이 한계였다.
- **Data Programming (e.g., Snorkel):** 레이블링 함수(Labeling Functions)를 통해 대량의 노이즈 레이블을 생성한다. 하지만 이는 주로 개별 인스턴스에 대한 Prior Belief만을 다루며, 인스턴스 간의 관계를 다루는 Joint Inference 능력이 부족하다.
- **Self-training:** 모델이 예측한 확신도가 높은 레이블을 다시 학습 데이터로 사용하는 방식이다. S4는 이를 일반화하여 단순 레이블뿐만 아니라 임의의 마르코프 로직(Markov Logic) 공식을 가상 증거로 추가하는 '구조적' 셀프 트레이닝을 수행한다.

## 🛠️ Methodology

### 1. Deep Probabilistic Logic (DPL) 기초

S4의 기반이 되는 DPL은 입력 $X$에 대해 출력 $Y$를 예측하는 모듈 $\Psi(x,y)$와 Self-supervision을 나타내는 가상 증거 $V$의 집합을 결합한다. 전체 확률 분포는 다음과 같이 정의된다.

$$P(K,Y|X) \propto \prod_{v} \Phi_{v}(X,Y) \cdot \prod_{i} \Psi(X_{i}, Y_{i})$$

여기서 $\Phi_{v}(X,Y)$는 가중치가 부여된 1차 논리 공식(First-order logic formulas)으로, $P(K|Y,X)$를 나타내는 잠재적 포텐셜 함수이다. $\Psi(X_{i}, Y_{i})$는 딥 뉴럴 네트워크(예: BERT)가 예측하는 $P(Y|X)$이다.

**학습 절차 (Variational EM):**

- **E-step:** Loopy Belief Propagation을 사용하여 잠재 레이블 $Y$에 대한 변분 근사 $\text{q}(Y)$를 계산한다.
- **M-step:** $\text{q}(Y)$를 확률적 레이블로 사용하여 $\Psi$를 표준 딥러닝 방식으로 학습시키고, 동시에 $\Phi$의 가중치 $w_{v}$를 최적화한다.

### 2. S4 Framework (Self-Supervised Self-Supervision)

S4는 DPL의 루프에 구조 학습과 능동 학습을 추가한다.

#### (1) Structured Self-Training (SST)

SST는 모델이 스스로 새로운 규칙 $v$를 제안하는 과정이다.

- **Attention-based Scoring:** 뉴럴 네트워크 $\Psi$의 어텐션 가중치를 활용한다. 특정 토큰 $t$가 특정 레이블 $l$과 강하게 연관되어 있다면, 이를 새로운 가상 증거로 제안한다.
  $$S_{\text{token}}(t,l) = \text{Attn}(t,l) - \sum_{l' \neq l} \text{Attn}(t,l')$$
- **Entropy-based Scoring:** 특정 특징 $b$를 가진 인스턴스들의 평균 사후 확률 분포의 엔트로피 $\text{Ent}(b)$를 계산하여, 엔트로피가 매우 낮은(즉, 확신이 강한) 특징을 선택한다.
- **Joint-inference Factors:** 현재 학습된 BERT 모델의 임베딩 유사도와 사전 학습된(Pre-trained) 모델의 유사도 차이를 계산하여, Task-specific한 유사성을 가진 인스턴스 쌍을 찾아내고 "두 인스턴스는 같은 레이블을 가질 것"이라는 규칙을 추가한다.

#### (2) Feature-based Active Learning (FAL)

SST와 반대로, 모델이 가장 불확실해하는(평균 엔트로피가 가장 높은) 특징 $b$를 선택하여 인간 전문가에게 해당 특징의 레이블이 무엇인지 묻는다. 전문가는 이를 수락(Accept)하거나 거부(Reject)함으로써 고품질의 규칙을 시스템에 주입한다.

### 3. 전체 알고리즘 흐름

1. 초기 시드 규칙 $I$로 DPL 학습을 시작한다.
2. SST가 수렴할 때까지 $\Psi$와 $K$를 업데이트하고 새로운 규칙을 자동으로 추가한다.
3. 정해진 쿼리 예산 $T$ 내에서 FAL을 통해 전문가의 검증을 거친 규칙을 추가한다.
4. 이 과정을 반복하여 최종 예측 모듈 $\Psi$를 완성한다.

## 📊 Results

### 실험 설정

- **데이터셋:** IMDb (영화 리뷰 이진 분류), Stanford Sentiment Treebank (StanSent, 이진화 버전), Yahoo! Answers (10개 클래스 분류).
- **모델:** BERT-base 및 Global-context attention layer 사용.
- **비교 대상:** BoW, Fully Supervised DNN, Self-training, Snorkel, DPL.

### 주요 결과

- **IMDb 데이터셋:**
  - 초기 규칙 6개($|I|=6$)만으로 S4-SST는 86%의 정확도를 달성했다. 이는 100개의 레이블된 데이터를 사용한 Self-training(69.9%)보다 훨씬 높은 수치이다.
  - DPL 대비 약 5%p, Snorkel 대비 약 8.9%p의 성능 향상을 보였다.
- **Stanford Sentiment Treebank:**
  - 더 복잡한 데이터셋임에도 불구하고, S4-SST는 초기 규칙 20개만으로 80% 정확도의 벽을 넘었다.
  - 특히 Joint-inference factor를 추가했을 때(S4-SST + J) 성능이 추가로 상승하였으며, 복잡한 의미론적 관계를 자동으로 유도해냄을 확인하였다.
- **Yahoo! Answers:**
  - 10개 클래스의 다중 분류 작업에서도 S4-SST는 DPL보다 약 11%p, Snorkel보다 약 16%p 높은 정확도를 기록하며 노이즈에 강건한 모습을 보였다.

### 분석

- **Robustness:** 초기 시드 규칙을 무작위로 설정했을 때도 S4-SST는 반복적인 구조 학습을 통해 성능을 회복하며 최종적으로 높은 정확도에 도달했다.
- **Efficiency:** 아주 적은 양의 인간 개입(시드 규칙 생성 및 소수의 쿼리 응답)만으로도 전체 지도 학습 데이터셋을 사용한 모델의 성능에 근접했다.

## 🧠 Insights & Discussion

### 강점

- **효율적인 인간 자원 활용:** 전문가가 모든 규칙을 설계하는 대신, "가장 중요한 규칙만 먼저 정의"하고 이후에는 "제안된 규칙을 검증"하는 방식으로 업무 부하를 획기적으로 줄였다.
- **심볼릭 로직과 딥러닝의 시너지:** 딥러닝의 특징 추출 능력(Attention)과 확률적 로직의 전파 능력(Joint Inference)을 결합하여, 단순한 패턴 매칭을 넘어선 구조적 학습을 가능하게 했다.
- **노이즈 강건성:** 잘못된 초기 규칙이 있더라도 반복적인 EM 과정과 구조 학습을 통해 이를 정제하고 더 정확한 규칙으로 대체하는 능력을 보였다.

### 한계 및 논의사항

- **계산 비용:** 반복적인 Variational EM과 구조 탐색 과정이 포함되어 있어, 단순 학습보다 계산 시간이 더 소요될 수 있다.
- **규칙 클래스의 제한:** 본 논문에서는 주로 토큰 기반의 단항 포텐셜과 유사도 기반의 이항 포텐셜에 집중했다. 더 복잡한 형태의 논리 공식(Higher-order factors)을 자동으로 유도하는 방법은 여전히 과제로 남아 있다.
- **검증 예산의 영향:** FAL에서 인간의 쿼리 예산 $T$가 너무 많거나 초기 시드가 충분할 경우, 오히려 능동 학습이 성능에 미미한 영향을 주거나 약간의 저하를 일으키는 경우가 관찰되었다.

## 📌 TL;DR

본 논문은 딥러닝과 확률적 로직을 결합한 DPL을 확장하여, **Self-supervision 규칙 자체를 스스로 학습하는 S4 프레임워크**를 제안한다. S4는 모델의 어텐션과 엔트로피를 이용해 새로운 규칙을 제안하는 **Structured Self-Training**과 전문가의 검증을 받는 **Feature-based Active Learning**을 결합한다. 실험 결과, S4는 매우 적은 양의 초기 지식만으로도 기존의 Self-supervision 방법론(Snorkel, DPL)을 크게 상회하며, 지도 학습 수준의 성능에 도달할 수 있음을 입증하였다. 이는 데이터 레이블링 비용이 극심한 전문 도메인에서 매우 실용적인 해결책이 될 가능성이 높다.
