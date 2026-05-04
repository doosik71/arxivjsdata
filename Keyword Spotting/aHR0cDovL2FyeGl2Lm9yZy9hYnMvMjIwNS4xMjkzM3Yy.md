# Boosting Tail Neural Network for Realtime Custom Keyword Spotting

Sihao Xue, Qianyao Shen, Guoqing Li (2023)

## 🧩 Problem to Solve

본 논문은 실시간 사용자 정의 키워드 스포팅(Realtime Custom Keyword Spotting, RCKS) 시스템이 직면한 산업적 과제를 해결하고자 한다. 일반적인 키워드 스포팅(KWS)은 고정된 키워드를 사용하므로 최적화가 용이하지만, RCKS는 사용자가 임의로 키워드를 설정할 수 있어야 하므로 모든 음향 상태(acoustic states)를 식별할 수 있는 강력한 분류 능력이 요구된다.

특히 이러한 시스템은 주로 자동차와 같은 저사양 연산 자원을 가진 온디바이스(On-Device) 환경에서 동작해야 한다. 따라서 낮은 연산 비용과 지연 시간(latency)을 유지하면서도, 높은 깨움률(wakeup rate)과 낮은 오경보율(false alarm rate)을 동시에 달성하는 것이 핵심적인 목표이다. RCKS는 단어 또는 짧은 구절만을 다루기에 언어 모델의 도움을 거의 받을 수 없으며, 결과적으로 음향 모델(acoustic model)의 성능이 결정적인 역할을 하게 된다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 뇌 과학의 부분 활성화 원리와 머신러닝의 부스팅(Boosting) 개념을 결합한 **Boosting Tail Neural Network (BTNN)** 아키텍처를 제안하는 것이다.

중심적인 설계 직관은 하나의 거대한 강한 분류기(strong classifier)를 사용하는 대신, 공유된 특징 추출기(feature embedding extractor)와 다수의 약한 이진 분류기(weak binary classifiers, 'Tail' networks)를 조합하는 것이다. 각 Tail 네트워크는 특정 음향 상태만을 전문적으로 판별하며, 추론 시에는 디코딩 과정에서 필요한 특정 상태의 Tail 네트워크만 활성화함으로써 실제 연산량을 획기적으로 줄일 수 있다.

## 📎 Related Works

기존의 KWS 연구들은 프로젝트 출시 전에 키워드가 결정된 환경에서 타겟 최적화를 수행하여 효율성을 높였다. 하지만 RCKS와 같이 유연성이 필요한 환경에서는 이러한 방식이 적용 불가능하다. 기존의 RCKS 알고리즘들은 낮은 연산 비용과 낮은 지연 시간이라는 제약 조건 하에서 높은 음향 분류 정확도를 달성하는 데 한계가 있었다. 본 논문은 이를 해결하기 위해 특징 임베딩(feature embedding)과 독립적인 이진 분류기 구조를 통해 성능과 효율성의 트레이드-오프를 최적화하고자 한다.

## 🛠️ Methodology

### 1. 시스템 구조 및 파이프라인
BTNN은 크게 **Feature Embedding** 부분과 **Tail Neural Network** 부분으로 구성된다.

- **Feature Embedding**: 입력 음향 신호를 추상적인 특징 공간으로 변환하는 비선형 변환 단계이다. 본 논문에서는 DFSMN(Deep-FSMN) 구조를 사용하며, 깊은 은닉층의 출력을 특징 임베딩으로 간주한다.
- **Tail Neural Network**: 임베딩된 특징을 입력으로 받아 특정 음향 상태인지를 판별하는 다수의 이진 분류기이다. 각 Tail 네트워크는 $128 \times 64 \times 32 \times 1$ 크기의 피라미드 구조(pyramid architecture)를 가진 피드포워드 모델이다.

### 2. 학습 목표 및 손실 함수
각 Tail 분류기는 독립적으로 학습되며, 각 음향 상태에 대해 평균 제곱 오차(MSE) 손실 함수를 사용한다. 긍정 샘플과 부정 샘플의 불균형을 해소하기 위해 스케일 계수 $S_i$를 적용한다.

$$loss_i = \begin{cases} MSE(o_{s_i}, 0) & \text{for negative samples} \\ MSE(o_{s_i}, 1) \times S_i & \text{for positive samples} \end{cases}$$

### 3. 음향 후처리 (Acoustic Post-processing)
MSE 손실을 사용하므로 모델의 출력값은 확률값이 아니다. 이를 확률로 변환하기 위해 개발 데이터셋을 통해 출력값의 통계적 분포를 분석한다. 

분포를 여러 구간으로 나누어 경계 확률 $P_n$을 계산하며, 다음과 같은 수식을 통해 확률을 산출한다.
$$P_n = \begin{cases} 0 & \text{for } n = 0 \\ 1 & \text{for } n = N \\ \frac{P_{n+1} - C_{n+1}}{C_{total}} & \text{for } 1 \le n \le N-1 \end{cases}$$

최종 확률 $p_s(x)$는 긍정 확률 $p_{s,p}(x)$와 부정 확률 $p_{s,n}(x)$를 기하 분포 형태로 결합하여 계산하며, 이때 상태별 변별력을 조절하는 스케일링 계수 $S_{s,p}$와 $S_{s,n}$이 사용된다.

$$p_s(x) = \left( \frac{p_{s,p}(x)}{S_{s,p}} \times \frac{p_{s,n}(x)}{S_{s,n}} \right)^{\frac{1}{S_{s,p} + S_{s,n}}}$$

### 4. 디코딩 절차
디코딩에는 Token Push 방식과 가중 유한 상태 트랜스듀서(WFST)가 사용된다. 특히 음향적으로 일부 프레임이 잘못 계산되어 인식이 실패하는 상황을 방지하기 위해 **Jump Arc**를 도입하였으며, 오경보를 줄이기 위해 Jump Arc에 패널티 점수를 부여한다.

추론 시에는 현재 활성화된 토큰(active token)에서 필요한 음향 상태만을 식별하고 해당 상태의 Tail 네트워크만 활성화하여 연산 효율을 극대화한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 3,000시간의 차량 내부 데이터 및 시뮬레이션된 소음(AEC simulation 등) 데이터 사용.
- **테스트 데이터**: 40개의 키워드가 포함된 5,000개의 긍정 발화와 키워드가 없는 50시간의 부정 데이터.
- **비교 대상**: 8-layer DFSMN, 10-layer DFSMN.
- **평가 지표**: 24시간당 오경보 1회(1 false alarm per 24 hours) 수준에서의 깨움률(Wakeup Rate).

### 정량적 결과
실험 결과, 제안된 BTNN 방식이 기존의 단일 강한 분류기 모델보다 우수한 성능을 보였다.

| Model | Wakeup Rate | 비고 |
| :--- | :---: | :--- |
| 8-layer FSMN | 89.14% | Base model |
| 10-layer FSMN | 90.40% | 연산량 증가 모델 |
| csBTNN (Constant Scale) | 91.10% | 8-layer 대비 약 18% 상대적 향상 |
| asBTNN (Adaption Scale) | 91.64% | 최적의 성능 |

특히 `asBTNN`은 상태별로 최적화된 스케일 계수를 적용함으로써 가장 높은 깨움률을 달성하였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 통찰은 모델의 전체 파라미터 크기가 커지더라도, 추론 시 필요한 부분만 활성화하는 전략을 통해 **실제 프레임당 연산량을 낮출 수 있다**는 점이다. 이는 인간의 뇌가 모든 신경망을 동시에 활성화하지 않는 방식과 유사하다.

또한, Softmax 기반 모델은 각 상태의 확률 합이 1이어야 하므로 상태 간 배타적인 관계를 강제하지만, BTNN의 독립적 이진 분류 구조는 **상태 공간의 중첩(overlapping)**을 허용한다. 이는 발음의 유사성이나 억양의 차이로 인한 음향적 혼동을 더 유연하게 처리할 수 있게 하며, 특히 늘어지는 발음(stretched voice)과 같은 상황에서 Softmax보다 안정적인 결과를 낸다는 강점이 있다.

다만, 모델 전체의 저장 용량(model size)은 증가하므로 메모리 제약이 극심한 환경에서의 저장 공간 최적화 문제는 향후 고려해야 할 사항으로 보인다.

## 📌 TL;DR

본 논문은 저사양 온디바이스 환경에서 효율적인 실시간 사용자 정의 키워드 스포팅(RCKS)을 구현하기 위해 **Boosting Tail Neural Network (BTNN)**를 제안한다. 공유 임베딩 층과 상태별 독립적인 Tail 네트워크 구조를 통해, 연산 효율성을 확보함과 동시에 음향 상태의 중첩을 허용함으로써 인식 성능을 향상시켰다. 실험 결과, 기존 FSMN 기반 모델 대비 깨움률을 유의미하게 높였으며, 이는 향후 온디바이스 ASR 및 KWS 시스템의 효율적인 설계에 중요한 방향성을 제시한다.