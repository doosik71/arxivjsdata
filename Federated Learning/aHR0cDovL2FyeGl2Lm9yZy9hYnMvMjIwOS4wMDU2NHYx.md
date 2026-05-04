# Federated Quantum Natural Gradient Descent for Quantum Federated Learning

Jun Qi(2022)

## 🧩 Problem to Solve

본 논문은 Quantum Federated Learning (QFL) 환경에서 발생하는 통신 오버헤드 문제를 해결하고자 한다. QFL은 여러 로컬 양자 장치에 분산된 학습 아키텍처를 가지는데, 이때 로컬 장치와 중앙 서버 간의 통신 비용을 최소화하기 위해서는 학습 알고리즘의 효율성을 높여 수렴 속도를 가속화하는 것이 필수적이다.

특히, 현재의 Near-term Intermediate-Scale Quantum (NISQ) 장치들은 물리적 큐빗의 수가 제한적이며 양자 오류 정정 능력이 부족하여 표현력(representation power)이 상당히 제한적이라는 하드웨어적 한계를 가지고 있다. 따라서 제한된 자원을 가진 NISQ 장치들을 활용한 분산 학습 시스템에서, 적은 횟수의 글로벌 모델 업데이트만으로도 빠르게 수렴할 수 있는 효율적인 최적화 알고리즘을 설계하는 것이 본 연구의 핵심 목표이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 단일 변분 양자 회로(Variational Quantum Circuit, VQC) 학습에 사용되는 Quantum Natural Gradient Descent (QNGD)를 연합 학습 환경으로 확장한 Federated Quantum Natural Gradient Descent (FQNGD) 알고리즘을 제안한 것이다.

FQNGD의 중심 아이디어는 파라미터 공간의 단순한 유클리드 기하학적 거리 대신, 양자 상태의 분포 공간에서의 거리를 측정하는 Fubini-Study metric tensor를 도입하는 것이다. 이를 통해 파라미터화 방식에 관계없이 최적의 업데이트 단계 크기를 선택할 수 있으며, 결과적으로 기존의 Stochastic Gradient Descent (SGD) 계열 알고리즘보다 훨씬 적은 학습 반복 횟수로 모델을 수렴시켜 전체 통신 비용을 획기적으로 줄일 수 있다.

## 📎 Related Works

기존의 연합 학습(FL) 연구들은 분산 컴퓨팅 시스템의 통신 효율성을 높이는 전략에 집중해 왔으며, 최근에는 이를 양자 컴퓨팅에 접목한 QFL 아키텍처 연구가 등장하였다. 특히 Chen et al.은 고전적 FL 패러다임을 기반으로 한 QFL 구조를 제시한 바 있다.

또한, 단일 VQC를 학습시키기 위한 최적화 방법으로 QNGD가 제안되었으며, 이는 Fubini-Study metric tensor를 사용하여 학습 효율을 높이는 방식이다. 본 논문은 이러한 QNGD의 특성을 FL 환경으로 확장하였다는 점에서 기존 연구와 차별화된다. 특히, 기존 QFL 연구들이 로컬 장치의 VQC 파라미터 자체를 전송했다면, 본 논문은 VQC의 그래디언트(gradient)를 전송하여 글로벌 모델을 업데이트함으로써 데이터 프라이버시와 효율성을 동시에 고려하였다.

## 🛠️ Methodology

### 전체 시스템 구조

본 연구의 QFL 시스템은 중앙의 글로벌 VQC 모델과 $M$개의 로컬 VQC 모델로 구성된다. 학습 과정은 다음의 세 단계로 이루어진다.

1. 중앙 서버가 글로벌 VQC 파라미터 $\bar{\theta}$를 $K$개의 로컬 참여자에게 전송한다.
2. 각 참여자는 자신의 로컬 데이터를 사용하여 VQC를 학습시키고, 계산된 모델 그래디언트 $\nabla L(\theta^k)$를 서버로 업로드한다.
3. 서버는 수집된 그래디언트들을 집계하여 글로벌 모델 파라미터 $\bar{\theta}$를 업데이트한다.

### Variational Quantum Circuit (VQC) 구성

VQC는 다음의 세 가지 주요 구성 요소로 이루어져 있다.

- **Tensor Product Encoding (TPE):** 고전적 입력 벡터 $x$를 양자 상태 $|x\rangle$로 변환하는 과정으로, $\text{R}_Y(\frac{\pi}{2}x_i)$ 게이트를 사용하여 일대일 매핑을 수행한다.
- **Parametric Quantum Circuit (PQC):** 양자 얽힘을 생성하는 CNOT 게이트와 학습 가능한 파라미터 $\alpha, \beta, \gamma$를 가진 단일 큐빗 회전 게이트 $\text{R}_X, \text{R}_Y, \text{R}_Z$로 구성된다.
- **Measurement:** 최종 양자 상태에서 기댓값 $\langle z \rangle$를 측정하여 고전적인 출력값으로 변환하고, 이를 손실 함수에 적용한다.

### FQNGD 알고리즘 및 수학적 원리

표준 SGD는 유클리드 공간에서 다음과 같이 업데이트를 수행한다.
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

반면, FQNGD는 분포 공간에서의 최속 강하법을 구현하기 위해 Fubini-Study metric tensor의 유사 역행렬(pseudo-inverse)인 $g^+(\theta)$를 도입한다. FQNGD의 글로벌 업데이트 식은 다음과 같다.
$$\bar{\theta}_{t+1} = \bar{\theta}_t - \eta \sum_{k=1}^K \frac{N_k}{N} g^+_k(\theta^{(k)}_t) \nabla L(\theta^{(k)}_t)$$
여기서 $\bar{\theta}$는 글로벌 파라미터, $\theta^{(k)}$는 $k$번째 참여자의 로컬 파라미터, $N_k$는 참여자의 데이터 수, $N$은 전체 데이터 수이다.

NISQ 장치에서의 계산 복잡도를 줄이기 위해, 본 논문은 Fubini-Study metric tensor를 블록 대각 행렬(block-diagonal approximation) 형태로 근사하여 사용한다. 각 파라미터 층 $l$에 대해 하미토니안 생성자 $H$를 이용한 메트릭 텐서의 성분은 다음과 같이 계산된다.
$$g^+_{l,i,j} = \langle \psi_l | H_l(i) H_l(j) | \psi_l \rangle - \langle \psi_l | H_l(i) | \psi_l \rangle \langle \psi_l | H_l(j) | \psi_l \rangle$$

## 📊 Results

### 실험 설정

- **데이터셋:** MNIST 데이터셋을 사용하여 이진 분류(숫자 {2, 5}) 및 삼진 분류(숫자 {1, 3, 7}) 작업을 수행하였다.
- **비교 대상:** Naive SGD, Adagrad, Adam 최적화 도구와 FQNGD를 비교하였다.
- **환경:** 6개의 동일한 로컬 VQC 참여자로 구성된 QFL 시스템을 시뮬레이션하였다.
- **평가 지표:** 분류 정확도(Accuracy)와 학습 곡선을 통한 수렴 속도를 측정하였다.

### 정량적 결과

이진 분류와 삼진 분류 모두에서 FQNGD가 가장 높은 정확도를 기록하였다.

- **이진 분류 정확도:** FQNGD($99.32\%$) $>$ Adam($98.87\%$) $>$ Adagrad($98.81\%$) $>$ SGD($98.48\%$)
- **삼진 분류 정확도:** FQNGD($99.12\%$) $>$ Adam($98.71\%$) $>$ Adagrad($98.63\%$) $>$ SGD($97.86\%$)

### 정성적 결과

학습 곡선 분석 결과, FQNGD는 다른 최적화 방법들에 비해 월등히 빠른 수렴 속도를 보였다. 이는 동일한 목표 정확도에 도달하기 위해 필요한 글로벌 모델 업데이트 횟수가 훨씬 적음을 의미하며, 결과적으로 QFL 시스템의 치명적인 문제인 통신 비용을 유의미하게 낮출 수 있음을 시사한다.

## 🧠 Insights & Discussion

본 논문은 양자 상태의 기하학적 구조를 반영한 Fubini-Study metric tensor를 연합 학습에 도입함으로써, 단순한 학습률 조정을 넘어선 본질적인 최적화 효율성을 달성하였다. 특히 VQC 모델의 특성에 맞게 설계된 $g^+(\theta)$가 Adam이나 Adagrad 같은 일반적인 적응형 학습률 알고리즘보다 더 우수한 성능을 보였다는 점은 고무적이다.

다만, 본 연구의 한계점과 논의가 필요한 사항은 다음과 같다.
첫째, 실험이 시뮬레이션 환경에서 진행되었으므로, 실제 NISQ 장치에서 발생할 수 있는 양자 노이즈와 디코히어런스(decoherence)가 FQNGD의 수렴 속도와 정확도에 미치는 영향이 명시적으로 분석되지 않았다.
둘째, 대규모 데이터셋에 대한 확장성 검증이 부족하며, 실제 분산 환경에서의 보안 위협(악의적인 참여자의 공격 등)에 대한 방어 기제는 고려되지 않았다.
셋째, 제안된 알고리즘이 VQC 외에 Quantum Convolutional Neural Networks (QCNN)와 같은 다른 양자 신경망 구조에서도 동일한 효율성을 보일지는 추가적인 연구가 필요하다.

## 📌 TL;DR

본 논문은 양자 연합 학습(QFL)의 통신 비용을 줄이기 위해, Fubini-Study metric tensor를 이용한 **Federated Quantum Natural Gradient Descent (FQNGD)** 알고리즘을 제안하였다. MNIST 데이터셋 실험을 통해 FQNGD가 기존 SGD, Adam, Adagrad보다 **빠른 수렴 속도**와 **높은 분류 정확도**를 보임을 입증하였다. 이 연구는 향후 NISQ 장치를 활용한 실전적 양자 분산 학습 시스템 구축 및 통신 효율 최적화 연구에 중요한 기초를 제공할 가능성이 크다.
